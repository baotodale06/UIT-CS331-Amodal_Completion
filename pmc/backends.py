"""Inference backend của PMC: import chậm và weights khởi tạo ở CPU."""
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
from .runtime import event, stage


ROOT = Path(__file__).resolve().parents[1]


def prepare_imports():
    # Ưu tiên fork diffusers trong repo để giữ mixed-context pipeline của PMC.
    for path in (ROOT / "Grounded-Segment-Anything", ROOT / "Grounded-Segment-Anything/GroundingDINO",
                 ROOT / "lama", ROOT / "InstaOrder", ROOT / "diffusers/src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class SamSession:
    """SamPredictor không có .to(); adapter xóa embedding trước khi offload weights."""
    def __init__(self, checkpoint):
        from segment_anything import build_sam, SamPredictor
        self.predictor = SamPredictor(build_sam(checkpoint=checkpoint).eval())

    def to(self, device):
        self.predictor.model.to(device)
        return self

    def reset_image(self):
        self.predictor.reset_image()


def register_backends(manager, args):
    prepare_imports()
    torch = manager.torch

    def dino_factory():
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        config = SLConfig.fromfile(args.gdino_config)
        config.device = "cpu"
        model = build_model(config)
        state = torch.load(args.gdino_ckpt, map_location="cpu", weights_only=True)
        result = model.load_state_dict(clean_state_dict(state["model"]), strict=False)
        event("dino_checkpoint_loaded", missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys)
        return model.eval()

    def order_factory():
        # Bỏ training wrapper vốn gọi cuda() và tạo optimizer ngay trong constructor.
        # Backbone và weights inference vẫn là InstaOrderNet_od gốc.
        from models.backbone import resnet50_cls
        model = resnet50_cls(in_channels=5, num_classes=[2, 3])
        state = torch.load(args.instaorder_ckpt, map_location="cpu", weights_only=True)["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        return model.eval()

    def lama_factory():
        from omegaconf import OmegaConf
        from saicinpainting.training.modules import make_generator
        config = OmegaConf.load(args.lama_config_path)
        if config.training_model.get('add_noise_kwargs'):
            raise ValueError('LaMa adapter hiện hỗ trợ big-lama chuẩn, không hỗ trợ noise augmentation')
        generator = make_generator(config, **config.generator)
        checkpoint = torch.load(args.lama_ckpt_path, map_location='cpu', weights_only=False)
        weights = {key[len('generator.'):]: value for key, value in checkpoint['state_dict'].items()
                   if key.startswith('generator.')}
        generator.load_state_dict(weights, strict=True)

        class LamaInference(torch.nn.Module):
            # Tái hiện forward predict_only của LaMa, bỏ training/evaluator dependencies.
            def __init__(self):
                super().__init__()
                self.generator = generator

            def forward(self, batch):
                image, mask = batch['image'], batch['mask']
                masked = image * (1-mask)
                if config.training_model.get('concat_mask', True):
                    masked = torch.cat([masked, mask], dim=1)
                predicted = self.generator(masked)
                if predicted.shape != image.shape:
                    predicted = torch.nn.functional.interpolate(predicted, image.shape[-2:], mode='bilinear', align_corners=False)
                return {'inpainted': mask*predicted+(1-mask)*image}

        return LamaInference().eval().requires_grad_(False)

    def diffusion_factory():
        import diffusers
        from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintMixedContextPipeline
        pipeline = StableDiffusionInpaintMixedContextPipeline if args.mc_timestep is not None else StableDiffusionInpaintPipeline
        event("diffusers_source", module=diffusers.__file__, version=diffusers.__version__)
        model = pipeline.from_pretrained(args.sd_model, torch_dtype=torch.float16)
        model.enable_attention_slicing()
        return model

    for name, factory in (("dino", dino_factory), ("sam", lambda: SamSession(args.sam_ckpt)),
                          ("instaorder", order_factory), ("lama", lama_factory),
                          ("diffusion", diffusion_factory)):
        manager.register(name, factory)


class InferenceEngine:
    def __init__(self, manager, args):
        self.manager, self.args = manager, args

    def segment(self, images, classes):
        """DINO cho cả pool → offload → SAM cho cả pool, tránh đổi model mỗi ảnh."""
        import groundingdino.datasets.transforms as T
        from groundingdino.util.utils import get_phrases_from_posmap
        torch = self.manager.torch
        transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        boxes_list, metadata = [], []
        caption = ". ".join(classes).lower().strip().rstrip(".") + "."
        with stage("grounding_dino", images=len(images)):
            with self.manager.use("dino") as model:
                for index, image in enumerate(images):
                    tensor, _ = transform(image, None)
                    output = model(tensor[None].to(self.manager.device), captions=[caption])
                    logits = output["pred_logits"][0].sigmoid().cpu()
                    boxes = output["pred_boxes"][0].cpu()
                    keep = logits.max(dim=1).values > self.args.box_threshold
                    logits, boxes = logits[keep], boxes[keep]
                    names, scores = [], []
                    for logit in logits:
                        names.append(get_phrases_from_posmap(logit > self.args.text_threshold,
                                                            model.tokenizer(caption), model.tokenizer))
                        scores.append(float(logit.max()))
                    # DINO trả normalized cxcywh; SAM nhận pixel xyxy.
                    w, h = image.size
                    boxes = boxes * torch.tensor([w, h, w, h])
                    boxes[:, :2] -= boxes[:, 2:] / 2
                    boxes[:, 2:] += boxes[:, :2]
                    boxes_list.append(boxes)
                    metadata.append((names, scores))
                    event("detection_result", candidate_index=index, detections=len(boxes))
                    del output, tensor
        masks_list = [None] * len(images)
        if any(len(boxes) for boxes in boxes_list):
            with stage("sam_segmentation", images=len(images)):
                with self.manager.use("sam") as session:
                    predictor = session.predictor
                    for index, (image, boxes) in enumerate(zip(images, boxes_list)):
                        if not len(boxes):
                            continue
                        predictor.set_image(np.asarray(image))
                        # Chunk boxes để nhiều detections không gây peak VRAM quá lớn.
                        chunks = []
                        for chunk in boxes.split(self.args.sam_box_batch_size):
                            transformed = predictor.transform.apply_boxes_torch(chunk, np.asarray(image).shape[:2])
                            masks, quality, low_res = predictor.predict_torch(
                                point_coords=None, point_labels=None, boxes=transformed.to(self.manager.device),
                                multimask_output=False)
                            chunks.append(masks[:, 0].cpu().numpy().astype(bool))
                            del masks, quality, low_res, transformed
                        masks_list[index] = np.concatenate(chunks)
                        predictor.reset_image()
        return [(mask, names, scores) for mask, (names, scores) in zip(masks_list, metadata)]

    def occluders(self, image, masks, target_index, bboxes):
        if len(masks) < 2:
            return []
        import inference as infer
        with stage("instaorder", instances=len(masks)):
            with self.manager.use("instaorder") as model:
                occ_order, _ = infer.infer_order_sup_occ_depth(
                    SimpleNamespace(model=model), np.asarray(image), np.asarray(masks, dtype=np.float32),
                    bboxes, pairs="all", method="InstaOrderNet_od", patch_or_image="resize",
                    input_size=384, disp_select_method="")
        return [masks[i] for i in np.where(occ_order.T[target_index] == 1)[0] if i != target_index]

    def remove_object(self, image, mask):
        """LaMa chỉ chuẩn bị một background/iteration; output không giữ CUDA tensor."""
        import torch
        with stage("lama_background"):
            with self.manager.use("lama") as model:
                batch = {"image": torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].float().div_(255).to(self.manager.device),
                         "mask": torch.from_numpy(np.array(mask, dtype=np.float32))[None, None].to(self.manager.device)}
                result = model(batch)
                rgb = result["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
                del result, batch
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    def generate(self, inputs, seeds, size, prompt):
        """Giữ diffusion trên GPU cho cả K seed, nhưng chỉ chạy batch_size=1."""
        torch = self.manager.torch
        images = []
        with stage("generate_candidates", count=len(seeds)):
            with self.manager.use("diffusion") as pipeline:
                for index, seed in enumerate(seeds):
                    with stage("diffusion_candidate", candidate_index=index, seed=seed):
                        # CPU generator cho phép tái lập nhiễu, không phụ thuộc RNG stage trước.
                        generator = torch.Generator(device="cpu").manual_seed(seed)
                        result = pipeline(prompt=prompt, generator=generator, height=size, width=size,
                                          num_inference_steps=self.args.num_inference_steps, **inputs)
                        if getattr(result, "nsfw_content_detected", None) and any(result.nsfw_content_detected):
                            raise RuntimeError("Diffusion safety checker đánh dấu candidate; xem log để xử lý")
                        images.append(result.images[0].convert("RGB"))
                        del result
        return images
