"""Generate the visible RGBA inputs used by the open-world paper's evaluator.

Source: saraao/amodal@14d59e09, main.py lines 441-496 and 692-708,
and LISA/app.py lines 220-295. Raw LISA masks are kept for target matching;
the eroded mask is used only for the visible appearance reference.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .dataset import sha256, write_json

LISA_REPO_COMMIT = "3cb2d4301f1af4691bd4f3938335ef06e76f155a"
LISA_MODEL = "xinlai/LISA-13B-llama2-v1"
LISA_REVISION = "b89000be11ad0a45512745a15063f2f6af1d9a5c"
VISION_MODEL = "openai/clip-vit-large-patch14"
VISION_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


def first_crop(mask):
    """Return the released first-iteration crop's origin and size, without a 6x canvas."""
    height, width = mask.shape
    rr, cc = np.where(mask)
    if not len(rr):
        return 0, 0, height, width
    origin_r, origin_c = 3 * height - height // 2, 3 * width - width // 2
    r0, r1 = max(0, int(rr.min()) + origin_r - 60), min(6 * height, int(rr.max()) + origin_r + 60)
    c0, c1 = max(0, int(cc.min()) + origin_c - 60), min(6 * width, int(cc.max()) + origin_c + 60)
    top, bottom = rr.min() <= 10, rr.max() >= height - 10
    left, right = cc.min() <= 10, cc.max() >= width - 10
    r0 -= 150 if top else 0
    r1 += 150 if bottom else 0
    c0 -= 150 if left else 0
    c1 += 150 if right else 0
    target = max(r1 - r0, c1 - c0)
    if c1 - c0 < target:
        diff = target - (c1 - c0)
        if left:
            c0 -= diff
        elif right:
            c1 += diff
        else:
            c0 -= math.floor(diff / 2)
            c1 += math.ceil(diff / 2)
    if r1 - r0 < target:
        diff = target - (r1 - r0)
        if top:
            r0 -= diff
        elif bottom:
            r1 += diff
        else:
            r0 -= math.floor(diff / 2)
            r1 += math.ceil(diff / 2)
    r0, r1 = max(0, r0), min(6 * height, r1)
    c0, c1 = max(0, c0), min(6 * width, c1)
    return r0 - origin_r, c0 - origin_c, r1 - r0, c1 - c0


def visible_rgba(rgb, mask):
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != (height, width):
        raise ValueError("LISA mask must retain original-image dimensions")
    # Erode once with a 5x5 square kernel on the zero-padded mask canvas.
    # MinFilter is the same binary operation as cv2.erode, without importing
    # OpenCV or the inference stack for CPU preparation/tests.
    padded = ImageOps.expand(Image.fromarray(mask.astype(np.uint8) * 255), border=2, fill=0)
    alpha = np.asarray(padded.filter(ImageFilter.MinFilter(5)).crop((2, 2, width + 2, height + 2)))
    full = np.dstack((rgb, alpha))
    row, col, crop_h, crop_w = first_crop(mask)
    crop = np.full((crop_h, crop_w, 4), [255, 255, 255, 0], dtype=np.uint8)
    src_r0, src_c0 = max(row, 0), max(col, 0)
    src_r1, src_c1 = min(row + crop_h, height), min(col + crop_w, width)
    if src_r1 > src_r0 and src_c1 > src_c0:
        crop[src_r0-row:src_r1-row, src_c0-col:src_c1-col] = full[src_r0:src_r1, src_c0:src_c1]
    return full, crop, {"original_size": [width, height], "offset_rc": [row, col]}


def reference_paths(root, row):
    root, relative = Path(root), Path(row["reference"])
    return {"full": root / relative, "crop": root / "crops" / relative,
            "mask": root / "masks" / relative,
            "record": root / "records" / relative.with_suffix(".json")}


def save_reference(root, row, rgb, mask, input_hash, provenance):
    paths = reference_paths(root, row)
    shape = np.asarray(rgb).shape[:2]
    mask = np.zeros(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    full, crop, geometry = visible_rgba(rgb, mask)
    for name, array in [("full", full), ("crop", crop), ("mask", mask.astype(np.uint8) * 255)]:
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(paths[name])
    status = "ready" if full[..., 3].any() else "eroded_empty" if mask.any() else "empty_mask"
    record = {"schema": 1, "id": row["id"], "label": row["label"], "input_sha256": input_hash,
              "status": status, "mask_pixels": int(mask.sum()),
              "eroded_mask_pixels": int((full[..., 3] > 0).sum()), "geometry": geometry,
              "provenance": provenance,
              "artifacts": {name: sha256(paths[name]) for name in ("full", "crop", "mask")}}
    write_json(paths["record"], record)
    return record


def cached_reference(root, row, input_hash):
    paths = reference_paths(root, row)
    if not paths["record"].is_file():
        return None
    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    if record["id"] != row["id"] or record["label"] != row["label"] or record["input_sha256"] != input_hash:
        raise ValueError(f"Visible-reference input/annotation changed for {row['id']}; use a new reference directory")
    for name, expected in record["artifacts"].items():
        if not paths[name].is_file() or sha256(paths[name]) != expected:
            raise ValueError(f"Generated reference artifact missing or changed: {paths[name]}")
    return record


class LisaSegmenter:
    """Batch equivalent of the authors' LISA Gradio app, with pinned weights.

    Uses direct unquantized PyTorch inference. DeepSpeed's optional FP16 kernel
    replacement and the web UI are unnecessary for the segmentation computation.
    """
    def __init__(self, repo, model=LISA_MODEL, precision="fp16", device_map="auto"):
        import sys
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer, CLIPImageProcessor
        repo = Path(repo).resolve()
        sys.path.insert(0, str(repo))
        from model.LISA import LISAForCausalLM
        from model.llava import conversation as conversation_lib
        from model.llava.mm_utils import tokenizer_image_token
        from model.segment_anything.utils.transforms import ResizeLongestSide
        from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN
        if not torch.cuda.is_available():
            raise ValueError("LISA reference generation requires a CUDA GPU; submit the Slurm workflow")
        torch.cuda.set_device(0)
        print(f"LISA sees {torch.cuda.device_count()} CUDA GPU(s); placement mode: {device_map}", flush=True)
        if precision == "bf16" and any(torch.cuda.get_device_capability(i)[0] < 8
                                      for i in range(torch.cuda.device_count())):
            raise ValueError("These GPUs do not support native BF16; use --lisa-precision fp16 for Quadro RTX 6000")
        lisa_path = model if Path(model).is_dir() else snapshot_download(
            model, revision=LISA_REVISION if model == LISA_MODEL else None,
            allow_patterns=["*.json", "*.bin", "tokenizer.model"])
        vision_path = snapshot_download(VISION_MODEL, revision=VISION_REVISION,
                                        allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
        self.dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(lisa_path, model_max_length=512,
                                                       padding_side="right", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        seg_index = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        config = AutoConfig.from_pretrained(lisa_path)
        # The merged checkpoint uses config.vision_tower to construct the CLIP
        # module. Use the pinned local snapshot instead of an unpinned HF name.
        config.vision_tower = config.mm_vision_tower = vision_path
        self.model = LISAForCausalLM.from_pretrained(
            lisa_path, config=config, low_cpu_mem_usage=True, torch_dtype=self.dtype,
            vision_tower=vision_path, seg_token_idx=seg_index)
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        # Keep construction/precision conversion on CPU. Moving the complete
        # 13B model to CUDA first exceeds a 24-GiB RTX 6000 even on an 8-GPU node.
        self.model = self.model.to(dtype=self.dtype)
        from .placement import place_lisa_model
        self.model, self.execution_info = place_lisa_model(self.model, device_map)
        self.processor = CLIPImageProcessor.from_pretrained(vision_path)
        self.resize = ResizeLongestSide(1024)
        self.conversations = conversation_lib
        self.image_tokenizer = tokenizer_image_token
        self.image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    def predict(self, image_path, query):
        import cv2
        import torch
        import torch.nn.functional as functional
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot decode LISA input: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = [image.shape[:2]]
        conversation = self.conversations.conv_templates["llava_v1"].copy()
        conversation.messages = []
        conversation.append_message(conversation.roles[0], self.image_token + "\n" + query)
        conversation.append_message(conversation.roles[1], "")
        image_clip = self.processor.preprocess(image, return_tensors="pt")["pixel_values"].to("cuda", self.dtype)
        resized = self.resize.apply_image(image)
        resize_list = [resized.shape[:2]]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float()
        mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        tensor = functional.pad(tensor, (0, 1024-tensor.shape[2], 0, 1024-tensor.shape[1]))
        tensor = tensor.unsqueeze(0).to("cuda", self.dtype)
        ids = self.image_tokenizer(conversation.get_prompt(), self.tokenizer, return_tensors="pt").unsqueeze(0).cuda()
        with torch.no_grad():
            _, predictions = self.model.evaluate(image_clip, tensor, ids, resize_list,
                                                  original_size, max_new_tokens=512, tokenizer=self.tokenizer)
        if not predictions or predictions[0].shape[0] == 0:
            return None
        return predictions[0].detach().cpu().numpy()[0] > 0
