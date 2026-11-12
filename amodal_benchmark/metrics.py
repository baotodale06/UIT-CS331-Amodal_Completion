"""The four Table 3 metrics, following the released eva_amodal.py operations.

The aggregate CLIP bookkeeping error is corrected: raw CLIP logits are reported,
not OpenCLIP cosine similarities. Unspecified CLIP scores for failures stay null.
"""
import numpy as np
from PIL import Image

from .protocol import white_rgb, align_rgba

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class MetricModels:
    def __init__(self, device="cuda"):
        import torch
        import lpips
        from torchvision import models, transforms
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.lpips = lpips.LPIPS(net="alex").eval().to(device)
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval().to(device)
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
        self.vgg_preprocess = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def clip_score(self, image, label):
        import torch
        with torch.no_grad():
            inputs = self.processor(text=[label], images=Image.fromarray(image),
                                    return_tensors="pt", padding=True).to(self.device)
            return self.clip(**inputs).logits_per_image.item()

    def score(self, reference_path, prediction_path, label, geometry, alignment="original-frame"):
        import cv2
        import torch
        import torch.nn.functional as functional
        from skimage.metrics import structural_similarity
        with Image.open(reference_path) as ref_image:
            if ref_image.mode != "RGBA":
                raise ValueError(f"Reference must be RGBA: {reference_path}")
            rgba = np.array(ref_image)
        with Image.open(prediction_path) as pred_image:
            pred_rgba = np.array(pred_image.convert("RGBA"))
        clip_prediction = white_rgb(pred_rgba)
        if alignment == "original-frame":
            rgba, pred_rgba = align_rgba(rgba, pred_rgba, geometry)
        elif alignment != "released":
            raise ValueError(f"Unknown metric alignment: {alignment}")
        reference, prediction = white_rgb(rgba), white_rgb(pred_rgba)
        # The released evaluator uses the bounding box of alpha < 255, rather
        # than the visible-object bbox. Keep that choice explicit and unchanged.
        region = (rgba[..., 3] < 255).astype(np.uint8)
        if not region.any():
            raise ValueError(f"Reference has no transparent/occluded region: {reference_path}")
        x, y, w, h = cv2.boundingRect(region)
        first = reference[y:y+h, x:x+w]
        second = prediction[y:y+h, x:x+w]
        if not first.size or not second.size:
            raise ValueError("Reference/prediction crop is empty; verify the authors' reference geometry")
        # VGG resizes its own inputs independently, exactly as eva_amodal.py.
        with torch.no_grad():
            f1 = self.vgg(self.vgg_preprocess(first).unsqueeze(0).to(self.device))
            f2 = self.vgg(self.vgg_preprocess(second).unsqueeze(0).to(self.device))
            feature = functional.cosine_similarity(f1, f2).mean().item()
            if first.shape != second.shape:
                common = (max(first.shape[1], second.shape[1]), max(first.shape[0], second.shape[0]))
                first = cv2.resize(first, common, interpolation=cv2.INTER_AREA)
                second = cv2.resize(second, common, interpolation=cv2.INTER_AREA)
            a = torch.from_numpy(first.copy()).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1
            b = torch.from_numpy(second.copy()).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1
            perceptual = self.lpips(a, b).item()
        minimum = min(first.shape[:2])
        win = min(7, minimum if minimum % 2 else minimum - 1)
        data_range = float(second.max()) - float(second.min())
        if win < 3 or data_range == 0:
            raise ValueError("Released SSIM protocol is undefined for this image (tiny/constant prediction)")
        structural = float(structural_similarity(first, second, channel_axis=-1,
                                                 data_range=data_range, win_size=win))
        # Released evaluator treats exact copies as failed completion. Preserve
        # this convention for comparison, even for an explicit success status.
        return {"CLIP": self.clip_score(clip_prediction, label),
                "LPIPS": 1. if perceptual == 0 else perceptual,
                "Feature Similarity": 0. if feature == 1 else feature,
                "SSIM": 0. if structural == 1 else structural}
