"""Worker độc lập: C2F-Seg không dùng loader GT, DDP, optimizer hay môi trường PMC."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def predict(args):
    print("c2f_stage=imports", flush=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    from skimage.transform import rescale
    from PIL import Image
    sys.path.insert(0, str(Path(args.source).resolve()))
    from taming_src.taming_models import VQModel
    import src.image_component as components
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = SimpleNamespace(**yaml.safe_load(Path(args.config).read_text()))
    vq_config = SimpleNamespace(**yaml.safe_load(Path(args.vq_config).read_text()))
    vq_config.device, vq_config.path = 'cpu', str(Path(args.vq_checkpoint).parent)
    # Tạo đúng các module inference, không gọi constructor training vốn yêu cầu DDP.
    print('c2f_stage=build_modules_cpu', flush=True)
    vq = VQModel(vq_config)
    transformer = components.MaskedTransformer(config)
    # Checkpoint đã có toàn bộ ResNet: không tải ImageNet weights lần nữa.
    original_resnet = components.resnet50
    with patch.object(components, 'resnet50', lambda **kwargs: original_resnet(weights=None)):
        encoder = components.Resnet_Encoder()
    refine = components.Refine_Module()
    print('c2f_stage=load_checkpoints_cpu', flush=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    vq_state = torch.load(args.vq_checkpoint, map_location='cpu', weights_only=False)
    for module, weights in ((vq, vq_state['g_model']), (transformer, state['model']),
                            (encoder, state['img_encoder']), (refine, state['refine'])):
        module.load_state_dict({k.removeprefix('module.'): v for k, v in weights.items()}, strict=True)
        module.eval().requires_grad_(False).to(args.device)
    del state, vq_state
    with np.load(args.input, allow_pickle=False) as data:
        image, visible = data['image'], data['visible'].astype(bool)
    # Crop chỉ lấy từ visible, tuyệt đối không đọc full-mask hoặc amodal GT box.
    from pmc.c2f_inputs import visible_crop_bounds
    r0, r1, c0, c1 = visible_crop_bounds(visible, args.expand)
    h, w = r1-r0, c1-c0
    rgb = rescale(image[r0:r1, c0:c1], (256/h, 256/w, 1))[:256, :256, :3]
    vm = rescale(visible[r0:r1, c0:c1], (256/h, 256/w))[:256, :256]
    rgb = np.pad(rgb, ((0, 256-rgb.shape[0]), (0, 256-rgb.shape[1]), (0, 0)))
    vm = np.pad(vm, ((0, 256-vm.shape[0]), (0, 256-vm.shape[1])))
    print('c2f_stage=inference_gpu', flush=True)
    with torch.inference_mode():
        features = encoder(torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(args.device))
        quant, _, info = vq.encode(torch.from_numpy(vm)[None, None].float().to(args.device))
        src = info[2].view(1, -1)
        bhwc = (1, quant.shape[2], quant.shape[3], quant.shape[1])
        current = torch.full_like(src, config.vocab_size)
        initial = (current == config.vocab_size).sum(-1)
        # Thuật toán MaskGIT T=3, top-k=3 và choice_temperature=4.5 theo upstream.
        for t in range(args.steps):
            logits = transformer(features[-1], src, current, mask=None)[..., :-1]
            values = logits.topk(3).values
            logits = logits.masked_fill(logits < values[..., -1:], -float('inf'))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()
            unknown = current == config.vocab_size
            sampled = torch.where(unknown, sampled, current)
            ratio = (t+1)/args.steps
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            confidence = torch.where(unknown, confidence, torch.full_like(confidence, float('inf')))
            length = torch.floor(initial * np.cos(ratio * np.pi/2)).unsqueeze(1)
            length = torch.maximum(torch.ones_like(length), torch.minimum(unknown.sum(-1, keepdim=True)-1, length))
            noisy = torch.log(confidence) + 4.5*(1-ratio)*torch.distributions.Gumbel(0, 1).sample(confidence.shape).to(args.device)
            cutoff = noisy.sort(dim=-1).values.gather(-1, length.long())
            current = torch.where(noisy < cutoff, config.vocab_size, sampled)
        coarse = vq.decode(vq.quantize.get_codebook_entry(sampled.reshape(-1), shape=bhwc)).mean(1, keepdim=True).clamp(0, 1)
        _, refined = refine(features, coarse)
        probability = F.interpolate(refined, (256, 256), mode='nearest').sigmoid()[0, 0].cpu().numpy()
    full = np.zeros(visible.shape, dtype=np.float32)
    full[r0:r1, c0:c1] = np.asarray(Image.fromarray(probability).resize((w, h), Image.Resampling.BILINEAR))
    np.savez_compressed(args.output, probability=full)
    print(json.dumps({'event': 'c2f_complete', 'bounds_rc': [r0,r1,c0,c1],
                      'peak_vram_mib': torch.cuda.max_memory_allocated()/2**20}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'vq_checkpoint', 'config', 'vq_config', 'input', 'output'):
        p.add_argument('--'+name)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--expand', type=float, default=2)
    p.add_argument('--check_environment', action='store_true')
    args = p.parse_args()
    if args.check_environment:
        import torch, torchvision, numpy, scipy, skimage, cv2, yaml
        sys.path.insert(0, str(Path(args.source).resolve()))
        from src.image_component import MaskedTransformer
        from taming_src.taming_models import VQModel
        print(json.dumps({'torch': torch.__version__, 'torchvision': torchvision.__version__,
                          'numpy': numpy.__version__, 'cuda_available': torch.cuda.is_available()}))
    else:
        for name in ('source', 'checkpoint', 'vq_checkpoint', 'config', 'vq_config', 'input', 'output'):
            if not getattr(args, name):
                p.error('--'+name+' là bắt buộc khi inference')
        predict(args)
