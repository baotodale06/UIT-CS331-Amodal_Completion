"""Adapter C2F-Seg có kiểm tra provenance và cache không chứa GT."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from .runtime import event, stage


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    """Ghi atomic để file metadata không bị cắt dở khi server dừng tiến trình."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def validate_c2f(args):
    """Hash cả weights, configs và source để cache hết hiệu lực khi model đổi."""
    source = Path(args.c2f_source).resolve()
    if not (source / 'src/image_component.py').is_file():
        raise FileNotFoundError('Thiếu C2F-Seg source; chạy scripts/download_c2f.py')
    identity = {'backend': 'C2F-Seg', 'adapter_version': 2, 'threshold': args.shape_threshold,
                'seed': args.c2f_seed, 'steps': args.c2f_steps, 'expand': args.c2f_expand}
    for name in ('c2f_checkpoint', 'c2f_vq_checkpoint', 'c2f_config', 'c2f_vq_config'):
        identity[name] = file_sha256(getattr(args, name))
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*.py')):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    identity['source_sha256'] = digest.hexdigest()
    identity['worker_sha256'] = file_sha256(Path(__file__).with_name('c2f_worker.py'))
    identity['preprocessing_sha256'] = file_sha256(Path(__file__).with_name('c2f_inputs.py'))
    return identity


def predict_c2f(args, image, visible, device):
    """Process riêng tránh xung đột modules utils/models và NumPy của repo cũ."""
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix='pmc-c2f-') as folder:
        input_path, output_path = Path(folder)/'input.npz', Path(folder)/'output.npz'
        np.savez_compressed(input_path, image=image, visible=visible)
        command = [str(args.c2f_python), '-m', 'pmc.c2f_worker']
        for name in ('source', 'checkpoint', 'vq_checkpoint', 'config', 'vq_config', 'seed', 'steps', 'expand'):
            command.extend(['--'+name, str(getattr(args, 'c2f_'+name))])
        command.extend(['--input', str(input_path), '--output', str(output_path), '--device', device])
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        import time
        log_path = Path(folder)/'worker.log'
        with log_path.open('w') as output_log, log_path.open() as input_log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                       stdout=output_log, stderr=subprocess.STDOUT, text=True)
            start = heartbeat = time.monotonic()
            tail = []
            def drain():
                for line in input_log:
                    tail.append(line.rstrip())
                    del tail[:-40]
                    event('c2f_worker_log', output=line.rstrip())
            try:
                while process.poll() is None:
                    drain()
                    now = time.monotonic()
                    if now-start > args.c2f_timeout:
                        raise TimeoutError(f'C2F timeout sau {args.c2f_timeout}s')
                    if now-heartbeat >= 30:
                        event('c2f_worker_running', seconds=round(now-start, 1), pid=process.pid)
                        heartbeat = now
                    time.sleep(0.2)
                drain()
                if process.returncode:
                    raise RuntimeError(f'C2F worker exit={process.returncode}: '+ '\n'.join(tail))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                    drain()
        with np.load(output_path, allow_pickle=False) as data:
            probability = data['probability']
        if probability.shape != visible.shape or not np.isfinite(probability).all():
            raise ValueError('C2F trả probability sai kích thước hoặc NaN')
        return (probability >= args.shape_threshold) | visible


class ShapePredictor:
    """Một prior/object từ ảnh gốc, cache theo nội dung + weights + source + config."""
    def __init__(self, args, identity, manager):
        self.args, self.identity, self.manager = args, identity, manager
        self.cache = Path(args.shape_cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def predict(self, image, visible, occluder):
        image = np.asarray(image, dtype=np.uint8)
        visible, occluder = np.asarray(visible, bool), np.asarray(occluder, bool)
        if image.shape[:2] != visible.shape or visible.shape != occluder.shape:
            raise ValueError("Input C2F không cùng hệ tọa độ")
        digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode())
        for arr in (image, visible, occluder):
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
        key = digest.hexdigest()
        path = self.cache / (key + ".npz")
        with stage("shape_prior", cache_key=key):
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    prior = data["prior"].astype(bool)
                if prior.shape != visible.shape or not np.all(prior[visible]):
                    raise ValueError(f"Cache prior sai contract: {path}")
                event("shape_cache_hit", path=str(path))
            else:
                if self.manager.active is not None:
                    raise RuntimeError("C2F chỉ chạy khi các model PMC đã offload")
                prior = predict_c2f(self.args, image, visible, self.manager.device)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, prior=prior)
                temporary.replace(path)
                save_json(path.with_suffix(".json"), {"cache_key": key, "predictor": self.identity})
                event("shape_cache_written", path=str(path))
        return prior
