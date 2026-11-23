"""Logging có ngữ cảnh và điều phối duy nhất một model trên GPU."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import time
import uuid

_CONTEXT = ContextVar("pmc_context", default={})
LOG = logging.getLogger("pmc")


class JsonFormatter(logging.Formatter):
    """Mỗi dòng JSON chứa thời gian UTC, stage/object/seed và traceback nếu có."""
    def format(self, record):
        data = dict(_CONTEXT.get())
        data.update(time=datetime.now(timezone.utc).isoformat(), level=record.levelname,
                    message=record.getMessage())
        data.update(getattr(record, "fields", {}))
        if record.exc_info:
            data["traceback"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str, allow_nan=False)


def configure_logging(directory, level="INFO"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(level)
    LOG.propagate = False
    for handler in LOG.handlers[:]:
        handler.close()
        LOG.removeHandler(handler)
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / "pipeline.jsonl", encoding="utf-8")):
        handler.setFormatter(JsonFormatter())
        LOG.addHandler(handler)
    logging.captureWarnings(True)
    warning_log = logging.getLogger("py.warnings")
    warning_log.handlers = LOG.handlers[:]
    warning_log.propagate = False


@contextmanager
def context(**fields):
    token = _CONTEXT.set({**_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def event(message, **fields):
    LOG.info(message, extra={"fields": fields})


def gpu_memory(torch_module=None, device=None):
    if torch_module is None or not torch_module.cuda.is_available():
        return {}
    cuda = torch_module.cuda
    free, total = cuda.mem_get_info(device)
    return {"allocated_mib": round(cuda.memory_allocated(device) / 2**20, 2),
            "reserved_mib": round(cuda.memory_reserved(device) / 2**20, 2),
            "peak_allocated_mib": round(cuda.max_memory_allocated(device) / 2**20, 2),
            "peak_reserved_mib": round(cuda.max_memory_reserved(device) / 2**20, 2),
            "free_mib": round(free / 2**20, 2), "total_mib": round(total / 2**20, 2)}


@contextmanager
def stage(name, torch_module=None, device=None, **fields):
    with context(stage=name, stage_id=uuid.uuid4().hex[:12], **fields):
        start = time.perf_counter()
        event("stage_start", **gpu_memory(torch_module, device))
        try:
            yield
        except Exception:
            LOG.exception("stage_failed", extra={"fields": {"seconds": time.perf_counter() - start,
                                                           **gpu_memory(torch_module, device)}})
            raise
        else:
            event("stage_end", seconds=round(time.perf_counter() - start, 3),
                  **gpu_memory(torch_module, device))


class ModelManager:
    """Factory phải tạo model ở CPU. Model rời GPU ngay khi thoát khối use()."""
    def __init__(self, device="cuda:0", policy="cpu", torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = device
        self.policy = policy
        self.factories = {}
        self.models = {}
        self.active = None

    def register(self, name, factory):
        self.factories[name] = factory

    @contextmanager
    def use(self, name):
        if self.active is not None:
            raise RuntimeError(f"Không được chạy đồng thời {self.active} và {name} trên GPU")
        self.active = name
        model = None
        try:
            if self.device.startswith("cuda"):
                self.torch.cuda.reset_peak_memory_stats(self.device)
            with stage("model_load", self.torch, self.device, model=name):
                if name not in self.models:
                    self.models[name] = self.factories[name]()
                model = self.models[name]
                model.to(self.device)
            with stage("model_inference", self.torch, self.device, model=name):
                with self.torch.inference_mode():
                    yield model
                if self.device.startswith("cuda"):
                    self.torch.cuda.synchronize(self.device)
        finally:
            try:
                # Không chỉ empty_cache: weights phải rời GPU, embedding phải được xóa.
                if model is not None:
                    if hasattr(model, "reset_image"):
                        model.reset_image()
                    model.to("cpu")
                if self.policy == "unload":
                    self.models.pop(name, None)
                model = None
                gc.collect()
                if self.device.startswith("cuda"):
                    self.torch.cuda.empty_cache()
                event("model_offloaded", model=name, policy=self.policy,
                      **gpu_memory(self.torch, self.device))
            finally:
                self.active = None

    def close(self):
        # Cho phép gọi cả khi pipeline kết thúc bằng exception.
        for model in self.models.values():
            if hasattr(model, "reset_image"):
                model.reset_image()
            model.to("cpu")
        self.models.clear()
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
