"""Memory-aware LISA placement for multiple individually small CUDA devices."""
GIB = 1024 ** 3


def plan_lisa_devices(layer_bytes, fixed_bytes, fixed_modules, budgets):
    """Balance contiguous decoder blocks; keep all other modules on CUDA 0.

    Map only disjoint module subtrees. An Accelerate map containing a root or
    model parent would recursively move every decoder to GPU 0 while attaching
    hooks, recreating the OOM before the child entries are processed.
    """
    devices = sorted(budgets)
    if not devices or devices[0] != 0 or not layer_bytes or any(n <= 0 for n in layer_bytes):
        raise ValueError("Expected visible GPU 0 and nonempty LISA decoder layers")
    if fixed_bytes < 0 or any(name in ("", "model", "model.layers") for name in fixed_modules):
        raise ValueError("LISA placement requires disjoint modules without a root/model parent entry")
    if fixed_bytes > budgets[0]:
        raise ValueError(f"GPU 0 cannot fit LISA's vision/segmentation components: "
                         f"{fixed_bytes / GIB:.2f} GiB weights, {budgets[0] / GIB:.2f} GiB budget. "
                         "Free memory on the first visible GPU or reorder your allocated GPUs.")

    def pack(ceiling):
        assignment, loads, layer = {}, {gpu: 0 for gpu in devices}, 0
        loads[0] = fixed_bytes
        for gpu in devices:
            capacity = min(budgets[gpu], ceiling)
            while layer < len(layer_bytes) and loads[gpu] + layer_bytes[layer] <= capacity:
                assignment[f"model.layers.{layer}"] = gpu
                loads[gpu] += layer_bytes[layer]
                layer += 1
        return assignment, loads, layer == len(layer_bytes)

    lower, upper = max(fixed_bytes, max(layer_bytes)), max(budgets.values())
    if not pack(upper)[2]:
        needed = (fixed_bytes + sum(layer_bytes)) / GIB
        raise ValueError(f"LISA needs {needed:.2f} GiB of weights, which do not fit the "
                         f"{len(devices)} selected visible GPU(s) with inference headroom. "
                         "Use --lisa-device-map auto and make at least two allocated RTX 6000 GPUs visible via CUDA_VISIBLE_DEVICES, "
                         "or request more GPUs from Slurm. GPU memory is not pooled automatically.")
    while lower < upper:
        middle = (lower + upper) // 2
        if pack(middle)[2]:
            upper = middle
        else:
            lower = middle + 1
    assignment, loads, _ = pack(lower)
    assignment.update({name: 0 for name in fixed_modules})
    return assignment, loads


def module_bytes(module):
    seen, size = set(), 0
    for tensors in (module.parameters(), module.buffers()):
        for tensor in tensors:
            if id(tensor) not in seen:
                seen.add(id(tensor))
                size += tensor.numel() * tensor.element_size()
    return size


def place_lisa_model(model, mode="auto"):
    """Dispatch CPU-loaded FP16 weights without ever staging them all on GPU 0."""
    import torch
    if mode not in ("auto", "single"):
        raise ValueError("LISA device map must be auto or single")
    count = torch.cuda.device_count()
    if not count:
        raise ValueError("No CUDA GPUs are visible for LISA reference generation")
    devices, budgets = [], {}
    for gpu in range(count if mode == "auto" else 1):
        free, total = torch.cuda.mem_get_info(gpu)
        # SAM's 1024px encoder, generation outputs and segmentation run on GPU
        # 0. Decoder-only cards need less temporary memory.
        reserve = (8 if gpu == 0 else 4) * GIB
        budgets[gpu] = max(0, free - reserve)
        devices.append({"index": gpu, "name": torch.cuda.get_device_name(gpu),
                        "total_gib": round(total / GIB, 2), "free_gib": round(free / GIB, 2),
                        "weight_budget_gib": round(budgets[gpu] / GIB, 2)})
    if (list(model.parameters(recurse=False)) or list(model.buffers(recurse=False))
            or list(model.model.parameters(recurse=False)) or list(model.model.buffers(recurse=False))):
        raise ValueError("Unsupported LISA layout: root/model has direct tensors outside mapped submodules")
    layers = list(model.model.layers)
    layer_sizes = [module_bytes(layer) for layer in layers]
    fixed_size = module_bytes(model) - sum(layer_sizes)
    fixed_modules = [name for name, _ in model.named_children() if name != "model"]
    fixed_modules += [f"model.{name}" for name, _ in model.model.named_children() if name != "layers"]
    mapping, loads = plan_lisa_devices(layer_sizes, fixed_size, fixed_modules, budgets)
    info = {"mode": mode, "visible_gpu_count": count, "devices": devices,
            "weight_gib_by_gpu": {str(gpu): round(size / GIB, 3) for gpu, size in loads.items()},
            "device_map": mapping, "main_device": "cuda:0", "quantized": False}
    for device in devices:
        gpu = device["index"]
        print(f"LISA GPU {gpu}: {device['name']}; free {device['free_gib']:.2f} GiB; "
              f"planned weights {loads[gpu] / GIB:.2f} GiB; runtime headroom retained", flush=True)
    if len(set(mapping.values())) == 1:
        model = model.to(device="cuda:0")
    else:
        from accelerate import dispatch_model
        # No map entry for '' or 'model': their hooks must not place all child
        # weights recursively on one card. Root output alignment is installed
        # by Accelerate, so LISA's generation hidden states return to CUDA 0.
        model = dispatch_model(model, device_map=mapping, main_device=0)
    return model.eval(), info
