"""Replay the selected robust prior on validation only and save native masks."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2 import make_network, observation_tensor, visible_box
from scripts.shape_robust_data import aggregate, geometry, native_canvas


def paired(rows):
    report = {}
    for metric in ("full_iou", "hidden_iou"):
        groups = defaultdict(list)
        for row in rows:
            a, b = row["v2_sam"][metric], row["constrained_sam"][metric]
            if (a is None) != (b is None):
                raise ValueError("Native hidden eligibility differs")
            if a is not None:
                groups[row["image_id"]].append([a,b])
        means = np.array([np.mean(groups[key], axis=0) for key in sorted(groups)])
        delta = means[:,1]-means[:,0]
        rng = np.random.default_rng(20261123)
        boot = delta[rng.integers(len(delta), size=(5000,len(delta)))].mean(axis=1)
        report[metric] = dict(images=len(means), objects=sum(map(len,groups.values())),
            v2_sam=float(means[:,0].mean()), constrained_sam=float(means[:,1].mean()),
            absolute_change=float(delta.mean()), absolute_ci=np.quantile(boot,[.025,.975]).tolist())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sam-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.training/"training.json").read_bytes())
    if receipt.get("schema") != 4 or receipt.get("complete") is not True or receipt.get("preflight") is not False:
        raise ValueError("Completed full robust training required")
    if receipt.get("holdout_used") is not False or receipt.get("benchmark_used") is not False:
        raise ValueError("Training scope changed")
    checkpoint = args.training/"best.pt"
    if sha256(checkpoint) != receipt["best_sha256"]:
        raise ValueError("Selected checkpoint hash changed")
    for name, expected in receipt["contract"]["sources"].items():
        if sha256(args.training_source/name) != expected:
            raise ValueError("Training source changed")
    config_path = args.training_source/"configs/research/shape-sam-constrained-20261123.json"
    if sha256(config_path) != receipt["contract"]["config_sha256"]:
        raise ValueError("Training configuration changed")
    config = json.loads(config_path.read_bytes())
    if receipt["epochs_completed"] != config["training"]["epochs"]:
        raise ValueError("Training epochs incomplete")
    manifest_path, baseline_path = args.data/"manifest.json", args.sam_validation/"result.json"
    if (sha256(manifest_path) != receipt["contract"]["data_sha256"]
            or sha256(baseline_path) != receipt["contract"]["validation_sam_result_sha256"]):
        raise ValueError("Validation data differs from training contract")
    manifest, baseline = (json.loads(path.read_bytes()) for path in (manifest_path, baseline_path))
    if {row["split"] for row in manifest["rows"]} != {"train", "validation"}:
        raise ValueError("Holdout must be absent")
    selected = [row for row in manifest["rows"] if row["split"] == "validation"]
    baseline_rows = {row["id"]:row for row in baseline["rows"]}
    best_path = args.training/"best-validation-native.json"
    if sha256(best_path) != receipt["best_validation_sha256"]:
        raise ValueError("Selected validation evidence changed")
    best = json.loads(best_path.read_bytes())
    best_rows = {row["id"]:row for row in best["rows"]}
    if len(selected) != 1264 or set(best_rows) != {row["id"] for row in selected} or set(best_rows) != set(baseline_rows):
        raise ValueError("Validation target inventory differs")
    baseline = receipt["initial_validation"]
    candidates = [dict(epoch=0, validation=baseline)] + [row for row in receipt["history"]
        if row["validation"]["full_iou"] >= baseline["full_iou"]]
    chosen = max(candidates, key=lambda row:(row["validation"]["hidden_iou"],row["validation"]["full_iou"]))
    if receipt["selected_epoch"] != chosen["epoch"] or receipt["new_checkpoint_eligible"] != (chosen["epoch"] > 0):
        raise ValueError("Constrained checkpoint eligibility changed")
    if best["epoch"] != chosen["epoch"] or best["aggregate"] != chosen["validation"]:
        raise ValueError("Checkpoint was not selected by the declared validation rule")
    if args.output.exists():
        raise FileExistsError("Preserve prior native evaluation")
    import torch
    if torch.cuda.device_count() != 1 or "QUADRO" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("One allocated Quadro required")
    probe = torch.randn(32,32,device="cuda")
    if not torch.allclose(probe,torch.fft.irfft2(torch.fft.rfft2(probe),s=probe.shape),atol=1e-5):
        raise RuntimeError("CUDA FFT failed")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    saved = torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved["contract"] != receipt["contract"] or saved["epoch"] != best["epoch"] or saved["validation"] != best["aggregate"]:
        raise ValueError("Checkpoint payload differs from selected validation")
    model = make_network(pretrained=False).cuda().eval()
    model.load_state_dict(saved["model"],strict=True)
    args.output.mkdir(parents=True)
    contract = dict(schema=1,at=datetime.now(timezone.utc).isoformat(),split="validation",objects=1264,images=217,
        training_receipt_sha256=sha256(args.training/"training.json"),checkpoint_sha256=sha256(checkpoint),
        selected_epoch=best["epoch"],data_manifest_sha256=sha256(manifest_path),
        baseline_result_sha256=sha256(baseline_path),source_sha256=sha256(Path(__file__)),
        geometry_policy=receipt["geometry_policy"],job_id=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,cuda_fft_round_trip=True,
        holdout_access=False,benchmark_access=False,
        interpretation="Validation-selected exploratory result, not untouched confirmation or benchmark improvement")
    write_json(args.output/"contract.json",contract)
    groups = defaultdict(list)
    for row in selected:
        groups[row["image_id"]].append(row)
    records, verified, max_difference = [], set(), 0.
    with torch.inference_mode():
        for index,(image_id,objects) in enumerate(sorted(groups.items()),1):
            for row in objects:
                for name in (row["image"],row["mask"]):
                    path=(args.data/name).resolve()
                    if not path.is_relative_to(args.data.resolve()):
                        raise ValueError("Validation path escaped data")
                    if name not in verified:
                        if sha256(path)!=manifest["artifacts"][name]:
                            raise ValueError("Validation asset changed")
                        verified.add(name)
            with Image.open(args.data/objects[0]["image"]) as source:
                rgb=np.array(source.convert("RGB"))
            for row in objects:
                original=baseline_rows[row["id"]]
                if any(original[key]!=row[key] for key in ("image_id","image","mask")):
                    raise ValueError("Baseline validation identity changed")
                sam_path=args.sam_validation/"objects"/row["id"]/"sam-visible.png"
                if sha256(sam_path)!=original["artifacts"]["sam-visible.png"]:
                    raise ValueError("SAM input mask changed")
                with Image.open(sam_path) as source:
                    support=np.array(source)>0
                with Image.open(args.data/row["mask"]) as source:
                    encoded=np.array(source)
                truth,visible=(encoded&1)>0,(encoded&2)>0
                if support.shape!=truth.shape:
                    raise ValueError("Native validation mask frames differ")
                if support.any():
                    box=visible_box(support,2.)
                    rgb_crop=np.array(Image.fromarray(rgb).crop(box))
                    mask_crop=np.array(Image.fromarray(support).crop(box))
                    x=torch.from_numpy(observation_tensor(rgb_crop,mask_crop)).unsqueeze(0).cuda()
                    logits=model(x)
                    probability=torch.nn.functional.interpolate(logits,mask_crop.shape,mode="bilinear",align_corners=False).sigmoid()
                    prediction=(native_canvas(probability[0,0].cpu().numpy(),box,truth.shape)>=.5) | support
                else:
                    box,prediction=None,np.zeros_like(truth)
                metrics=geometry(prediction,truth,visible)
                for metric,value in metrics.items():
                    expected=best_rows[row["id"]][metric]
                    if (value is None)!=(expected is None):
                        raise ValueError("Selected validation eligibility changed")
                    if value is not None:
                        max_difference=max(max_difference,abs(value-expected))
                directory=args.output/"objects"/row["id"]
                directory.mkdir(parents=True)
                Image.fromarray(prediction.astype(np.uint8)*255).save(directory/"prediction.png")
                record=dict(id=row["id"],image_id=image_id,image=row["image"],mask=row["mask"],box=box,
                    sam_sha256=sha256(sam_path),prediction_sha256=sha256(directory/"prediction.png"),
                    v2_sam=original["sam_input"],constrained_sam=metrics)
                write_json(directory/"result.json",record)
                records.append(record)
            write_json(args.output/"progress.json",dict(images=index,objects=len(records),status="running"))
    if max_difference>1e-10:
        raise ValueError("Native checkpoint replay differs from selected validation: "+str(max_difference))
    result=dict(complete=True,objects=len(records),images=len(groups),selected_epoch=best["epoch"],
        contract_sha256=sha256(args.output/"contract.json"),max_selected_validation_difference=max_difference,
        geometry=paired(records),rows=records,benchmark_success=False)
    write_json(args.output/"result.json",result)
    write_json(args.output/"progress.json",dict(images=217,objects=1264,status="complete"))
    print(json.dumps({key:value for key,value in result.items() if key!="rows"},indent=2),flush=True)


if __name__=="__main__":
    main()
