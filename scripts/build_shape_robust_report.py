"""Report recovered native validation development; never infer benchmark gains."""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    study = ROOT/"output/shape-sam-robustness-20261123"
    replay = json.loads((study/"native-replay.json").read_bytes())
    cost = json.loads((study/"allocation-cost.json").read_bytes())
    evidence = study/"recovered"
    training = json.loads((evidence/"training/training.json").read_bytes())
    if (replay.get("verified") is not True or replay.get("benchmark_success") is not False
            or cost.get("final") is not True or training.get("complete") is not True
            or sha256(study/"training/best.pt") != replay["checkpoint_sha256"]
            or sha256(evidence/"training/training.json") != replay["training_sha256"]
            or sha256(evidence/"native-validation/result.json") != replay["result_sha256"]):
        raise ValueError("Complete verified recovery and final accounting are required")
    for stem, manifest_name in (("source", "source-manifest.json"),
            ("training-source", "training-source-manifest.json"), ("native-source", "native-source-manifest.json")):
        if sha256(evidence/manifest_name) != sha256(study/manifest_name):
            raise ValueError("Deployed source manifest differs from recovered source")
        manifest = json.loads((evidence/manifest_name).read_bytes())
        if sha256(evidence/(stem+".zip")) != manifest["zip_sha256"]:
            raise ValueError("Recovered source archive changed")
        import hashlib
        with zipfile.ZipFile(evidence/(stem+".zip")) as bundle:
            if set(bundle.namelist()) != set(manifest["files"]):
                raise ValueError("Source archive inventory changed")
            for name, expected in manifest["files"].items():
                if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                    raise ValueError("Source archive file changed")
    if any(row["state"] != "COMPLETED" or row["exit_code"] != "0:0" for row in cost["jobs"].values()):
        raise ValueError("Review unsuccessful development jobs before reporting")
    output = study/"report"
    if output.exists():
        raise FileExistsError("Preserve the previous report")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir()
    history = training["history"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    for axis, metric, label in zip(axes[:2], ("full_iou", "hidden_iou"), ("Native full IoU", "Native hidden IoU")):
        axis.plot([row["epoch"] for row in history], [row["validation"][metric] for row in history], color="#176b87", label="Fine-tuning")
        axis.axhline(replay["geometry"][metric]["v2_sam"], color="#be6a15", linestyle="--", label="Frozen prior + SAM")
        axis.axvline(replay["selected_epoch"], color="0.4", linestyle=":", label="Selected epoch")
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].plot([row["epoch"] for row in history], [row["loss"] for row in history], color="#176b87")
    axes[2].set(xlabel="Epoch", ylabel="Training loss")
    axes[2].grid(alpha=.2)
    fig.suptitle("Development validation only — not full-benchmark quality metrics", fontsize=10)
    for extension in ("png", "svg"):
        fig.savefig(output/("training."+extension), dpi=160)
    plt.close(fig)
    rows = []
    for metric, label in (("full_iou", "Full IoU"), ("hidden_iou", "Hidden IoU")):
        value = replay["geometry"][metric]
        lo, hi = value["absolute_ci"]
        relative = 100*value["absolute_change"]/value["v2_sam"]
        rows.append(f"| {label} | {value['v2_sam']:.6f} | {value['robust_sam']:.6f} | {100*value['absolute_change']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {relative:+.3f}% |")
    jobs = "\n".join(f"| {job} | {row['role']} | {row['seconds']} | {row['reserved_gpu_hours']:.6f} |" for job, row in cost["jobs"].items())
    text = f"""# Shape prior trained with SAM masks: validation development

All 20 fine-tuning epochs completed. The fixed native hidden-IoU selection rule
chose epoch **{replay['selected_epoch']}**. Independent GPU checkpoint replay and local
native-mask/statistical replay passed for all 1,264 validation objects across
217 images. This is exploratory validation evidence. The active >=3% requirement
on CLIP, LPIPS, VGG Feature Similarity and SSIM remains unassessed for this model.

| Native image-mean metric | Frozen prior, SAM input | Fine-tuned prior, SAM input | Change (IoU percentage points) | Paired 95% interval (points) | Relative change |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Full IoU includes all 217 images and 1,264 objects. Hidden IoU includes 177 images
and 680 objects. Both use the same native amodal truth and annotated hidden
region, including truth outside the input-defined crop. Intervals use 5,000
paired image bootstrap draws, seed 20261123. They are nominal development
intervals after validation checkpoint selection, not untouched confirmation.

![Training and native validation curves](training.png)

The model starts from the completed v2 epoch-47 checkpoint and fine-tunes all
parameters on the same 9,494 training objects. Inputs mix 75% SAM support and
25% annotated support, with extra occlusion augmentation. True annotated
visibility defines hidden supervision. The five-channel RGB/mask model restricts
RGB to input support and predicts a shape probability mask without hard-restoring
SAM pixels. There is no Surface ranker, appearance loss, RGB correction or
combined method. The frozen v2 benchmark was not changed.

SAM was prompted with boxes derived from annotated visible support. This captures
mask-boundary errors but not GroundingDINO selection errors. No holdout or
benchmark evaluation was used for this fine-tuning. Native geometry cannot
establish the required image-quality gains or successful-output coverage.

| Job | Stage | Allocated seconds | Allocated GPU-hours |
| --- | --- | ---: | ---: |
{jobs}

This experiment adds **{cost['total_reserved_gpu_hours']:.6f} allocated GPU-hours**.
The inherited v2 training cost and preceding SAM diagnostic are separate costs;
the total above is not the complete research cost. All listed jobs completed
with exit 0:0. Checkpoint, source snapshots, native masks, training-mask inventory
and receipts are recovered under `../recovered/` and `../training/best.pt`.

Selected checkpoint SHA-256: `{replay['checkpoint_sha256']}`.
See `../native-replay.json`, `../native-export.json`, `../allocation-cost.json`,
and `../../shape-validation-sam-20261123/REPORT.md` for the original diagnostic.
"""
    (output/"REPORT.md").write_text(text, encoding="utf8")
    write_json(output/"receipt.json", dict(native_replay_sha256=sha256(study/"native-replay.json"),
        allocation_sha256=sha256(study/"allocation-cost.json"), source_sha256=sha256(Path(__file__)),
        files={path.name:sha256(path) for path in output.iterdir() if path.is_file()}, benchmark_success=False))
    print(str(output/"REPORT.md"))


if __name__ == "__main__":
    main()
