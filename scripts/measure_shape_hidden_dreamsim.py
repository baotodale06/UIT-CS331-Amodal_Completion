"""Score the hidden core of cached controlled candidates with the frozen DreamSim ensemble."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from analyze_surface_followup import paired_interval
from amodal_benchmark.dataset import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/surface-confirmation128-20261110"
OUT = ROOT / "output/shape-20261121"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def image(path, expected, mode):
    if sha256(path) != expected:
        raise ValueError(f"Image differs from the frozen input at {path}")
    with Image.open(path) as src:
        return np.array(src.convert(mode))


def main():
    from dreamsim import dreamsim

    cache = ROOT / ".amodal-cache/dreamsim"
    manifest = read(cache / "manifest.json")
    for filename, digest in manifest["files"].items():
        if sha256(cache/filename) != digest:
            raise ValueError(f"DreamSim cache changed at {filename}")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    model, preprocess = dreamsim(pretrained=True, device="cpu", cache_dir=str(cache))
    model.eval()
    cases = read(SOURCE / "cases.json")
    audit = read(OUT / "controlled-audit.json")
    if not audit["passed"] or audit["cases"] != len(cases) or audit["candidates"] != 8*len(cases):
        raise ValueError("The controlled candidate audit has not verified this panel")
    results = []
    for case in cases:
        cid = case["case_id"]
        choices = OUT / "controlled" / cid / "result.json"
        frozen = read(choices)
        assert frozen["target_id"] == case["target_id"] and frozen["source"] == case["source"]
        records = read(SOURCE / "inference/decisions" / (cid+".json"))["candidate_record_sha256"]
        evaluation = SOURCE / "evaluation_only" / cid
        hashes = case["evaluation_hashes"]
        core = image(evaluation/"hidden_core.png", hashes["hidden_core.png"], "L") > 0
        assert core.sum() == case["hidden_core_pixels"] and core.any()
        images = [image(evaluation/"truth.png", hashes["truth.png"], "RGB")]
        pool = SOURCE / "full/candidates" / cid
        for seed in range(11, 89, 11):
            meta = pool / f"seed-{seed}.json"
            assert sha256(meta) == records[meta.name]
            filename = f"seed-{seed}.png"
            images.append(image(pool/filename, read(meta)["artifacts"][filename], "RGB"))
        tensors = []
        for rgb in images:
            rgb[~core] = 127
            tensors.append(preprocess(Image.fromarray(rgb)).reshape(1,3,224,224))
        batch = torch.cat(tensors)
        with torch.inference_mode():
            features = model.embed(batch)
            distances = (1 - F.cosine_similarity(features[1:], features[:1])).tolist()
            if not results:
                direct = float(model(batch[:1], batch[1:2]).item())
                np.testing.assert_allclose(distances[0], direct, rtol=0, atol=2e-5)
        scores = {policy:distances[index] for policy,index in frozen["indices"].items()}
        for k in (4,8):
            scores[f"uniform{k}"] = float(np.mean(distances[:k]))
        results.append(dict(id=case["target_id"], source=case["source"], hidden_pixels=int(core.sum()),
                            decision_sha256=sha256(choices), scores=scores, candidate_distances=distances))
    sources = [r["source"] for r in results]
    summaries = {}
    for policy in ("first", "shape4", "shape8", "surface8", "combined8"):
        comparator = "uniform4" if policy == "shape4" else "uniform8"
        summaries[policy] = paired_interval([r["scores"][comparator] for r in results],
                                            [r["scores"][policy] for r in results], sources)
    write_json(OUT/"hidden-dreamsim.json", dict(job=os.environ.get("SLURM_JOB_ID"),
               metric="Original DreamSim ensemble cosine distance on RGB, pixels outside hidden_core set to 127",
               device="cpu", supplemental=True, cache_manifest_sha256=sha256(cache/"manifest.json"),
               cases_sha256=sha256(SOURCE/"cases.json"), controlled_audit_sha256=sha256(OUT/"controlled-audit.json"),
               source_sha256=sha256(Path(__file__)), summary=summaries, rows=results))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
