"""Replay frozen Shape, Surface and combined decisions on the 128 original Surface pools."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from train_shape import OUT, TEST, read_json
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape import choose, make_network, overlap, predict, score_masks


def verified(path, expected, mode):
    if sha256(path) != expected:
        raise ValueError(f"Changed input {path}")
    with Image.open(path) as im:
        arr = np.array(im.convert(mode))
    return arr if mode == "RGB" else arr > 0


def evaluate(model, row):
    cid = row["case_id"]
    inputs = TEST/"inputs"/cid
    visible = verified(inputs/"visible.png", row["input_hashes"]["visible.png"], "L")
    occlusion = verified(inputs/"occlusion.png", row["input_hashes"]["occlusion.png"], "L")
    observed = verified(inputs/"observed.png", row["input_hashes"]["observed.png"], "RGB")
    prior = predict(model, observed, visible, occlusion)
    candidates = TEST/"full"/"candidates"/cid
    surface = read_json(TEST/"inference"/"decisions"/(cid+".json"))
    masks = []
    for seed in range(11, 89, 11):
        name = f"seed-{seed}"
        record = candidates/(name+".json")
        assert sha256(record) == surface["candidate_record_sha256"][record.name]
        meta = read_json(record)
        assert meta["seed"] == seed and meta["case_id"] == cid and meta["generation_complete"]
        assert meta["input_hashes"] == row["input_hashes"]
        assert sha256(candidates/(name+".png")) == meta["artifacts"][name+".png"]
        masks.append(verified(candidates/(name+"-mask.png"), meta["artifacts"][name+"-mask.png"], "L"))
    scores = score_masks(prior, masks)
    weighted = [None if s is None else .60*s + .25*(m & visible).sum()/max(1,visible.sum())
                - .15*(m & ~prior).sum()/max(1,m.sum()) for m,s in zip(masks,scores)]
    choices = dict(first=0, surface8=surface["index"])
    fallbacks = {}
    for k in (4,8):
        for method, values in (("shape",scores),("shape_weighted",weighted)):
            choices[method+str(k)], fallbacks[method+str(k)] = choose(values[:k])
    choices["combined8"], fallbacks["combined8"] = choose(scores,surface["losses"])
    # All deployable choices have been made before reading the evaluation-only directory.
    truth = verified(TEST/"evaluation_only"/cid/"full.png", row["evaluation_hashes"]["full.png"], "L")
    assert np.array_equal(truth & ~occlusion, visible)
    metrics = read_json(TEST/"metrics"/(cid+".json"))
    assert metrics["case_id"] == cid and choices["surface8"] == metrics["selection"]["trained_surface"]
    assert len(metrics["raw"]) == 8
    quality = [dict(m, iou=overlap(mask, truth), hidden_iou=overlap(mask & occlusion, truth & occlusion))
               for m,mask in zip(metrics["raw"], masks)]
    for k in (4,8):
        choices[f"oracle{k}"] = int(np.argmax([m["iou"] for m in quality[:k]]))
    values = {name:quality[idx] for name,idx in choices.items()}
    for k in (4,8):
        values[f"uniform{k}"] = {key:float(np.mean([m[key] for m in quality[:k]])) for key in quality[0]}
    output = OUT/"controlled"/cid
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prior.astype("uint8")*255).save(output/"prior.png")
    write_json(output/"result.json", dict(case_id=cid, target_id=row["target_id"], source=row["source"],
               label=row["label"], device=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
               scores=scores, weighted=weighted, surface_losses=surface["losses"], indices=choices,
               fallback=fallbacks, candidate_metrics=quality, metrics=values, prior_iou=overlap(prior,truth),
               prior_hidden_iou=overlap(prior & occlusion,truth & occlusion),
               reference_metrics_sha256=sha256(TEST/"metrics"/(cid+".json"))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--worker-count", type=int, required=True)
    args = p.parse_args()
    assert sha256(OUT/"prior.pt") == read_json(OUT/"training.json")["checkpoint_sha256"]
    model = make_network().cuda().eval()
    model.load_state_dict(torch.load(OUT/"prior.pt", map_location="cpu"))
    for row in read_json(TEST/"cases.json")[args.worker_index::args.worker_count]:
        evaluate(model, row)


if __name__ == "__main__":
    main()

