"""Check the frozen natural-image subset against the images used in controlled studies."""
from collections import defaultdict
import json
from pathlib import Path

from prepare_surface_confirmation import split_audit
from amodal_benchmark.dataset import read_official, sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/open_world_amodal"
OUT = ROOT / "output/shape-20261121"
ARCHIVE = ROOT / "output/surface-annotated-full-review-20261114/evidence"


def read(path):
    return json.loads(path.read_bytes())


def main():
    training = read(OUT / "training.json")
    inventory_path = ARCHIVE / "unseen-inventory.json"
    inventory = read(inventory_path)
    if sha256(inventory_path) != read(ARCHIVE / "independent-audit.json")["unseen_inventory_sha256"]:
        raise ValueError("Historical target inventory changed")
    cohort = read(ROOT / "output/surface-annotated-20261112-design/cohort.json")
    previous = []
    trained = []
    for name, digest in cohort["previous_manifest_hashes"].items():
        path = ROOT / name.replace("\\", "/")
        if sha256(path) != digest:
            raise ValueError(f"Controlled-study manifest changed: {path}")
        cases = read(path)
        previous.extend(cases)
        folder = path.parent.relative_to(ROOT).as_posix()
        if folder in training["sources"]:
            if training["sources"][folder] != digest:
                raise ValueError("Training receipt differs from controlled-study manifest")
            trained.extend(cases)
    training_ids = {r["image_identity"] for r in trained}
    if len(trained) != 183 or training_ids != set(training["identities"]):
        raise ValueError("Shape training records disagree")
    previous_ids = {r["image_identity"] for r in previous}
    previous_hashes = {r["source_sha256"] for r in previous}
    training_hashes = {r["source_sha256"] for r in trained}
    checksums = read(DATA / "image_checksums.json")
    rows, _ = read_official(DATA)
    ids = set(inventory["target_ids"])
    held = [r for r in rows if r["id"] in ids]
    if [r["id"] for r in held] != inventory["target_ids"] or len(held) != 2017:
        raise ValueError("Held-out records disagree with the official benchmark")
    paths = defaultdict(list)
    for row in held:
        paths[row["image"]].append(row)
    identities = {r["source"] + "/" + r["original_id"] for r in held}
    identity_by_path = {r["image"]: r["source"] + "/" + r["original_id"] for r in held}
    if len(identities) != inventory["images"] or len(previous_ids) != cohort["previous_images"]:
        raise ValueError("Image identity counts changed")
    # Use a path as the audit key so each target-specific source file is screened.
    fresh = [dict(case_id=path, image_identity=path, source_image=path, source_sha256=checksums[path])
             for path in paths]
    screen = split_audit(previous, fresh, DATA)
    duplicate_ids = [(identity_by_path[a], identity_by_path[b]) for a, b in screen["exact_within"]
                     if identity_by_path[a] != identity_by_path[b]]
    matched_paths = {pair[0] for pair in screen["exact_cross"] + screen["near_cross"]}
    matches = [r["id"] for r in held if r["image"] in matched_paths]
    train_matches = {pair[0] for pair in screen["exact_cross"] + screen["near_cross"] if pair[1] in training_ids}
    full_training_overlap = [r for r in rows if r["source"] + "/" + r["original_id"] in training_ids
                             or checksums[r["image"]] in training_hashes]
    receipt = dict(
        inventory_sha256=sha256(inventory_path), training_sha256=sha256(OUT / "training.json"),
        previous_manifest_hashes=cohort["previous_manifest_hashes"], previous_images=len(previous_ids),
        training_images=len(training_ids), full_training_overlap_targets=len(full_training_overlap),
        targets=len(held), image_identities=len(identities), source_files=len(paths),
        shared_identities=sorted(identities & previous_ids),
        shared_file_hashes=sorted({checksums[path] for path in paths} & previous_hashes),
        within_subset_identical_pairs_across_ids=duplicate_ids,
        cross_study_flagged_targets=matches,
        shape_training_flagged_targets=[r["id"] for r in held if r["image"] in train_matches],
        decoded_and_thumbnail_screen=screen,
    )
    write_json(OUT / "unseen-image-audit.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("previous_manifest_hashes", "decoded_and_thumbnail_screen")}, indent=2))


if __name__ == "__main__":
    main()
