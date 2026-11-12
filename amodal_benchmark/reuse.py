"""Reuse verified, previously defined outputs when enabling backdrop sizing."""
import json
from pathlib import Path
import shutil

from .dataset import sha256, write_json
from .parallel import output_lock
from .protocol import read_cached_result
from .references import cached_reference, reference_paths


def reuse_completed(source, output, config, rows, dataset_root, reference_dir):
    """Import only a released-SAM run whose sole change is the sizing repair.

    The repair leaves all previously supported lookups unchanged. A completed
    old target therefore has the same semantics; interrupted targets are retried.
    Learned metric caches are deliberately recomputed under the new run.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or not source.is_dir():
        raise ValueError("--reuse-results-from requires a different, existing run directory")
    with output_lock(source):
        config_path = source / "run_config.json"
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {**saved, "resize_background": True}
        if (saved.get("pipeline_profile") != "released-sam" or
                saved.get("resize_background") is not False or config != expected):
            raise ValueError("Reuse settings differ beyond enabling background sizing; use identical inference settings")
        source_refs = source / "visible_references"
        if json.loads((source_refs / "reference_config.json").read_text(encoding="utf-8")) != json.loads(
                (reference_dir / "reference_config.json").read_text(encoding="utf-8")):
            raise ValueError("Reuse reference settings differ; the source must use its default visible_references directory")
        manifest_path = output / "reused_results.json"
        manifest = {"source_run": str(source), "source_config_sha256": sha256(config_path),
                    "compatible_change": {"resize_background": {"from": False, "to": True}},
                    "target_ids": [], "pending_target_ids": []}
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(previous.get(key) != manifest[key] for key in ("source_run", "source_config_sha256")):
                raise ValueError("Reuse provenance differs from the previous import")
            manifest = previous
        completed_ids = set(manifest["target_ids"])
        pending_ids = set(manifest.get("pending_target_ids", []))
        pending = []
        for row in rows:
            destination = output / "targets" / row["id"]
            image = dataset_root / row["image"]
            published = (destination / "result.json").exists()
            if (published and row["id"] not in pending_ids) or not image.is_file():
                continue
            image_hash = sha256(image)
            original = source / "targets" / row["id"]
            record = read_cached_result(original / "result.json", row, image_hash)
            if record is None:
                continue
            reference = cached_reference(source_refs, row, image_hash)
            if (reference is None or reference.get("provenance", {}).get("backend") != "grounded-sam" or
                    reference["artifacts"] != record.get("evaluation_reference_artifacts")):
                raise ValueError(f"Reuse reference missing or changed for {row['id']}")
            existing = cached_reference(reference_dir, row, image_hash)
            if existing is not None and existing["artifacts"] != reference["artifacts"]:
                raise ValueError(f"Reuse reference differs from the destination for {row['id']}")
            if published:
                # A prior import may have published its result just before the
                # manifest update was interrupted. Verify it before adopting it.
                recovered = read_cached_result(destination / "result.json", row, image_hash)
                if recovered != record or existing is None:
                    raise ValueError(f"Interrupted reuse result differs from its source for {row['id']}")
                completed_ids.add(row["id"])
                pending_ids.discard(row["id"])
            else:
                pending.append((row, record, existing is None, reference))
                pending_ids.add(row["id"])
        # Validate the entire selected import before writing any result artifacts.
        # Journal intended imports before publication so their provenance can be
        # recovered even if the process stops between result and manifest writes.
        manifest.update(target_ids=sorted(completed_ids), pending_target_ids=sorted(pending_ids))
        write_json(manifest_path, manifest)
        for row, record, copy_reference, reference in pending:
            if copy_reference:
                original_paths = reference_paths(source_refs, row)
                destination_paths = reference_paths(reference_dir, row)
                for name in ("full", "crop", "mask"):
                    path = destination_paths[name]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original_paths[name], path)
                # Publish the record atomically and last. A partial JSON record
                # would stop preflight before an interrupted import can recover.
                write_json(destination_paths["record"], reference)
            original = source / "targets" / row["id"]
            destination = output / "targets" / row["id"]
            destination.mkdir(parents=True, exist_ok=True)
            if record["status"] == "success":
                for name in ("completion.png", "completion.jpg", "mask.png"):
                    if (original / name).is_file():
                        shutil.copy2(original / name, destination / name)
            write_json(destination / "result.json", record)
            completed_ids.add(row["id"])
            pending_ids.discard(row["id"])
            manifest.update(target_ids=sorted(completed_ids), pending_target_ids=sorted(pending_ids))
            write_json(manifest_path, manifest)
        print(f"Reused {len(pending)} verified target result(s) from {source}; source outputs were preserved.", flush=True)
