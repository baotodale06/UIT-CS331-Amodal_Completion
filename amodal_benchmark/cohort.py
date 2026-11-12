"""Explicit research cohorts without changing official annotations or query context."""
import json
from pathlib import Path

from .dataset import sha256


def select_manifest_rows(rows, path):
    """Select a nonempty, unique inventory in official order, failing closed."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    identifiers = manifest.get("target_ids") if isinstance(manifest, dict) else None
    if not isinstance(identifiers, list) or not identifiers or any(
            not isinstance(value, str) for value in identifiers):
        raise ValueError("Cohort requires a nonempty target_ids string list")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Cohort contains duplicate target IDs")
    by_id = {row["id"]: row for row in rows}
    unknown = set(identifiers) - set(by_id)
    if unknown:
        raise ValueError("Cohort target outside selected official sources: " + ", ".join(sorted(unknown)))
    selected = set(identifiers)
    result = [row for row in rows if row["id"] in selected]
    if identifiers != [row["id"] for row in result]:
        raise ValueError("Cohort target IDs must retain official annotation order")
    return result


def cohort_identity(path):
    return {"manifest_sha256": sha256(path),
            "target_ids": json.loads(Path(path).read_text(encoding="utf-8"))["target_ids"]}
