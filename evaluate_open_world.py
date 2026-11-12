#!/usr/bin/env python3
"""Run the CS331 PD-MC implementation on the verified open-world benchmark."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

from amodal_benchmark.dataset import SOURCES, audit_files, read_official, sha256, write_json
from amodal_benchmark.detection import (DETECTION_CONFIG, annotation_queries, expanded_segmenter,
                                       query_digest)
from amodal_benchmark.inference import (SD_MODEL, SD_REVISION, UPSTREAM_COMMIT,
                                       complete_target, load_upstream)
from amodal_benchmark.matching import MATCHING_CONFIG, choose_semantic_target
from amodal_benchmark.parallel import output_lock, partition_rows, plan_devices, run_workers
from amodal_benchmark.protocol import (METRICS, aggregate, choose_target,
                                      ensure_run_config, failure_metrics, failure_stage, read_cached_result)
from amodal_benchmark.references import cached_reference, reference_paths, save_reference
from amodal_benchmark.reproduction import data_identity, paper_comparison, render_comparison


def parse_args(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_path = config_parser.parse_known_args(raw)[0].config
    if config_path is not None:
        from amodal_benchmark.config import evaluator_arguments, load_config
        raw = evaluator_arguments(load_config(config_path)) + raw
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="Validated JSON experiment contract; explicit flags override it")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo-dir", type=Path)
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("--pipeline-profile", choices=("benchmark", "released-sam"), default="benchmark",
                   help="released-sam uses released PD-MC gates, label matching and SAM appearance references; no LISA")
    p.add_argument("--background-sizing", choices=("released", "auto"),
                   help="Auto enlarges undersized Mixed Context backdrops; default: released for released-sam, auto for benchmark")
    p.add_argument("--reuse-results-from", type=Path,
                   help="Reuse verified released-sam outputs in a new run when enabling --background-sizing auto")
    p.add_argument("--classes-file", type=Path, help="Default: pinned upstream classes.txt; changing it changes the baseline")
    p.add_argument("--detector-vocabulary", choices=("released", "annotated"), default="released",
                   help="Annotated adds missing per-image annotation categories as separate detector queries")
    p.add_argument("--sources", default=",".join(SOURCES), help="Comma-separated VG,COCOA,free_images,laion")
    p.add_argument("--limit", type=int, help="Smoke-test only; marks the report as a subset")
    p.add_argument("--target-manifest", type=Path,
                   help="Frozen JSON target_ids cohort; retains official order and full per-image annotation queries")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mc-timestep", type=int, default=35)
    p.add_argument("--max-iter", type=int, default=5, help="Released max_iter_id (inclusive, same as upstream)")
    p.add_argument("--sd-model", default=SD_MODEL)
    p.add_argument("--surface-freeze", type=Path, help="Opt-in frozen Surface-PMC progressive selector")
    p.add_argument("--surface-dreamsim-cache", type=Path, help="Pinned DreamSim cache for surface features")
    p.add_argument("--surface-development-review", action="store_true",
                   help="Permit a provisional surface policy only in a limited development smoke")
    p.add_argument("--target-selection", choices=("auto", "label", "label-semantic", "reference", "reference-best"), default="auto",
                   help="Auto uses best-IoU reference matching; reference retains the old near-tie veto")
    p.add_argument("--boundary-check", choices=("paper", "released"),
                   help="Paper fixes reversed boundary coordinates; released reproduces the old run")
    p.add_argument("--metric-alignment", choices=("released", "original-frame"), default="released",
                   help="Default compares the generated sd_img_cut-style crop as in eva_amodal.py")
    p.add_argument("--inference-only", action="store_true", help="Generate outputs without unavailable reference metrics")
    p.add_argument("--score-only", action="store_true", help="Score saved results without loading the diffusion pipeline")
    p.add_argument("--allow-incomplete", action="store_true", help="Keep missing-image rows explicitly; report a partial benchmark")
    p.add_argument("--check-only", action="store_true", help="CPU data preflight; no model loading or output changes")
    p.add_argument("--workers", default="1", help="Independent PD-MC GPU workers: auto or a positive count (default: 1)")
    p.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    p.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    args = p.parse_args(raw)
    if args.surface_freeze:
        if not args.surface_dreamsim_cache or args.pipeline_profile != "released-sam":
            p.error("Surface-PMC requires --surface-dreamsim-cache and --pipeline-profile released-sam")
        if args.config or args.reuse_results_from:
            p.error("Surface-PMC uses a separate CLI experiment with fresh outputs")
    elif args.surface_development_review or args.surface_dreamsim_cache:
        p.error("Surface options require --surface-freeze")
    if args.surface_development_review and (not args.limit or args.limit > 32):
        p.error("Provisional surface review requires an explicit smoke limit of at most 32")
    if args.pipeline_profile == "released-sam":
        if args.target_selection not in ("auto", "label", "label-semantic") or args.boundary_check == "paper":
            p.error("released-sam requires label or label-semantic matching and --boundary-check released")
        if args.target_selection == "auto":
            args.target_selection = "label"
        args.boundary_check = "released"
    elif args.boundary_check is None:
        args.boundary_check = "paper"
    if args.target_selection == "label-semantic" and args.pipeline_profile != "released-sam":
        p.error("--target-selection label-semantic requires the released-sam profile")
    if args.detector_vocabulary == "annotated" and (args.pipeline_profile != "released-sam" or
                                                   args.target_selection != "label-semantic"):
        p.error("--detector-vocabulary annotated requires released-sam and label-semantic matching")
    if args.background_sizing is None:
        args.background_sizing = "released" if args.pipeline_profile == "released-sam" else "auto"
    if args.pipeline_profile == "benchmark" and args.background_sizing != "auto":
        p.error("--background-sizing released is only supported with the released-sam profile")
    if args.reuse_results_from is not None and (args.pipeline_profile != "released-sam" or
            args.background_sizing != "auto" or args.score_only or args.worker_index is not None):
        p.error("--reuse-results-from requires coordinator inference with released-sam and --background-sizing auto")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.target_manifest and (args.limit is not None or args.config):
        p.error("--target-manifest cannot be combined with --limit or --config")
    if args.max_iter < 1 or not 0 <= args.mc_timestep <= 49 or args.seed < 0:
        p.error("Require --max-iter >= 1, 0 <= --mc-timestep <= 49, --seed >= 0")
    if args.inference_only and args.score_only:
        p.error("--inference-only and --score-only cannot be combined")
    if args.workers != "auto" and (not args.workers.isascii() or
                                   not args.workers.isdigit() or int(args.workers) < 1):
        p.error("--workers must be auto or a positive integer")
    if args.worker_index is not None:
        if (args.worker_count < 2 or not 0 <= args.worker_index < args.worker_count or
                args.workers != "1" or not args.inference_only or args.score_only):
            p.error("Internal workers require a valid index/count, --workers 1 and --inference-only")
    elif args.worker_count != 1:
        p.error("--worker-count requires --worker-index")
    if args.config is not None:
        from amodal_benchmark.config import effective_config, load_config
        effective_config(load_config(args.config), args)
    return args


def preflight(args):
    rows, counts = read_official(args.dataset_root)
    sources = args.sources.split(",")
    if not sources or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        raise ValueError("--sources must list unique names from VG,COCOA,free_images,laion")
    rows = [r for r in rows if r["source"] in sources]
    if args.target_manifest:
        from amodal_benchmark.cohort import select_manifest_rows
        rows = select_manifest_rows(rows, args.target_manifest)
    if args.limit:
        rows = rows[:args.limit]
    if args.worker_index is not None:
        rows = partition_rows(rows, args.worker_count)[args.worker_index]
    sam_references = args.pipeline_profile == "released-sam"
    if sam_references and args.reference_dir is None:
        args.reference_dir = args.output_dir / "visible_references"
    elif args.reference_dir is None and not args.inference_only:
        args.reference_dir = args.dataset_root / "visible_references"
    if args.target_selection == "auto":
        args.target_selection = "reference-best" if args.reference_dir is not None else "label"
    report = audit_files(args.dataset_root, rows, args.reference_dir, decode=True)
    report["reference_records"] = {}
    problems = []
    if sam_references:
        marker = args.reference_dir / "reference_config.json"
        if args.reference_dir.exists() and any(args.reference_dir.iterdir()) and not marker.is_file():
            problems.append("SAM references require their own reference_config.json; use a new reference directory.")
        if marker.is_file():
            config = json.loads(marker.read_text(encoding="utf-8"))
            if config.get("backend") != "grounded-sam":
                problems.append("The reference directory does not contain grounded-sam references.")
            elif args.score_only:
                saved = json.loads((args.output_dir / "run_config.json").read_text(encoding="utf-8"))
                if config != sam_reference_config(saved):
                    problems.append("SAM reference configuration differs from the saved inference run; "
                                    "use the original reference directory.")
        for row in rows:
            image_path = args.dataset_root / row["image"]
            if not image_path.is_file():
                continue
            record = cached_reference(args.reference_dir, row, sha256(image_path))
            if record is not None:
                if record.get("provenance", {}).get("backend") != "grounded-sam":
                    problems.append(f"Non-SAM reference for {row['id']}; use a new reference directory.")
                report["reference_records"][row["id"]] = record
            elif args.score_only:
                problems.append(f"SAM reference missing for {row['id']}; finish inference first.")
    elif not args.inference_only or args.target_selection.startswith("reference"):
        marker = args.reference_dir / "reference_config.json" if args.reference_dir else None
        if marker is not None and marker.is_file():
            if json.loads(marker.read_text(encoding="utf-8")).get("backend") == "grounded-sam":
                raise ValueError("SAM references define a separate protocol; the benchmark profile requires LISA references.")
        if not args.reference_dir:
            problems.append("Generate the visible references with prepare_visible_references.py, "
                            "or use the Slurm benchmark command to prepare them automatically.")
        elif report["missing_references"] or report["corrupt_references"]:
            problems.append(f"Reference RGBA files: {len(report['missing_references'])} missing, "
                            f"{len(report['corrupt_references'])} invalid. The Slurm benchmark command "
                            "generates these with LISA automatically when --reference-dir is omitted.")
        else:
            for row in rows:
                image_path = args.dataset_root / row["image"]
                if not image_path.is_file():
                    continue
                record = cached_reference(args.reference_dir, row, sha256(image_path))
                if record is not None:
                    if record.get("provenance", {}).get("backend") == "grounded-sam":
                        raise ValueError(f"SAM reference for {row['id']} cannot be reported as an independent LISA reference.")
                    report["reference_records"][row["id"]] = record
                if not args.inference_only and args.metric_alignment == "released" and record is None:
                    problems.append("Released-crop scoring needs generated reference records and crops. "
                                    "Run prepare_visible_references.py, or explicitly select --metric-alignment original-frame.")
                    break
    if report["corrupt_images"]:
        problems.append(f"{len(report['corrupt_images'])} corrupt input images; repair the installation.")
    if report["missing_images"] and not args.allow_incomplete:
        problems.append(f"{len(report['missing_images'])} input images missing. Run prepare_open_world.py; "
                        "choose --sources explicitly for an available subset.")
    if problems:
        raise ValueError("\n".join(problems))
    return rows, counts, report


def seed_for(base, identifier):
    return (base + int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "big")) % 2**32


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sam_reference_config(config):
    """Keep generated reference identity consistent across inference and scoring."""
    reference = {
        "schema": 1, "backend": "grounded-sam", "upstream_commit": config["upstream_commit"],
        "upstream_main_sha256": config["upstream_main_sha256"],
        "vocabulary_sha256": config["vocabulary_sha256"], "seed": config["seed"],
        "target_selection": config["target_selection"], "preprocessing": "5x5-erosion/first-crop-v1",
        "checkpoints": {name: value for name, value in config["checkpoints"].items()
                        if name.replace("\\", "/").startswith("Grounded-Segment-Anything/")}}
    if "label_matching" in config:
        reference["label_matching"] = config["label_matching"]
    if "detector_vocabulary" in config:
        reference["detector_vocabulary"] = config["detector_vocabulary"]
    return reference


def save_summary(output, rows, counts, audit, args):
    summary = {"selected_targets": len(rows), "paper_targets": 2565,
               "full_image_coverage": len(rows) == 2565 and not audit["missing_images"],
               "evaluation_performed": not args.inference_only,
               "exact_table_reproduction": False,
               "boundary_check": args.boundary_check,
               "target_selection": args.target_selection,
               "metric_alignment": args.metric_alignment,
               "protocol_notes": [
                   "Visible references are generated using LISA and the released 5x5 erosion/first-crop procedure.",
                   "Target selection is an explicitly recorded adapter rule, not a claimed author-supplied implementation.",
                   "Paper boundary mode repairs the released min/max coordinate reversal, following CS331 Appendix A.1.",
                   "Released alignment compares sd_img_cut-style crops; original-frame alignment is an optional separate protocol.",
                   "Empty LISA references are reported separately and excluded from model-failure and appearance denominators.",
                   "CLIP is raw ViT-B/32 logits on successful completions; failure scores are unspecified by the paper. See CLIP_n.",
                   "Appearance penalties on failed completions: LPIPS=1, VGG similarity=0, SSIM=0.",
                   "No human-preference or original CS331 pseudo-occlusion benchmark scores are inferred."],
               "source_counts_in_paper": counts, "statistics": aggregate(rows, detailed=True),
               "data_identity": data_identity(rows, audit)}
    if args.target_manifest:
        from amodal_benchmark.cohort import cohort_identity
        summary["research_cohort"] = cohort_identity(args.target_manifest)
    if args.pipeline_profile == "released-sam":
        summary.update(pipeline_profile="released-sam", reference_backend="grounded-sam",
            resize_background=args.background_sizing == "auto",
            protocol_notes=[
                "Released PD-MC Grounding DINO + SAM segmentation, InstaOrder, LaMa and Mixed Context diffusion.",
                ("Released boundary behavior is retained; undersized backdrops are enlarged when required."
                 if args.background_sizing == "auto" else
                 "Released boundary behavior is retained; the local backdrop enlargement repair is disabled."),
                "Annotated labels select a unique detection; missing or ambiguous detections are model failures.",
                "Appearance references use the selected initial SAM mask with 5x5 erosion and first-crop preprocessing.",
                "SAM references differ from the OpenWorld paper's LISA references; appearance scores are a separate protocol.",
                "Failed completions receive LPIPS=1, VGG=0, SSIM=0, including failed target detections.",
                "Successful completions with empty eroded references receive CLIP only; inspect all metric denominators.",
                "CLIP is raw ViT-B/32 logits on successful completions only; visible masks are not hidden-pixel ground truth."])
        if args.target_selection == "label-semantic":
            summary["protocol_notes"][2] = (
                "Target identity uses versioned class aliases and exact-mask deduplication; distinct instances remain ambiguous.")
            summary["label_matching"] = json.loads((output / "run_config.json").read_text())["label_matching"]
        saved = json.loads((output / "run_config.json").read_text())
        if "detector_vocabulary" in saved:
            summary["detector_vocabulary"] = saved["detector_vocabulary"]
            summary["protocol_notes"].append(
                "Annotation-guided vocabulary experiment: scene categories plus separate queries for "
                "missing categories annotated in each image; the same queries run during completion checks.")
    summary["protocol_notes"].append(
        "Success denotes a saved output; completed_outputs and iteration_limited_outputs report its stop state separately.")
    write_json(output / "summary.json", summary)
    comparison = paper_comparison(summary)
    write_json(output / "paper_comparison.json", comparison)
    (output / "paper_comparison.md").write_text(render_comparison({"runs": [
        {"run_dir": str(output), "run_config": {"boundary_check": args.boundary_check,
         "target_selection": args.target_selection, "seed": args.seed}, "comparison": comparison}]}), encoding="utf-8")
    fields = ["id", "source", "label", "status", "failure_reason", "failure_stage", "reference_status", "seed",
              "target_selection", "selected_iou", "detected_class", "detector_score", "iterations",
              "iteration_limit_reached", *METRICS]
    with (output / "per_target.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "failure_stage": failure_stage(row)} for row in rows)
    fields = ["Source", "Targets", "Saved outputs", "Completed", "At iteration limit", "Unknown stop",
              "Failures", "Failure rate", "Missing", "CLIP", "CLIP n", "LPIPS", "VGG", "SSIM"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for source, stat in summary["statistics"].items():
        values = [source, stat["targets"], stat["successes"], stat["completed_outputs"],
                  stat["iteration_limited_outputs"], stat["completion_state_unknown"], stat["failures"],
                  f"{100 * stat['failure_rate']:.2f}%" if stat["failure_rate"] is not None else None,
                  stat["missing_images"],
                  stat["CLIP"], stat["CLIP_n"], stat["LPIPS"], stat["Feature Similarity"], stat["SSIM"]]
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else "unavailable" if v is None else str(v)
                                       for v in values) + " |")
    lines += ["", "CLIP uses successful completions only; inspect per-metric denominators in summary.json.",
              "The report records an independent reproduction attempt; it does not certify exact Table 3 equivalence."]
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def worker_command(args):
    """Forward the same experiment settings, with only inference done in workers."""
    command = [sys.executable, str(Path(__file__).resolve()),
               "--dataset-root", str(args.dataset_root), "--output-dir", str(args.output_dir),
               "--repo-dir", str(args.repo_dir.resolve()), "--sources", args.sources,
               "--seed", str(args.seed), "--mc-timestep", str(args.mc_timestep),
               "--max-iter", str(args.max_iter), "--sd-model", args.sd_model,
               "--target-selection", args.target_selection, "--metric-alignment", args.metric_alignment,
               "--boundary-check", args.boundary_check,
               "--pipeline-profile", args.pipeline_profile,
               "--detector-vocabulary", args.detector_vocabulary,
               "--background-sizing", args.background_sizing,
               "--workers", "1", "--inference-only"]
    if args.reference_dir is not None:
        command += ["--reference-dir", str(args.reference_dir)]
    if args.classes_file is not None:
        command += ["--classes-file", str(args.classes_file.resolve())]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.target_manifest:
        command += ["--target-manifest", str(args.target_manifest.resolve())]
    if args.allow_incomplete:
        command.append("--allow-incomplete")
    if args.surface_freeze:
        command += ["--surface-freeze", str(args.surface_freeze.resolve()),
                    "--surface-dreamsim-cache", str(args.surface_dreamsim_cache.resolve())]
        if args.surface_development_review:
            command.append("--surface-development-review")
    return command


def main(argv=None):
    args = parse_args(argv)
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.target_manifest:
        args.target_manifest = args.target_manifest.resolve()
    if args.surface_freeze:
        args.surface_freeze = args.surface_freeze.resolve()
        args.surface_dreamsim_cache = args.surface_dreamsim_cache.resolve()
    if Path(args.sd_model).is_dir():
        args.sd_model = str(Path(args.sd_model).resolve())
    if args.reference_dir:
        args.reference_dir = args.reference_dir.resolve()
    if args.score_only:
        config_path = args.output_dir / "run_config.json"
        if not config_path.is_file():
            raise ValueError("--score-only requires an existing inference run_config.json")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        saved_profile = saved_config.get("pipeline_profile", "benchmark")
        if args.pipeline_profile != saved_profile:
            raise ValueError("--score-only pipeline profile differs from the saved inference run")
        args.target_selection = saved_config["target_selection"]
        args.boundary_check = saved_config.get("boundary_check", "released")
        args.seed = saved_config["seed"]
        args.mc_timestep = saved_config["mc_timestep"]
        args.max_iter = saved_config["max_iter_id"]
        args.sd_model = saved_config["sd_model"]
        args.background_sizing = "auto" if saved_config.get("resize_background", True) else "released"
        args.detector_vocabulary = saved_config.get("detector_vocabulary", {}).get("mode", "released")
    rows, counts, audit = preflight(args)
    if args.check_only:
        print(f"Data preflight passed for {len(rows)} selected targets; "
              f"{len(audit['missing_images'])} explicitly missing images. No models loaded.")
        return 0
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        if not (output / "run_config.json").is_file():
            raise ValueError("Internal workers require a coordinator-created run_config.json")
        return evaluate(args, rows, counts, audit)
    with output_lock(output):
        from amodal_benchmark.provenance import execution_snapshot
        if args.target_manifest:
            from amodal_benchmark.cohort import cohort_identity
            identity = cohort_identity(args.target_manifest)
            manifest_copy = output / "cohort_manifests" / (identity["manifest_sha256"] + ".json")
            manifest_copy.parent.mkdir(parents=True, exist_ok=True)
            manifest_copy.write_bytes(args.target_manifest.read_bytes())
        resolved = None
        if args.config is not None:
            from amodal_benchmark.config import effective_config, load_config
            resolved = effective_config(load_config(args.config), args)
        with execution_snapshot(output, args, rows, resolved):
            return evaluate(args, rows, counts, audit)


def evaluate(args, rows, counts, audit):
    output = args.output_dir
    sam_references = args.pipeline_profile == "released-sam"
    upstream = models = None
    original_segment = None
    image_queries = {}
    cached_image = None
    cache = None
    records = []
    if not args.score_only:
        if args.repo_dir is None:
            raise ValueError("--repo-dir is required for inference; use the Slurm launcher to provision it")
        repo = args.repo_dir.resolve()
        vocabulary_path = (args.classes_file or repo / "classes.txt").resolve()
        vocabulary = vocabulary_path.read_text(encoding="utf-8").splitlines()
        if not vocabulary:
            raise ValueError("Detection vocabulary is empty")
        if args.detector_vocabulary == "annotated":
            all_rows, _ = read_official(args.dataset_root)
            image_queries = annotation_queries(all_rows, vocabulary)
        checkpoints = [repo / "Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
                       repo / "Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
                       repo / "InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar",
                       repo / "lama/big-lama/models/best.ckpt"]
        config = {"schema": 1, "adapter_version": 3, "upstream_commit": UPSTREAM_COMMIT,
                  "upstream_main_sha256": sha256(repo / "main.py"), "sd_model": args.sd_model,
                  "sd_revision": SD_REVISION if args.sd_model == SD_MODEL else None,
                  "mc_timestep": args.mc_timestep, "max_iter_id": args.max_iter, "seed": args.seed,
                  "target_selection": args.target_selection, "vocabulary_sha256": sha256(vocabulary_path),
                  "annotation_sha256": {s: SOURCES[s][4] for s in SOURCES}}
        if args.boundary_check != "released" or args.target_selection == "reference-best":
            config.update(adapter_version=4, boundary_check=args.boundary_check)
        if sam_references:
            config.update(adapter_version=5, pipeline_profile="released-sam", boundary_check="released",
                          reference_backend="grounded-sam", resize_background=args.background_sizing == "auto")
            if args.target_selection == "label-semantic":
                config.update(adapter_version=6, label_matching={**MATCHING_CONFIG,
                    "implementation_sha256": sha256(Path(__file__).resolve().parent / "amodal_benchmark/matching.py")})
        if args.detector_vocabulary == "annotated":
            config.update(adapter_version=7, detector_vocabulary={**DETECTION_CONFIG,
                "image_queries_sha256": query_digest(image_queries),
                "implementation_sha256": sha256(Path(__file__).resolve().parent / "amodal_benchmark/detection.py")})
        if args.surface_freeze:
            from amodal_benchmark.surface_runtime import surface_contract
            config['surface_policy'] = surface_contract(args.surface_freeze, args.surface_dreamsim_cache,
                                                        args.surface_development_review)
        if args.worker_index is None:
            config["checkpoints"] = {str(p.relative_to(repo)): sha256(p) for p in checkpoints}
            if Path(args.sd_model).is_dir():
                config["local_sd_files"] = {str(p.relative_to(args.sd_model)): sha256(p)
                                            for p in Path(args.sd_model).rglob("*") if p.is_file()}
            ensure_run_config(output / "run_config.json", config)
            if sam_references:
                ensure_run_config(args.reference_dir / "reference_config.json", sam_reference_config(config))
                if args.reuse_results_from is not None:
                    from amodal_benchmark.reuse import reuse_completed
                    reuse_completed(args.reuse_results_from, output, config, rows,
                                    args.dataset_root, args.reference_dir)
        else:
            # The coordinator has already verified shared multi-GB files.
            # Workers check their semantic arguments without rescanning them.
            saved = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in config.items()):
                raise ValueError("Worker settings differ from the coordinator's run configuration")
    elif not (output / "run_config.json").is_file():
        raise ValueError("--score-only requires an existing inference run_config.json")
    if args.worker_index is None and not args.score_only and args.workers != "1":
        import torch
        devices = plan_devices(args.workers, torch.cuda.device_count(),
                               os.environ.get("CUDA_VISIBLE_DEVICES"),
                               len({row["image"] for row in rows}))
        if len(devices) > 1:
            execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
            print(f"Running PD-MC inference on {len(devices)} allocated GPUs. "
                  "Model settings and per-target seeds are unchanged.", flush=True)
            run_workers(worker_command(args), devices, output / "workers" / execution_id)
            # Collect validated records in the original annotation order; only
            # this coordinator scores and writes the full benchmark reports.
            args.score_only = True
            print("All PD-MC workers finished. Collecting results and computing metrics.", flush=True)
    for n, row in enumerate(rows, 1):
        folder = output / "targets" / row["id"]
        result_path = folder / "result.json"
        image_path = args.dataset_root / row["image"]
        image_hash = sha256(image_path) if image_path.exists() else None
        record = {**row, "input_sha256": image_hash}
        prepared_reference = audit["reference_records"].get(row["id"])
        if sam_references and prepared_reference is None and image_hash is not None:
            # Workers may have just generated this reference after coordinator preflight.
            prepared_reference = cached_reference(args.reference_dir, row, image_hash)
            if prepared_reference is not None:
                audit["reference_records"][row["id"]] = prepared_reference
        ref_paths = reference_paths(args.reference_dir, row) if args.reference_dir is not None else None
        selection_path = (ref_paths["mask"] if prepared_reference is not None else ref_paths["full"]) if ref_paths else None
        reference_hash = (sha256(selection_path)
                          if args.target_selection.startswith("reference") and image_hash is not None else None)
        cached = read_cached_result(result_path, row, image_hash, reference_hash)
        if cached is not None:
            if sam_references and image_hash is not None:
                if (prepared_reference is None or cached.get("evaluation_reference_artifacts") !=
                        prepared_reference["artifacts"]):
                    raise ValueError(f"SAM evaluation reference missing or changed for {row['id']}; use a new output directory")
            record = cached
        elif args.score_only:
            raise ValueError(f"Inference result is missing for {row['id']}; finish inference first")
        elif image_hash is None:
            record["status"] = "missing_image"
            write_json(result_path, record)
        elif args.target_selection.startswith("reference") and prepared_reference is not None and prepared_reference["status"] != "ready":
            record.update(status="reference_failed", reference_status=prepared_reference["status"],
                          selection_reference_sha256=reference_hash)
            write_json(result_path, record)
        else:
            if upstream is None:
                os.chdir(repo)
                if sam_references:
                    upstream = load_upstream(repo, args.sd_model, args.boundary_check, args.background_sizing == "auto")
                else:
                    upstream = load_upstream(repo, args.sd_model, args.boundary_check)
                models = upstream.load_models(
                    str(repo / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                    str(checkpoints[0]), str(checkpoints[2]), str(repo / "lama/big-lama/config.yaml"),
                    str(checkpoints[3]), args.mc_timestep)
                if args.surface_freeze:
                    from amodal_benchmark.surface_runtime import SurfaceRuntimeFeatures, install_progressive_surface
                    frozen = config['surface_policy']['frozen_method']
                    features = SurfaceRuntimeFeatures(args.surface_dreamsim_cache) if frozen.get('pretrained_visible_features') else None
                    install_progressive_surface(upstream, models, checkpoints[1], frozen, features)
                original_segment = upstream.segment
            if cached_image != row["image"]:
                with Image.open(image_path) as image:
                    pil = image.convert("RGB")
                set_seed(seed_for(args.seed, row["image"]))
                if args.detector_vocabulary == "annotated":
                    upstream.segment = expanded_segmenter(original_segment, vocabulary,
                                                          image_queries.get(row["image"], []))
                tensor = upstream.transform_image(pil)
                rgb, masks, classes, scores = upstream.segment(models[0], upstream.run_sam,
                                                               str(checkpoints[1]), pil, tensor, vocabulary)
                effective_vocabulary = vocabulary
                detection_details = None
                if args.detector_vocabulary == "annotated":
                    effective_vocabulary = list(dict.fromkeys([
                        *vocabulary, *image_queries.get(row["image"], []),
                        *upstream.segment.extra_detected_classes]))
                    detection_details = {"mode": "annotated", "queries": upstream.segment.query_trace,
                                         "eligibility_vocabulary": effective_vocabulary}
                cache = pil, rgb, masks, classes, scores, effective_vocabulary, detection_details
                cached_image = row["image"]
            pil, rgb, masks, classes, scores, effective_vocabulary, detection_details = cache
            if detection_details is not None:
                record["detection_diagnostics"] = detection_details
            ref_alpha = None
            if args.target_selection.startswith("reference"):
                with Image.open(selection_path) as image:
                    ref_alpha = np.array(image.getchannel("A") if image.mode == "RGBA" else image.convert("L"))
                record["selection_reference_sha256"] = reference_hash
            selection_details = {}
            if args.target_selection == "label-semantic":
                target, selection = choose_semantic_target(masks, classes, row["label"], scores, selection_details)
            else:
                target, selection = choose_target(masks if masks is not None else [], classes, row["label"], ref_alpha,
                    reference_policy="best" if args.target_selection == "reference-best" else "strict",
                    diagnostics=selection_details)
            record["selection_diagnostics"] = selection_details
            for candidate in selection_details.get("candidates", []):
                candidate["detector_score"] = float(scores[candidate["index"]])
            record["target_selection"] = selection
            record["seed"] = seed_for(args.seed, row["id"])
            if sam_references:
                mask = masks[target].copy() if target is not None else np.zeros((pil.height, pil.width), dtype=bool)
                if prepared_reference is not None:
                    with Image.open(ref_paths["mask"]) as image:
                        old_mask = np.asarray(image) > 0
                    if not np.array_equal(old_mask, mask):
                        raise ValueError(f"Regenerated SAM reference changed for {row['id']}; use a new reference directory")
                    reference = prepared_reference
                else:
                    reference = save_reference(args.reference_dir, row, np.asarray(pil), mask, image_hash,
                        {"backend": "grounded-sam", "detection_seed": seed_for(args.seed, row["image"]),
                         "target_selection": selection, "selected_index": target,
                         "detected_class": classes[target] if target is not None else None})
                audit["reference_records"][row["id"]] = reference
                record["evaluation_reference_artifacts"] = reference["artifacts"]
            if target is None:
                record["status"] = selection
                record["failure_reason"] = selection_details.get("reason", selection)
            else:
                if "best_iou" in selection_details:
                    record["selected_iou"] = selection_details["best_iou"]
                set_seed(record["seed"])
                record.update(complete_target(
                    upstream, models, pil.copy(), rgb.copy(), masks.copy(), classes, scores, target,
                    effective_vocabulary, image_path, folder, checkpoints[1], repo / "images/gray_wallpaper.jpeg",
                    args.mc_timestep, args.max_iter))
                if record["status"] == "success":
                    record["completion_sha256"] = sha256(folder / "completion.png")
            write_json(result_path, record)
        records.append(record)
        reason = f" ({record['failure_reason']})" if record.get("failure_reason") else ""
        print(f"[{n}/{len(rows)}] {row['id']}: {record['status']}{reason}", flush=True)
    if args.worker_index is not None:
        return 0
    # Free all inference models before loading metric networks on the GPU.
    del models, upstream, cache
    gc.collect()
    if not args.inference_only:
        import torch
        from amodal_benchmark.metrics import MetricModels, CLIP_MODEL, CLIP_REVISION
        torch.cuda.empty_cache()
        ensure_run_config(output / "metric_config.json", {"protocol_version": 3, "clip_model": CLIP_MODEL,
                          "clip_revision": CLIP_REVISION, "clip_failures": "excluded_with_denominator",
                          "alignment": args.metric_alignment,
                          "lpips": "alex/0.1.4", "vgg": "VGG16_IMAGENET1K_V1/features",
                          "ssim_data_range": "prediction.max-prediction.min", "exact_copy_penalties": True})
        metric_models = None
        for record in records:
            if record["status"] in ("missing_image", "reference_failed"):
                continue
            ref_paths = reference_paths(args.reference_dir, record)
            prepared_reference = audit["reference_records"].get(record["id"])
            record["reference_status"] = prepared_reference["status"] if prepared_reference else "external"
            if sam_references and record["status"] != "success":
                # Detection failure also means no SAM reference. It still belongs
                # in the pipeline failure rate and appearance-penalty denominators.
                record.update(failure_metrics())
                continue
            if prepared_reference is not None and prepared_reference["status"] != "ready":
                # Reference segmentation is unavailable; do not penalize PD-MC
                # for the separate model's failure. CLIP can still score a result.
                if record["status"] == "success":
                    from amodal_benchmark.protocol import white_rgb
                    folder = output / "targets" / record["id"]
                    signature = {"reference_sha256": sha256(ref_paths["full"]),
                                 "completion_sha256": record["completion_sha256"], "clip_only": True}
                    saved = folder / "metrics.json"
                    values = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
                    if values.get("inputs") != signature:
                        if metric_models is None:
                            metric_models = MetricModels()
                        with Image.open(folder / "completion.png") as image:
                            score = metric_models.clip_score(white_rgb(np.array(image)), record["label"])
                        values = {"inputs": signature, "metrics": {**dict.fromkeys(METRICS), "CLIP": score}}
                        write_json(saved, values)
                    record.update(values["metrics"])
                continue
            reference = ref_paths["crop"] if args.metric_alignment == "released" else ref_paths["full"]
            if record["status"] != "success":
                record.update(failure_metrics())
                continue
            folder = output / "targets" / record["id"]
            signature = {"reference_sha256": sha256(reference),
                         "completion_sha256": record["completion_sha256"]}
            saved = folder / "metrics.json"
            values = json.loads(saved.read_text()) if saved.exists() else {}
            if values.get("inputs") != signature:
                if metric_models is None:
                    metric_models = MetricModels()
                values = {"inputs": signature,
                          "metrics": metric_models.score(reference, folder / "completion.png", record["label"], record["geometry"], args.metric_alignment)}
                write_json(saved, values)
            record.update(values["metrics"])
    save_summary(output, records, counts, audit, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
