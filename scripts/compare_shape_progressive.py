"""Verify progressive Shape runs and compute matched estimates with the frozen Surface bootstrap."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from math import isclose
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256, write_json, read_official
from amodal_benchmark.cohort import select_manifest_rows
from amodal_benchmark.protocol import failure_metrics, METRICS
from amodal_benchmark.provenance import digest
from amodal_benchmark.references import reference_paths
from amodal_benchmark.verify import verify_run
from amodal_benchmark.shape import choose, score_masks
from amodal_benchmark.surface_ensemble import EnsembleConfig, refine_surface_pool
from amodal_benchmark.surface_refinement import RefinementConfig, predict_rank, refine_surface
from amodal_benchmark.surface_runtime import progressive_seed, project_original_observation
from scripts.audit_surface_progressive_result import estimate
from scripts.snapshot_shape_timeout import file_inventory, verify_timeout_snapshots

# First 1,024 official IDs. The independent review used 932 archived native
# results from job 44946 and 92 results from its verified cohort snapshot.
SHAPE_SELECTOR_SNAPSHOT = {
    "targets": 1024,
    "target_ids_sha256": "95de7c8a08d2bf86e450b1bdd7ace9b66e81aedc4aff7a18e08d66a17417ad6b",
    "result_artifacts_sha256": "9d48cd9c284e0532e5a460080ee674bea7c14f384576725a7ec1bb3e9592bb0b",
}


def read(path):
    return json.loads(path.read_text())


def verify_selector_snapshot(execution, snapshot):
    ids = execution["target_ids"][:snapshot["targets"]]
    order_digest = digest(ids)
    if (execution["status"] != "complete" or len(ids) != snapshot["targets"]
            or order_digest != snapshot["target_ids_sha256"]):
        raise ValueError("The closed execution does not match the archived selector target order")
    hashes = execution["result_artifacts"]
    if not set(ids).issubset(hashes):
        raise ValueError("The execution is missing archived selector records")
    observed = digest({tid: hashes[tid] for tid in ids})
    if observed != snapshot["result_artifacts_sha256"]:
        raise ValueError("Final target records differ from the archived selector review")
    return dict(targets=len(ids), target_ids_sha256=order_digest, result_artifacts_sha256=observed)


def compare_first_pass(shape_dir, combined_dir):
    shape_path = min(shape_dir.glob("iteration-*/decision.json"), default=None)
    combined_path = min(combined_dir.glob("iteration-*/decision.json"), default=None)
    if shape_path is None or combined_path is None:
        if shape_path is not None:
            return "shape_only"
        return "combined_only" if combined_path is not None else "neither_sampled"
    shape = read(shape_path)
    combined = read(combined_path)
    names = ["observed.png", "visible.png", "occlusion.png", "prior.png"]
    names.extend(f"{kind}-{i}.png" for kind in ("candidate","mask") for i in range(8))
    matching = all(shape[k] == combined[k] for k in ("iteration","crop_rc","target_seed","seeds"))
    if not matching or any(shape["artifacts"][n] != combined["artifacts"][n] for n in names):
        return "different_inputs_or_pool"
    return "same_slot" if shape["index"] == combined["index"] else "changed_slot"


def execution_scores(run, execution):
    path = run / "per_target.csv"
    if sha256(path) != execution["artifacts"]["per_target.csv"]:
        raise ValueError(f"Target scores differ from the verified execution for {run.name}")
    with path.open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric]) if row[metric] else None
    scores = {row["id"]: row for row in rows}
    if len(scores) != len(rows):
        raise ValueError(f"Duplicate target in the score table for {run.name}")
    return scores


def verify_vg_scores(run, execution, coverage, replay):
    ids = [tid for tid in execution["target_ids"] if tid.startswith("VG/")]
    if (execution["status"] != "complete" or not ids or ids != [r["id"] for r in coverage]
            or ids != [r["id"] for r in replay]):
        raise ValueError("Closed Visual Genome target order differs from the CPU audits")
    scores = execution_scores(run, execution)
    saved = 0
    for snapshot, early in zip(coverage, replay):
        tid = early["id"]
        native = (run / "targets" / tid / "result.json").read_bytes()
        checksum = hashlib.sha256(native).hexdigest()
        if checksum != execution["result_artifacts"][tid] or checksum != snapshot["result_sha256"]:
            raise ValueError(f"Visual Genome native result changed for {tid}")
        record = json.loads(native)
        row = scores[tid]
        if (record["id"] != tid or record["status"] != row["status"] or row["status"] != snapshot["shape"]
                or row["status"] != early["status"] or row["SSIM"] is None
                or not isclose(row["SSIM"], early["SSIM"], rel_tol=0., abs_tol=1e-12)):
            raise ValueError(f"Visual Genome status or SSIM changed for {tid}")
        if row["status"] == "success":
            if (early["completion_sha256"] != record["completion_sha256"]
                    or early["crop_sha256"] != record["evaluation_reference_artifacts"]["crop"]):
                raise ValueError(f"Visual Genome SSIM input changed for {tid}")
            saved += 1
    return dict(targets=len(ids), saved=saved, scores_sha256=execution["artifacts"]["per_target.csv"])


def verify_full_registration(run, registry, verification):
    """Match the saved ledger entry to the execution and its verified full scores."""
    execution = read(run / "execution.json")
    if (execution["status"] != "complete" or not verification["full_evaluation"]
            or not verification["verified"] or not verification["provenance_verified"]
            or verification["selected_targets"] != 2565):
        raise ValueError("Registration audit requires a verified full execution")
    path = verification["run_dir"]
    entry_id = digest({"run": path, "execution": execution["id"]})
    raw = registry.read_bytes()
    rows = csv.DictReader(io.StringIO(raw.decode("utf8"), newline=""))
    entries = [row for row in rows if row["entry_id"] == entry_id]
    if len(entries) != 1:
        raise ValueError(f"Expected one registration for execution {execution['id']}")
    entry = entries[0]
    identity = verification["data_identity"]
    protocol = {"inference": read(run / "run_config.json"), "metrics": read(run / "metric_config.json"),
                "data": identity}
    expected = {
        "run_dir": path, "status": "complete", "protocol_sha256": digest(protocol),
        "summary_sha256": sha256(run / "summary.json"), "inputs_sha256": identity["inputs_sha256"],
        "references_sha256": identity["references_sha256"],
    }
    overall = verification["statistics"]["Overall"]
    for key in ("targets", "successes", "failures", *(metric + "_n" for metric in METRICS)):
        expected[key] = str(overall[key])
    for key, value in expected.items():
        if entry[key] != value:
            raise ValueError(f"Registration differs from the verified run for {key}")
    for key in ("failure_rate", *METRICS):
        value = overall[key]
        if value is None and entry[key] == "":
            continue
        if value is None or not entry[key] or not isclose(float(entry[key]), value, abs_tol=1e-12, rel_tol=0.):
            raise ValueError(f"Registration differs from the verified run for {key}")
    return dict(entry_id=entry_id, execution_id=execution["id"], registry_sha256=hashlib.sha256(raw).hexdigest())


def verify_cohort_reuse(run, snapshot, cohort_ids):
    previous = read(snapshot/"execution.json")
    full = read(run/"execution.json")
    saved = previous["result_artifacts"]
    if previous["status"] != "complete" or full["status"] != "complete" or set(saved) != cohort_ids:
        raise ValueError(f"Cohort and full execution inventories disagree for {run.name}")
    changed = [tid for tid, digest in saved.items() if full["result_artifacts"].get(tid) != digest]
    if changed or cohort_ids.intersection(full["new_or_updated_target_ids"]):
        raise ValueError(f"Full continuation changed completed cohort results for {run.name}")
    earlier_scores = execution_scores(snapshot, previous)
    full_scores = execution_scores(run, full)
    if earlier_scores.keys() != cohort_ids or not cohort_ids.issubset(full_scores):
        raise ValueError(f"Frozen targets differ between score tables for {run.name}")
    for tid, original in earlier_scores.items():
        resumed = full_scores[tid]
        for column, value in original.items():
            if column not in resumed or resumed[column] != value:
                raise ValueError(f"Full continuation changed cohort score row for {tid} / {column}")
    return dict(preserved_target_records=len(saved), preserved_score_rows=len(earlier_scores),
                preserved_timeout_records=verify_timeout_snapshots(run, full),
                snapshot_execution_sha256=sha256(snapshot/"execution.json"),
                full_execution_sha256=sha256(run/"execution.json"))


def verify_reference_transfers(source, destination, rows, receipts):
    """Replay the copy receipts against both completed runs' native references."""
    if not receipts:
        raise ValueError("Missing reference transfer receipt")
    source_files = file_inventory(source/"visible_references")
    required = {"reference_config.json"}
    for row in rows:
        required.update(p.as_posix() for p in reference_paths(Path("."), row).values())
    if not required.issubset(source_files):
        raise ValueError("The source reference inventory is missing files for the selected targets")
    verified = {}
    for path in receipts:
        receipt = read(path)
        if (receipt["verified"] is not True or receipt["source_targets"] != len(rows)
                or (ROOT/receipt["source"]).resolve() != source.resolve()
                or (ROOT/receipt["destination"]).resolve() != destination.resolve()
                or type(receipt["new_files"]) is not int or not 0 <= receipt["new_files"] <= len(source_files)
                or receipt["files"] != source_files):
            raise ValueError(f"Reference transfer receipt differs from the source run: {path.name}")
        verified[path.name] = {"sha256": sha256(path), "new_files": receipt["new_files"]}
    for relative, expected in source_files.items():
        copied = destination/"visible_references"/relative
        if not copied.is_file() or sha256(copied) != expected:
            raise ValueError(f"Copied reference differs in {destination.name}: {relative}")
    return dict(targets=len(rows), files=len(source_files), receipts=verified)


def historical_cohort_lpips(pairs, arm, study):
    run = study.parent/"surface-annotated-20261112-freshrefs/surface-cohort"
    execution = read(run/"execution.json")
    score_path = run/"per_target.csv"
    controls = execution_scores(run, execution)
    references = {r["id"]:r for r in read(study/"reference-drift-cohort.json")["records"]}
    ids = [r["id"] for r in pairs]
    if len(ids) != 204 or ids != execution["target_ids"] or set(ids) != controls.keys() or set(ids) != references.keys():
        raise ValueError("Historical Surface comparison requires the frozen 204 targets")
    all_targets, identical = [], []
    for r in pairs:
        control, ref = controls[r["id"]], references[r["id"]]
        if (int(control["seed"]) != r["seed"] or control["source"] != r["source"]
                or ref["input_sha256"] != r["input_sha256"] or ref["current_artifacts"] != r["method_reference"]):
            raise ValueError(f'Historical cohort inputs disagree for {r["id"]}')
        contrast = dict(source=r["source"], image_identity=r["image_identity"],
                        baseline={"LPIPS":float(control["LPIPS"])}, surface=r[arm])
        all_targets.append(contrast)
        if ref["historical_artifacts"] == ref["current_artifacts"]:
            identical.append(contrast)
    return dict(all=estimate(all_targets,"LPIPS"), identical_references=estimate(identical,"LPIPS"),
                surface_execution_sha256=sha256(run/"execution.json"), surface_scores_sha256=sha256(score_path))


def natural_geometry(folder, rec, gt, *, exclude=None):
    """Compare native alpha with an original-frame outline, optionally omitting annotated pixels."""
    if exclude is not None:
        if exclude.shape != gt.shape:
            raise ValueError("Excluded pixels must share the COCOA annotation frame")
        gt = gt & ~exclude
        if not gt.any():
            return None
    if rec["status"]!="success":
        return 0.
    with Image.open(folder/"mask.png") as im:
        mask = np.array(im)>0
    completion = folder/"completion.png"
    if sha256(completion) != rec["completion_sha256"]:
        raise ValueError(f"Completion differs from its saved result for {rec['id']}")
    with Image.open(completion) as im:
        if im.mode != "RGBA" or not np.array_equal(mask, np.asarray(im.getchannel("A")) > 0):
            raise ValueError(f"Scored silhouette differs from completion alpha for {rec['id']}")
    # COCOA annotates the original image. Score only within its annotated extent.
    h,w = gt.shape
    r,c = rec["geometry"]["offset_rc"]
    if list(rec["geometry"]["original_size"]) != [w,h]:
        raise ValueError("Natural reference orientation differs")
    y,x = max(0,r),max(0,c)
    y2,x2 = min(h,r+mask.shape[0]),min(w,c+mask.shape[1])
    prediction = np.zeros_like(gt)
    if y2>y and x2>x:
        prediction[y:y2,x:x2] = mask[y-r:y2-r,x-c:x2-c]
    if exclude is not None:
        prediction &= ~exclude
    return (prediction & gt).sum()/max(1,(prediction|gt).sum())


def original_observation(run, record, crop_rc, size):
    """Map the original RGB and un-eroded SAM mask into this pass's six-image canvas."""
    with Image.open(ROOT/"datasets/open_world_amodal"/record["image"]) as source:
        rgb = source.convert("RGB")
    with Image.open(run/"visible_references/masks"/record["reference"]) as source:
        visible = source.convert("L")
    if rgb.size != visible.size:
        raise ValueError("Reference and source dimensions differ")
    w, h = rgb.size
    top, left = 3*h-h//2, 3*w-w//2
    r0,r1,c0,c1 = crop_rc
    box = (c0-left, r0-top, c1-left, r1-top)
    return project_original_observation(np.array(rgb.crop(box)), np.array(visible.crop(box))>0, size)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("arm",choices=["shape","combined"])
    p.add_argument("stage",choices=["cohort","full"])
    p.add_argument("--pmc-only", action="store_true",
                   help="Audit against the matched PMC snapshot as soon as this arm finishes")
    p.add_argument("--historical-controls", action="store_true",
                   help="Read both registered Quadro controls and record cross-device reference differences")
    args=p.parse_args()
    if args.historical_controls and (args.stage != "full" or args.pmc_only):
        p.error("Historical controls apply to the full comparison with both published arms")
    out=ROOT/"output/shape-20261121"
    ref=out/"progressive-l40s"
    design=ROOT/"output/surface-annotated-20261112-design"
    suffix="-cohort" if args.stage=="cohort" else ""
    comparators = ("baseline",) if args.pmc_only else ("baseline", "surface")
    controls_dir = ROOT/"output/surface-annotated-20261112-freshrefs" if args.historical_controls else ref
    runs={m:controls_dir/(m+suffix) for m in comparators}
    runs[args.arm]=ref/(args.arm+suffix)
    reports={m:verify_run(d,ROOT/"datasets/open_world_amodal",require_full=not bool(suffix),require_provenance=True)
             for m,d in runs.items()}
    for k in ("targets_sha256","inputs_sha256"):
        assert len({v["data_identity"][k] for v in reports.values()})==1
    ref_ids = {m:r["data_identity"]["references_sha256"] for m,r in reports.items()}
    assert len({ref_ids[m] for m in comparators}) == 1
    if not args.historical_controls:
        assert ref_ids[args.arm] == ref_ids["baseline"]
    configs={m:read(d/"run_config.json") for m,d in runs.items()}
    contract=configs[args.arm].pop("shape_policy")
    surface_contract = contract["surface"]
    if "surface" in comparators:
        control_policy = configs["surface"].pop("surface_policy")
        if args.arm == "combined":
            ignored = {"preloaded_library_sha256"} if args.historical_controls else set()
            assert {k:v for k,v in surface_contract.items() if k not in ignored} == {
                k:v for k,v in control_policy.items() if k not in ignored}
    if args.arm == "combined":
        freeze = design / "method-freeze.json"
        assert surface_contract["freeze_sha256"] == sha256(freeze)
        assert surface_contract["frozen_method"] == read(freeze)
    else:
        assert surface_contract is None
    assert contract["method"] == args.arm and all(cfg == configs["baseline"] for cfg in configs.values())
    assert contract["seeds"] == list(range(11,89,11))
    assert contract["checkpoint_sha256"]==sha256(out/"prior.pt")
    assert contract["training_sha256"]==sha256(out/"training.json")
    metric_config = read(runs["baseline"]/"metric_config.json")
    assert all(read(d/"metric_config.json") == metric_config for d in runs.values())
    if args.historical_controls and metric_config["alignment"] != "released":
        raise ValueError("Scoring-crop comparison requires released metric alignment")
    cohort=set(read(design/"cohort.json")["target_ids"])
    reuse = None
    registration = None
    if args.stage == "full":
        registration = verify_full_registration(runs[args.arm], ref/f"{args.arm}-registry.csv", reports[args.arm])
        reuse = verify_cohort_reuse(runs[args.arm], ref/(args.arm+"-cohort"), cohort)
    unseen=set(read(design/"unseen-inventory.json")["target_ids"])
    rows = read_official(ROOT/"datasets/open_world_amodal")[0]
    if suffix:
        rows = select_manifest_rows(rows, design/"cohort.json")
    expected_ids = [row["id"] for row in rows]
    packages = []
    transfers = None
    selector_snapshot = None
    vg_replay = None
    for folder in runs.values():
        execution = read(folder/"execution.json")
        assert execution["target_ids"] == expected_ids
        if args.stage == "full" and args.arm == "shape" and folder == runs["shape"]:
            selector_snapshot = verify_selector_snapshot(execution, SHAPE_SELECTOR_SNAPSHOT)
            ssim_path = ref / "shape-vg-ssim.json"
            coverage_path = ref / "shape-vg-inference-coverage.json"
            replay_hash = sha256(ssim_path)
            if replay_hash != "20f6d153315a672e337807f595c3418898dec9b11b0264a499624804e78c1af4":
                raise ValueError("Visual Genome SSIM receipt differs from CPU job 45327")
            replay = read(ssim_path)
            if sha256(coverage_path) != replay["coverage_sha256"]:
                raise ValueError("Visual Genome coverage receipt differs from the SSIM audit")
            vg_replay = verify_vg_scores(folder, execution, read(coverage_path)["rows"], replay["rows"])
            vg_replay.update(ssim_sha256=replay_hash, coverage_sha256=replay["coverage_sha256"])
        environment = read(folder/"provenance"/execution["id"]/"environment.json")
        packages.append(environment["packages"])
        if args.stage == "full" and args.arm == "combined" and folder == runs["combined"]:
            final_job = environment["execution_environment"]["SLURM_JOB_ID"]
            jobs = [*reuse["preserved_timeout_records"], final_job]
            receipts = [ref/f"combined-refs-full-{job}.json" for job in jobs]
            transfers = verify_reference_transfers(ref/"shape", folder, rows, receipts)
    assert all(environment == packages[0] for environment in packages)
    geometry=out/"evaluation_only/cocoa"
    truth={r["id"]:r for r in read(geometry/"manifest.json")["rows"]}
    pairs=[]
    pools=fallbacks=0
    for row in rows:
        tid=row["id"]
        target={m:d/"targets"/tid for m,d in runs.items()}
        records={m:read(d/"result.json") for m,d in target.items()}
        for k in ("input_sha256","seed"):
            assert all(record[k] == records["baseline"][k] for record in records.values())
        refs = {m:r["evaluation_reference_artifacts"] for m,r in records.items()}
        assert all(refs[m] == refs["baseline"] for m in comparators)
        equal_refs = refs[args.arm] == refs["baseline"]
        assert args.historical_controls or equal_refs
        values={m:read(d/"metrics.json")["metrics"] if records[m]["status"]=="success" else failure_metrics()
                for m,d in target.items()}
        paths = sorted((target[args.arm]/"shape").glob("iteration-*/decision.json"))
        traces = records[args.arm].get("diagnostics",{}).get("shape_iterations",[])
        if len(paths) != len(traces) or (records[args.arm]["status"]=="success" and not traces):
            raise ValueError("Shape decision inventory disagrees with target trace for "+tid)
        for path, trace in zip(paths, traces):
            decision=read(path)
            assert trace["decision_sha256"]==sha256(path)
            assert (trace["index"],trace["fallback"])==(decision["index"],decision["fallback"])
            seed = records[args.arm]["seed"]
            assert seed==decision["target_seed"]
            assert decision["seeds"]==[progressive_seed(seed,decision["iteration"],slot) for slot in contract["seeds"]]
            assert all(sha256(path.parent/n)==digest for n,digest in decision["artifacts"].items())
            prior=np.array(Image.open(path.parent/"prior.png"))>0
            masks=[np.array(Image.open(path.parent/f"mask-{i}.png"))>0 for i in range(8)]
            scores=score_masks(prior,masks)
            assert scores==decision["scores"]
            idx,fb=choose(scores,decision.get("surface_losses"))
            assert (idx,fb)==(decision["index"],decision["fallback"])
            if args.arm=="shape":
                assert decision["artifacts"]["selected.png"]==decision["artifacts"][f"candidate-{idx}.png"]
            else:
                frozen = surface_contract["frozen_method"]
                surface_idx, losses = predict_rank(decision["observable_scores"], frozen["ranker"])
                assert surface_idx==decision["surface_index"]
                np.testing.assert_allclose(np.array(losses,float), np.array(decision["surface_losses"],float),
                                           atol=1e-10,rtol=0,equal_nan=True)
                if suffix:
                    with Image.open(path.parent/"selected.png") as selected:
                        observed, visible = original_observation(runs[args.arm], records[args.arm], decision["crop_rc"], selected.size)
                        expected = np.array(selected.convert("RGB"))
                    np.testing.assert_array_equal(visible, np.array(Image.open(path.parent/"visible.png"))>0)
                    adjusted = []
                    for i, mask in enumerate(masks):
                        with Image.open(path.parent/f"candidate-{i}.png") as candidate:
                            adjusted.append(refine_surface(observed,visible,np.array(candidate.convert("RGB")),mask,
                                                    RefinementConfig(**frozen["refinement"]))[0])
                    pool, _ = refine_surface_pool(adjusted,masks,visible,EnsembleConfig(**frozen["ensemble"]))
                    np.testing.assert_array_equal(expected, pool[idx])
            pools+=1
            fallbacks+=int(fb)
        gt=truth.get(tid)
        if gt:
            path=geometry/gt["case"]
            assert sha256(path)==gt["sha256"]
            with Image.open(path) as image:
                annotation = np.array(image)
            amodal = (annotation & 1) != 0
            visible = (annotation & 2) != 0
            for m in runs:
                values[m]["amodal_iou"]=natural_geometry(target[m],records[m],amodal)
                if args.historical_controls:
                    values[m]["hidden_iou"] = natural_geometry(target[m], records[m], amodal, exclude=visible)
        pairs.append(dict(id=tid,source=row["source"],image_identity=row["source"]+"/"+row["original_id"],
                          **values, status={m:rec["status"] for m,rec in records.items()},
                          seed=records[args.arm]["seed"], input_sha256=records[args.arm]["input_sha256"],
                          reference_equal=equal_refs, method_reference=refs[args.arm], gt_bin=gt["bin"] if gt else None))
        if args.historical_controls:
            pairs[-1]["historical_scoring_crop_sha256"] = refs["baseline"]["crop"]
    assert len(pairs)==(204 if suffix else 2565)
    first_pass = None
    if args.arm == "combined":
        for row in pairs:
            path = Path("targets")/row["id"]/"shape"
            row["first_pass_comparison"] = compare_first_pass(ref/("shape"+suffix)/path, runs[args.arm]/path)
        first_pass = dict(Counter(row["first_pass_comparison"] for row in pairs))
    independent = out/"cocoa-historical-controls.json"
    if args.historical_controls and independent.is_file():
        replay = read(independent)
        if replay["manifest_sha256"] != sha256(geometry/"manifest.json"):
            raise ValueError("Historical polygon audit used a different COCOA manifest")
        expected = {r["id"]:r for r in replay["targets"]}
        joined = {r["id"]:r for r in pairs if r["gt_bin"] is not None}
        if joined.keys() != expected.keys():
            raise ValueError("COCOA targets differ from the historical polygon audit")
        for tid, row in joined.items():
            for arm in comparators:
                if row[arm]["amodal_iou"] != expected[tid][arm]["amodal_iou"]:
                    raise ValueError(f"Historical alpha projection disagrees for {arm} / {tid}")
        hidden_path = out / "full-cocoa-hidden-historical.json"
        if hidden_path.is_file():
            hidden = read(hidden_path)
            if (hidden["control_replay_sha256"] != sha256(independent)
                    or hidden["geometry_sha256"] != replay["manifest_sha256"]
                    or hidden["executions"] != {arm: sha256(runs[arm]/"execution.json") for arm in comparators}
                    or [r["id"] for r in hidden["rows"]] != list(joined)):
                raise ValueError("Historical hidden-region replay used different runs or annotations")
            for row in hidden["rows"]:
                for arm in comparators:
                    if joined[row["id"]][arm]["hidden_iou"] != row[arm]["hidden_iou"]:
                        raise ValueError(f"Historical hidden-region IoU disagrees for {arm} / {row['id']}")
    summaries={}
    for comparator in comparators:
        if args.historical_controls:
            same_crop = [dict(source=r["source"], image_identity=r["image_identity"],
                              baseline=r[comparator], surface=r[args.arm])
                         for r in pairs if r["method_reference"]["crop"] == r["historical_scoring_crop_sha256"]]
            summaries[comparator+"-same-scoring-crop"] = {"LPIPS": estimate(same_crop, "LPIPS")}
        groups = [("all",pairs),("cohort",[r for r in pairs if r["id"] in cohort]),
                  ("unseen",[r for r in pairs if r["id"] in unseen])]
        if args.historical_controls:
            groups.append(("same-reference",[r for r in pairs if r["reference_equal"]]))
        for name, group in groups:
            if not group:
                continue
            table=[dict(source=r["source"],image_identity=r["image_identity"],baseline=r[comparator],surface=r[args.arm])
                   for r in group]
            stats={metric:estimate(table,metric) for metric in METRICS}
            joined=[r for r in table if "amodal_iou" in r["baseline"]]
            if joined:
                stats["amodal_iou"]=estimate(joined,"amodal_iou")
                if args.historical_controls:
                    stats["hidden_iou"] = estimate(joined, "hidden_iou")
            summaries[comparator+"-"+name]=stats
        for bin_name in sorted({r["gt_bin"] for r in pairs if r["gt_bin"]}):
            group=[dict(source=r["source"],image_identity=r["image_identity"],baseline=r[comparator],surface=r[args.arm])
                   for r in pairs if r["gt_bin"]==bin_name]
            summaries[comparator+"-occlusion-"+bin_name]=estimate(group,"amodal_iou")
    prefix = f"{args.arm}-{args.stage}" + ("-vs-pmc" if args.pmc_only else "")
    write_json(out/f"{prefix}-pairs.json",pairs)
    write_json(out/f"{prefix}-summary.json",dict(targets=len(pairs),metrics=summaries,pools=pools,
               first_pass_comparison=first_pass,
               cohort_reuse=reuse, reference_transfers=transfers, registration=registration,
               early_selector_snapshot=selector_snapshot, vg_ssim_replay=vg_replay,
               historical_surface_cohort=historical_cohort_lpips(pairs,args.arm,out) if suffix and args.pmc_only else None,
               controls=comparators, historical_controls=args.historical_controls,
               identical_reference_targets=sum(r["reference_equal"] for r in pairs),
               reference_identities=ref_ids, data_identity=reports[args.arm]["data_identity"],
               pairs_sha256=sha256(out/f"{prefix}-pairs.json"), audit_sha256=sha256(Path(__file__)),
               fallbacks=fallbacks,coverage={m:dict(Counter(r["status"][m] for r in pairs)) for m in runs}))


if __name__=="__main__":
    main()

