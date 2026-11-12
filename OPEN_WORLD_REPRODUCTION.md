# PD-MC evaluation on the Open-World Amodal Appearance Completion benchmark

For paper components mapped to Python files and functions, execution flow, breakpoints, and failure diagnosis, see [the paper-to-code debugging guide](docs/PAPER_CODE_MAP.md).

## Released PD-MC with Grounding DINO + SAM only

Use `run_pdmc_open_world.sh` to evaluate PD-MC without downloading, loading, or running LISA:

```bash
bash run_pdmc_open_world.sh --local \
  --benchmark-root ./datasets/open_world_amodal \
  --cache-dir /scratch/$USER/amodal-cache \
  --output-dir ./output/pdmc-open-world-sam-semantic-v2 \
  --benchmark-limit 2
```

Remove `--benchmark-limit 2` for all 2,565 annotated targets. The same output directory can grow from a smoke run into the full evaluation and can resume interrupted runs. Omit `--local` on a login node to submit through Slurm; add the usual `--partition`, `--gres`, `--cpus`, `--mem`, and `--time` options. `--benchmark-workers auto` uses one independent PD-MC worker per allocated, CUDA-visible GPU. The shared cache and pinned PD-MC environment/checkpoints can be reused from previous runs.

This entry point runs the pinned [released PD-MC implementation](https://github.com/k8xu/amodal/blob/bfbfcdec581484586774d9ca53f5c603c97c725b/main.py): Grounding DINO + SAM instance segmentation, InstaOrder occluder ordering, LaMa background removal, and progressive Mixed Context diffusion. Defaults retain the fixed released vocabulary, MC timestep **35**, inclusive `max_iter_id=5`, and the **released boundary check**, including its known coordinate-order bug. The dedicated Bash and Python entry points default to semantic label matching and automatic background sizing, which enables the validated backdrop repair described below. Model-source compatibility changes, deterministic per-image/per-target seeds, output bookkeeping, and dataset target matching remain local evaluation infrastructure; this is not a claim of bit-exact paper results.

Annotations select a **unique semantic label match** from the released scene segmentation. Explicit aliases resolve `television` to `tv`, `brown table` to `dining table`, and `white sofa` to `couch`. The matcher handles plural forms and object labels before spatial descriptions, and respects compound nouns such as `hot dog` and `table lamp`. Exactly equal binary masks are deduplicated, retaining the highest detector score (stable first index on ties) and its original detected-class diffusion prompt. Distinct masks, including overlapping same-class instances, remain ambiguous. Missing detections and unresolved targets count as pipeline failures; every candidate's class, score, mask area and bounding box is recorded in `selection_diagnostics`. This matching rule differs from the reference-IoU matching used by the LISA benchmark workflow.

The corrected matcher is `--benchmark-target-selection label-semantic` (Python: `--target-selection label-semantic`). Its version, alias digest and implementation hash are recorded in both inference and reference configurations. Start a **new output and reference directory**; do not import results from the previous label matcher. The dedicated launcher's default output is `output/pdmc-open-world-sam-semantic-v2`. The shared `run_amodal_slurm.sh --benchmark-pipeline-profile released-sam` and `evaluate_open_world.py --pipeline-profile released-sam` keep legacy defaults (`label`, `released` background sizing). To resume an old SAM experiment through a dedicated launcher, explicitly select `--benchmark-target-selection label` and its original `--benchmark-background-sizing released|auto` (Python: omit the `benchmark-` prefix). Score-only restores the saved matching and sizing settings and verifies the reference configuration against the inference run.

Matcher v2 fixes the remaining `teddy bear toy` rejection and adds explicit `toy bear`, `motor bike`, `surfing board`, and `snow board` variants, including plurals. Toy bears map to `teddy bear`, rather than a live `bear`; unrelated labels such as `dog toy` are not shortened. These corrections affect target identity only. They do not guarantee completion when the detector misses the object or PD-MC rejects it. Completed v1 semantic runs can still be scored with `--benchmark-score-only` and their explicit original output directory. Continuing v1 inference requires the original v1 matcher source; the v2 matcher rejects that configuration.

For this v1-to-v2 update, upload only the two changed runtime files from local PowerShell:

```powershell
scp .\amodal_benchmark\matching.py daoqm@slurm.csse.rose-hulman.edu:Vu_Hung/cv_nangcao/amodal_benchmark/
scp .\run_pdmc_open_world.sh daoqm@slurm.csse.rose-hulman.edu:Vu_Hung/cv_nangcao/
```

Then submit a fresh four-worker run on the server:

```bash
cd ~/Vu_Hung/cv_nangcao
sbatch --partition=debug-gpu --gres=gpu:4 \
  --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --output=slurm-%x-%j.out \
  run_pdmc_open_world.sh --run-only \
  --cache-dir /dev/shm/daoqm/amodal-cache \
  --benchmark-workers 4 \
  --output-dir ./output/pdmc-open-world-sam-semantic-v2
```

For LPIPS, VGG feature similarity, and SSIM, the evaluator saves each selected **initial SAM mask** as a visible reference, applies the existing 5×5 erosion and first-crop preprocessing, and compares it with the completion. CLIP measures successful completions against their annotated labels. All four metrics, per-source/overall failure rates, failure reasons, and denominators appear in `summary.json`, `per_target.csv`, and `table.md`. Failed detections and completions receive LPIPS=1, VGG=0, and SSIM=0; they have no CLIP score. Successful completions with empty eroded reference masks receive CLIP only, with the smaller appearance denominators reported explicitly.

SAM references are stored in `OUTPUT/visible_references`, independently of the existing LISA references. They are generated from the same scene detections used for completion, so they measure preservation of PD-MC's detected visible content rather than independent segmentation accuracy or hidden-pixel ground truth. These appearance scores define a **separate protocol from the paper's LISA-reference evaluation**; `summary.json` identifies `reference_backend: grounded-sam`, and paper-comparison appearance entries are marked `different_reference_backend`. Use a new output directory when switching profiles or changing inference settings. Do not point `--reference-dir` at a LISA reference directory.

`successes` retains the released convention: an output was saved, even if progressive completion hit the iteration cap. Reports now split saved outputs into `completed_outputs`, `iteration_limited_outputs`, and `completion_state_unknown` for historical records without a stop flag. `failure_stage_counts` distinguishes initial segmentation, target selection, and PD-MC gates; the CSV includes `failure_stage` and `iteration_limit_reached`. Historical CSVs remain readable, and the string `False` is parsed as false. Failure rates and metric denominators retain their original definitions.

The launcher also accepts `--benchmark-inference-only` and `--benchmark-score-only`. Inference-only runs still save SAM references so they can be scored later. For direct use inside the provisioned environment with the shared runner's import paths configured:

```bash
python3 evaluate_pdmc_open_world.py \
  --dataset-root ./datasets/open_world_amodal \
  --output-dir ./output/pdmc-open-world-sam-semantic-v2 --limit 2 --check-only

python3 evaluate_pdmc_open_world.py \
  --dataset-root ./datasets/open_world_amodal \
  --repo-dir /scratch/$USER/amodal-cache/amodal \
  --output-dir ./output/pdmc-open-world-sam-semantic-v2 --workers auto

python3 evaluate_pdmc_open_world.py \
  --dataset-root ./datasets/open_world_amodal \
  --output-dir ./output/pdmc-open-world-sam-semantic-v2 --score-only
```

The check-only command validates source images on CPU without requiring models or previously generated references. The Python score-only command validates saved outputs and SAM artifacts without loading PD-MC inference models. Custom vocabularies, models, timesteps, or iteration limits are recorded as changed experiments. The profile rejects reference-based target matching, the repaired boundary mode, and LISA reference-preparation flags.

Copy `run_pdmc_open_world.sh`, `evaluate_pdmc_open_world.py`, the updated `run_amodal_slurm.sh` and `evaluate_open_world.py`, and the complete `amodal_benchmark/` directory to the cluster, preserving their relative paths. The existing LISA workflow below remains available through `run_amodal_slurm.sh` without the `released-sam` profile.

For example, transfer the wrapper repair from the project directory (replace the host and destination):

```bash
rsync -av run_pdmc_open_world.sh evaluate_pdmc_open_world.py \
  run_amodal_slurm.sh evaluate_open_world.py OPEN_WORLD_REPRODUCTION.md \
  USER@CLUSTER:/path/to/cv_nangcao/
rsync -av --exclude='__pycache__/' amodal_benchmark/ \
  USER@CLUSTER:/path/to/cv_nangcao/amodal_benchmark/

# Inside an existing GPU allocation, use a fresh directory for the smoke run.
bash run_pdmc_open_world.sh --local --run-only \
  --cache-dir /scratch/$USER/amodal-cache \
  --benchmark-limit 2 --benchmark-workers 1 \
  --output-dir ./output/pdmc-open-world-sam-semantic-v2-smoke
```

CPU tests exercise real evaluator orchestration with model substitutes. Real inference and `tests/validate_cuda_extension.sh ENV_PREFIX UPSTREAM_CHECKOUT` / `tests/validate_pipeline_imports.sh ENV_PREFIX UPSTREAM_CHECKOUT` require the provisioned cluster environment.

### Expand the detector vocabulary using image annotations

Use `--benchmark-detector-vocabulary annotated` (Python: `--detector-vocabulary annotated`) for a separate experiment that adds categories missing from the original scene vocabulary. It requires the SAM profile and `label-semantic` matching. The default remains `released`, preserving completed v2 experiments.

For each image, the evaluator keeps the original scene categories for occluder reasoning and adds the categories annotated in that image, such as `plate`, `bathtub`, or `guitar`. Existing semantic aliases produce canonical queries; other labels retain their normalized object phrase before a spatial description. Each added category gets its own detector query. Base categories are split only if the actual detector tokenizer exceeds the available token budget, capped at 256 tokens including punctuation and special tokens. Grounding DINO [truncates longer input](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/models/GroundingDINO/groundingdino.py), so the full set of 553 labels is never concatenated into one caption.

The same image queries are used during initial segmentation and every completion check. The detector's returned phrases, scores, thresholds, and diffusion prompts are preserved. Returned phrases from added queries join the query eligibility vocabulary; this permits `plate` returned for an added `white plate` query. Distinct instance masks remain ambiguous. Existing area/confidence and occlusion gates still apply, so expanded vocabulary does not guarantee successful completions.

Annotation queries use all official rows for the image, independently of smoke limits and worker partitioning. Their digest, implementation hash and policy appear in inference/reference configurations and `summary.json`; `result.json` records the query categories, returned phrases and eligibility vocabulary in `detection_diagnostics`. This uses benchmark annotations to guide detection and is an **annotation-guided vocabulary experiment**, rather than the original released-vocabulary baseline. Use a new output/reference directory; existing results cannot be imported or resumed under the changed policy. Score-only restores the saved policy.

From local PowerShell, update a server already running matcher v2:

```powershell
scp evaluate_open_world.py run_amodal_slurm.sh run_pdmc_open_world.sh daoqm@slurm.csse.rose-hulman.edu:Vu_Hung/cv_nangcao/
scp .\amodal_benchmark\detection.py daoqm@slurm.csse.rose-hulman.edu:Vu_Hung/cv_nangcao/amodal_benchmark/
```

Submit an eight-target, four-worker smoke run on the server:

```bash
cd ~/Vu_Hung/cv_nangcao
sbatch --partition=debug-gpu --gres=gpu:4 \
  --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --output=slurm-%x-%j.out \
  run_pdmc_open_world.sh --run-only \
  --cache-dir /dev/shm/daoqm/amodal-cache \
  --benchmark-detector-vocabulary annotated \
  --benchmark-workers 4 --benchmark-limit 8 \
  --output-dir ./output/pdmc-open-world-sam-annotated
```

After the smoke job finishes, remove `--benchmark-limit 8` and resubmit with the same output directory for the full evaluation. This preserves the smoke outputs and inference configuration. CUDA validation and real inference require the cluster environment; CPU tests cover query construction, token limits, serial/four-worker behavior, completion query forwarding, configuration guards, and score-only restoration.

### Recover from empty completion masks

An `IndexError: index 0 is out of bounds for axis 0 with size 0` at `analyze_masks(... masks[0] ...)` can occur when all detector queries find nothing in a generated completion. The expanded detector returns an empty `(0,H,W)` mask array; the pinned upstream occlusion check recognizes only `None` as missing segmentation.

The runtime hook in `amodal_benchmark/inference.py` converts an empty completion-mask array to that existing `None` sentinel. The target is recorded as `no_completion` with reason `completion_not_detected`, and its iteration trace records `completion_segmentation_failure: empty_mask_array`. Other targets continue. Nonempty masks and unrelated model errors retain their existing behavior.

This repairs a path that previously crashed before publishing a target result. Detector/matcher settings and identities stay unchanged, so keep the existing output/reference directory and configuration. Completed targets are reused, and interrupted targets are retried. After the failed Slurm job and its workers have exited, upload only this file from local PowerShell:

```powershell
scp .\amodal_benchmark\inference.py daoqm@slurm.csse.rose-hulman.edu:Vu_Hung/cv_nangcao/amodal_benchmark/
```

Resubmit the full annotated-vocabulary command above without `--benchmark-limit 8`, using the same `output/pdmc-open-world-sam-annotated` directory. No checkpoint/environment setup or configuration-file edits are needed. Regression tests reproduce the exact pinned `analyze_masks` crash, verify failed-completion reporting, preserve valid completion pixels, and exercise resume with one and four workers. Real GPU validation remains a cluster check.

### Recover from the fixed-backdrop indexing error

`IndexError: index 1200 is out of bounds ... size 1200` in `run_mixed_context_diffusion` means that a progressive crop exceeds the fixed wallpaper dimensions. It is independent of worker count. Automatic sizing is enabled by default in the dedicated launchers; the shared released-SAM profile requires explicit `--benchmark-background-sizing auto`. This enables aspect-preserving bicubic enlargement only when an indexed background pixel would otherwise be out of bounds. The visible query pixels and all previously supported lookups keep their original behavior. The SAM/Grounding DINO models, released boundary gate, detection vocabulary, seeds, thresholds and diffusion parameters remain the same.

Wait for the interrupted job and its workers to exit, then copy the updated `evaluate_open_world.py`, `run_amodal_slurm.sh`, `run_pdmc_open_world.sh`, and the new `amodal_benchmark/reuse.py` to the server. Submit a repaired run from the login node:

```bash
bash run_pdmc_open_world.sh \
  --partition debug-gpu --gres gpu:4 --benchmark-workers 4 \
  --cpus 16 --mem 64G \
  --cache-dir /dev/shm/daoqm/amodal-cache \
  --benchmark-target-selection label \
  --benchmark-background-sizing auto \
  --benchmark-reuse-results-from ./output/pdmc-open-world-sam \
  --output-dir ./output/pdmc-open-world-sam-bgfix
```

Use a **new output directory** because `resize_background` changes from false to true in `run_config.json`. Do not edit or delete the original configuration to bypass the guard. The reuse option accepts only a released-SAM source whose inference configuration differs solely by enabling this repair. The example explicitly retains legacy `label` matching; corrected semantic matching cannot reuse those records. It locks the source against an active coordinator and verifies model/vocabulary configuration, input hashes, completion hashes, and all reference artifacts before copying the selected completed records. The source must keep its default `SOURCE_OUTPUT/visible_references` directory. Other inference settings must match the source; repeat any custom `--benchmark-seed`, `--mc-timestep`, `--max-iter`, `--sd-model`, or `--classes-file` options.

Completed source targets are valid to reuse because the sizing repair leaves every previously supported lookup unchanged. The interrupted target has no completed record and is retried. Originals are preserved; `reused_results.json` records their source configuration and confirmed IDs, with a pending-ID journal for interruption recovery. Reference records and result records are published atomically. Appearance metrics are recomputed in the new directory. If copying is interrupted, resume with the reuse option to recover its pending provenance. After copying finishes, subsequent resumes may omit `--benchmark-reuse-results-from`, but must retain `--benchmark-target-selection label`, `--benchmark-background-sizing auto` and the new output path. Direct Python equivalents are `--target-selection label --background-sizing auto --reuse-results-from OLD_OUTPUT`; Python score-only reads the saved settings automatically.

CPU regressions reproduce the exact 1200-pixel error using the pinned upstream function, exercise repaired serial/parallel evaluation and scoring, and verify import preservation and refusal of changed configurations or corrupt completions. GPU confirmation is still required on the affected cluster allocation.

`supplement_paper.pdf` is **Open-World Amodal Appearance Completion**, arXiv:2411.13019v1, including its supplementary material. `CS331_paper.pdf` is **Amodal Completion via Progressive Mixed Context Diffusion** (PD-MC, CVPR 2024). This integration runs the latter method on the former paper's real-occlusion benchmark, targeting the **PD-MC row of Table 3**. Visible-object references are generated from the images using the released LISA model and preprocessing; a separate reference dataset is not required.

## Paper comparison and controlled reproduction runs

The saved corrected report in `output/pdmc-paper-v2` contains 1,413 successful completions, 1,151 model failures, and one reference failure. Its overall model failure rate is 44.89%, compared with the paper's 45.5%; the source-level rates still differ. Inspect source-level deltas before changing inference settings.

Compare existing reports without loading models or modifying the source runs:

```bash
python3 compare_open_world.py \
  --run-dir ./output/pdmc-full --run-dir ./output/pdmc-paper-v2 \
  --output-dir ./output/reproduction-audit
```

The CLI validates every target row against `summary.json` and writes `comparison.json` and `comparison.md`. The report includes Tables 3/5 reference values, signed deltas, source coverage, reference exclusions, per-metric denominators, and failure reasons. CLIP is named **CLIP_success** because its denominator contains successful completions only; its comparability with the paper remains unresolved. Failure-rate deltas use percentage points. Full source coverage does not imply that every target has a valid reference or score.

New evaluator runs also write `paper_comparison.json` and `paper_comparison.md` automatically. Their `summary.json` includes hashes identifying the selected targets, source-image hashes, and generated reference artifacts. The comparison CLI only pools different inference seeds when the complete inference/metric configurations and these identities match. It reports per-seed means and sample standard deviations (`ddof=1`), with all instance denominators retained. Older reports without verified input/reference identities remain separate; one run does not estimate a standard deviation. Re-scoring an old run can populate these identities when the corresponding original images, generated references, and saved results are available and pass the evaluator's existing checks.

Plan the two controls on the **cluster**, from the project directory, using an already prepared reference directory:

```bash
python3 run_open_world_controls.py \
  --dataset-root ./datasets/open_world_amodal \
  --reference-dir ./datasets/open_world_amodal/visible_references \
  --cache-dir /scratch/$USER/amodal-cache \
  --output-root ./output/pdmc-controls \
  --partition debug-gpu
```

This writes `output/pdmc-controls-plan/plan.json` and `commands.sh`; it does not submit jobs. Repeat the command with **`--execute`**, or run the generated `commands.sh`, to run CPU data/reference preflight and invoke the existing Slurm runner. On a login node, each command submits its own job, chained with `afterok` dependencies so jobs sharing the repository, environment and model cache run sequentially. Job IDs and actual submission commands are saved in the plan. A failed submission stops the launcher; a failed compute job prevents its dependent jobs from starting. Within an allocation, the runner executes sequentially; `--local` explicitly selects that behavior. Defaults request four GPUs, 16 CPUs, 64 GiB RAM and 24 hours per job; override `--gres`, `--workers`, `--cpus`, `--mem`, `--time`, and `--partition` for the cluster. No jobs are submitted by the report CLI or the default planning command. Regenerate plans on the cluster: generated scripts contain the paths and working directory used when planning. Keep independent launches sharing the same cache from overlapping.

| Experiment | Boundary check | Target matching | Output for seed 0 |
| --- | --- | --- | --- |
| `boundary-released` | `released` | `reference-best` | `pdmc-controls/boundary-released/seed-0` |
| `matching-strict` | `paper` | `reference` | `pdmc-controls/matching-strict/seed-0` |
| `corrected` (optional) | `paper` | `reference-best` | `pdmc-controls/corrected/seed-0` |

The default plan contains the first two experiments at seed 0. Add `--experiments corrected boundary-released matching-strict` to also rerun the corrected configuration with the same current code/environment. Add `--seeds 0 1 2` for three inference trials per experiment: this creates nine jobs when all three experiments are selected. All commands explicitly reuse the same `--reference-dir`; they never request LISA regeneration. Keep that directory unchanged while jobs are pending or running. Reference changes are recorded in result identities and prevent pooling across seeds. Use the shared runner's reference-only workflow first if references are missing.

The controls retain timestep **35**, inclusive upstream `max_iter_id=5`, fixed released vocabulary and `released` scoring alignment. Matching remains a local adapter assumption. These controls diagnose boundary and matching effects; choosing a configuration solely because its table values are closest does not establish the authors' original protocol. A timestep-20 run belongs in a separately labelled ablation.

Once runs finish, pass their directories to `compare_open_world.py`, repeating `--run-dir` for each experiment/seed. Use a separate report output directory. Existing `pdmc-full` and `pdmc-paper-v2` reports remain available for historical comparison.

## Corrected reproduction run

The first full run produced 1,225/2,565 completions (52.24% non-success, versus Table 5's 45.5%). Two reproducible problems have been addressed:

- **Boundary detection:** pinned PD-MC `check_touch_boundary` unpacks minimum/maximum coordinates in reverse. A large object touching an edge is consequently missed, preventing outpainting or stopping progressive completion early. Default `--benchmark-boundary-check paper` fixes the coordinate order, implementing CS331 Appendix A.1's 10-pixel rule. Buffer/padding sizes, confidence/area thresholds, vocabulary, diffusion and occlusion models remain as released. The open-world authors' own boundary function already has the correct order.
- **Duplicate target detections:** the previous adapter rejects even identical masks when their IoUs differ by less than 0.05. Default `auto` now resolves to `reference-best`: select the highest-IoU detected mask, keeping the 0.5 identity threshold and resolving exact ties by first detection. This uses the released PD-MC mask-matching pattern, but the authors' benchmark adapter remains unavailable; this rule is an explicit reproduction assumption.

Both annotated label and actual detected class remain visible in `per_target.csv`. A shirt detected as `bed` is a detector error; replacing its prompt with the annotation or expanding the fixed vocabulary would change the baseline. New result records include selection candidates/IoUs and actual stop reasons: `detector_score_below_threshold`, `query_mask_too_small`, `unknown_detected_class`, `no_occlusion_detected`, or `completion_not_detected`. `summary.json` reports `status_counts`, `failure_reasons`, and `failure_rate` per source, separating missing images and reference failures.

Copy these runtime files, preserving the module directory:

```powershell
scp run_amodal_slurm.sh evaluate_open_world.py daoqm@slurm.csse.rose-hulman.edu:~/Vu_Hung/cv_nangcao/
scp amodal_benchmark/inference.py amodal_benchmark/protocol.py amodal_benchmark/reproduction.py daoqm@slurm.csse.rose-hulman.edu:~/Vu_Hung/cv_nangcao/amodal_benchmark/
```

Keep `output/pdmc-full` as the baseline. Corrected semantics are recorded as adapter version 4 and require a **new output directory**; existing datasets, models, and visible references can be reused. Submit a batch job from the project directory:

```bash
cd ~/Vu_Hung/cv_nangcao
mkdir -p logs
AMODAL_SCRIPT_DIR="$PWD" sbatch \
  --job-name=pdmc-paper-v2 --partition=debug-gpu --nodelist=gebru \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 \
  --mem=64G --time=24:00:00 --chdir="$PWD" \
  --output="$PWD/logs/pdmc-paper-v2-%j.log" \
  run_amodal_slurm.sh --cpus 16 \
  --benchmark-root ./datasets/open_world_amodal --benchmark-workers 4 \
  --reference-dir ./datasets/open_world_amodal/visible_references \
  --benchmark-boundary-check paper --benchmark-target-selection reference-best \
  --cache-dir /dev/shm/daoqm/amodal-cache --output-dir ./output/pdmc-paper-v2
```

For a short first check, add `--benchmark-sources free_images --benchmark-limit 20`. Then repeat without those two options to cover all 2,565 targets in the same corrected output directory, reusing those completed smoke targets. Repeating the full command resumes interrupted corrected runs. Batch execution continues after SSH disconnects.

To resume the **old** `pdmc-full` experiment, explicitly add `--benchmark-boundary-check released --benchmark-target-selection reference` and keep its original parameters. Score-only evaluation reads the saved boundary/selection settings automatically. Do not delete configurations to bypass the guard or copy old completions into the corrected directory.

CPU tests exercise the pinned upstream gates, boundary cases and crops, distinguish pre/post-diffusion failures, check unchanged interior-case output pixels, and verify serial/parallel resume and legacy scoring. A new full GPU run is needed to measure whether these corrections narrow the published gap. Regenerated references, reconstructed LAION bytes, and unspecified author adapter/seed/CLIP-failure details still prevent a promise of exact table agreement.

## Verified dataset and installation status

The correct benchmark is the authors' selected evaluation subset, not the complete COCO, COCO-A, Visual Genome, or LAION datasets. The installed annotation files match every count in Table 1:

| Source | Original images | Target instances | Classes | Installed original images |
| --- | ---: | ---: | ---: | ---: |
| VG | 1,234 | 1,294 | 269 | 1,234 |
| COCO-A (`COCOA`) | 751 | 845 | 256 | 751 |
| Free images (`free_images`) | 228 | 249 | 147 | 228 |
| LAION (`laion`) | 166 | 177 | 97 | 166 |
| Total | 2,379 | 2,565 | 553 distinct overall | 2,379 |

All installed images were fully decoded. The free-image release contains 249 JPEG files: target-specific copies represent 228 original images. VG and COCO are stored once per original image. Download provenance and SHA-256 values are in `datasets/open_world_amodal/image_download_manifest.json`; the installed content hashes are in `image_checksums.json`. The verification report lists every missing file and preserves all 2,565 targets in `manifest.json`.

The ZIP supplied locally, `Open-World Amodal Appearance Completion-20261015T075717Z-1-001.zip`, is the correct official metadata release. Its ten files match the separately downloaded release byte-for-byte. Its nested `website.zip` contains only the free images. Neither archive contains VG/COCO/LAION originals or reference visible-object RGBA images. The outer and nested archive CRC checks passed.

All 166 selected LAION image identities were recovered. Their nine-digit keys match row positions in the official first LAION metadata parquet: row 115 is the violin image, row 390 contains the sofa, and additional examples were checked. The installed 256×256 images use the pinned 2021 img2dataset resize, centered white padding, and JPEG encoding procedure. Recovery comprises 146 original URLs, 18 exact-sample-ID public mirror recoveries with lossless JPEG XL-to-JPEG reconstruction, one public cache of the exact original courthouse-dog URL, and one Cambridge publisher image of the matching ISBN book cover containing the guitar target.

This is **not a byte-exact extraction of the original `00000.tar`**. Re-encoding and the publisher cover's different source resolution can affect numerical results. The cover's exact ISBN and edition were checked; no unrelated image was substituted. `laion_source_reconstruction_provenance.json` records sample IDs, URLs, raw/output hashes, sizes, and recovery methods. `laion_acquisition_verification.json` verifies all 166 originals and 177 target entries. Preserved raw inputs and `downloads/reconstruct_laion_selected.py` allow offline reconstruction.

Visible references are generated on the GPU node by `prepare_visible_references.py`. The public paper and code specify LISA-13B-llama2-v1 and provide the actual crop/erosion procedure; their absence as precomputed PNGs is not an installation blocker.

The downloaded release's `official/readme.txt` records the authors' distribution terms. It directs users to the original VG/COCO/LAION sources for those images and describes the free-image providers. Image rights remain with the original providers.

## Transfer and prepare

Copy these files to the cluster, retaining their relative locations:

```text
run_amodal_slurm.sh
prepare_open_world.py
prepare_visible_references.py
evaluate_open_world.py
compare_open_world.py
run_open_world_controls.py
amodal_benchmark/
datasets/open_world_amodal/
```

The installed dataset includes all selected source images, so those do not need to be downloaded again on the node. For a fresh installation of the VG/COCO/free sources using the supplied archive:

```bash
python3 -m pip install numpy Pillow requests
python3 prepare_open_world.py \
  --release-zip "Open-World Amodal Appearance Completion-20261015T075717Z-1-001.zip" \
  --download-images --allow-incomplete
```

Alternatively, use `--download-metadata --download-images --allow-incomplete` to retrieve the public release. Downloads select only the required VG and COCO originals. The existing `input/` sample images are not benchmark inputs.

If an original LAION archive is available, it can replace the reconstructed LAION files:

```bash
python3 prepare_open_world.py --laion-tar /scratch/data/original-laion400m/00000.tar
```

The importer extracts only the requested basenames, checks decoding, and records the supplied archive's SHA-256. Because the authors supply no shard checksum, establishing the archive's original provenance remains necessary; matching basenames alone is insufficient.

The Slurm workflow automatically generates references under `DATASET/visible_references`. Original-frame eroded RGBA files use these names:

```text
references/
  VG/1159295_bus.png
  COCOA/COCO_test2014_000000296835_toothbrush.png
  free_images/coins-1015125_640_potato.png
  laion/000000115_violin.png
```

Names follow the official annotation basenames. Alongside each original-frame PNG, `crops/SOURCE/TARGET.png` holds the released `sd_img_cut`-style reference; `masks/SOURCE/TARGET.png` holds the raw, uneroded LISA mask. `records/SOURCE/TARGET.json` stores hashes, crop offsets, seed, query, and status. Empty LISA predictions are recorded separately. After generation, verify the installation with:

```bash
python3 prepare_open_world.py --reference-dir /scratch/data/references
```

Exit code 2 indicates missing/corrupt input images. `verification.json` separately reports reference completeness; `evaluate_open_world.py --check-only` enforces references before evaluation. `--allow-incomplete` records a partial installation and returns zero deliberately.

## Slurm smoke test with automatic visible-reference generation

If you are already on the compute node, add `--local` to execute directly with its current GPU allocation. This also works when the shell does not retain `SLURM_JOB_ID`:

```bash
bash run_amodal_slurm.sh --local \
  --benchmark-root ./datasets/open_world_amodal \
  --cache-dir /scratch/$USER/amodal-cache \
  --output-dir ./output/pdmc-evaluation
```

Add `--benchmark-limit 2` for a smoke test. Existing Slurm allocations are still detected automatically. Partition, GPU, memory, and time requests apply when submitting a new job; local execution uses the resources already assigned to the node.

Start with two targets and reuse the existing environment/cache options. This generates references, runs PD-MC, and computes the four metrics:

```bash
bash run_amodal_slurm.sh \
  --benchmark-root ./datasets/open_world_amodal \
  --benchmark-limit 2 \
  --cache-dir /scratch/$USER/amodal-cache \
  --output-dir /scratch/$USER/pdmc-smoke \
  --partition YOUR_GPU_PARTITION --gres gpu:2 \
  --cpus 8 --mem 64G --time 24:00:00
```

Add `--account YOUR_ACCOUNT` if required. The same compiler/CUDA/dependency repairs and model provisioning as the qualitative runner apply. **A single 24-GiB RTX 6000 cannot hold unquantized LISA-13B. Default `--lisa-device-map auto` now balances its decoder layers across all CUDA-visible GPUs**, while retaining the vision/segmentation components on the first visible GPU and reserving inference headroom. This supports the eight-RTX-6000 node without quantizing the checkpoint. A single sufficiently large GPU also works. Use FP16 on the Turing Quadro RTX 6000; BF16 is rejected on unsupported GPUs. Keep the cache on persistent storage with room for the approximately 30GB LISA checkpoint plus other models. The stages use separate processes and release the preceding model before loading the next.

If you are already on your eight-GPU compute node, rerun with the existing cache:

```bash
bash run_amodal_slurm.sh --local \
  --benchmark-root ./datasets/open_world_amodal \
  --lisa-device-map auto --benchmark-limit 2 \
  --cache-dir /dev/shm/daoqm/amodal-cache \
  --output-dir ./output/pdmc-smoke
```

The runner prints the number of GPUs visible to PyTorch, their free-memory budgets, and planned weight placement. CUDA visibility must include multiple GPUs allocated to your job; `nvidia-smi -L` alone may list eight physical cards even if Slurm exposes only one to PyTorch. For a newly submitted job on this node, request the desired GPU count, e.g. `--gres gpu:8`. The runner respects `CUDA_VISIBLE_DEVICES` and does not expose GPUs outside the current allocation. `--lisa-device-map single` is available for a sufficiently large single card. No cache deletion or model re-download is needed for this repair. Keep the same reference directory: placement metadata is stored separately under `visible_references/executions/`, preserving the prior reference configuration and resume checks.

Remove `--benchmark-limit 2` and use a separate output directory to process all **2,565 targets** from all four sources. `--benchmark-sources` is available for explicit subset experiments.

## Parallel PD-MC inference on allocated GPUs

Benchmark mode now defaults to `--benchmark-workers auto`: one independent PD-MC worker per CUDA-visible GPU, capped to the number of selected images. This is separate from LISA's multi-GPU model placement. Workers process different images using the same checkpoints, vocabulary, thresholds, diffusion settings, and stable per-target seeds. All targets of one image stay together. The coordinator computes metrics and writes the combined `per_target.csv`, `summary.json`, and `table.md` after every inference worker succeeds.

Resume the existing full run on the compute node:

```bash
cd ~/Vu_Hung/cv_nangcao
bash run_amodal_slurm.sh --local \
  --benchmark-root ./datasets/open_world_amodal \
  --benchmark-workers auto --lisa-device-map auto \
  --cache-dir /dev/shm/daoqm/amodal-cache \
  --output-dir ./output/pdmc-paper-v2
```

An initial installation needs the complete `amodal_benchmark/` directory, including `parallel.py` and `reproduction.py`. Existing valid targets within the same protocol are reused; changing worker count does not change the semantic run configuration. Use `--benchmark-workers 4` to select up to four allocated GPUs, or `--benchmark-workers 1` for serial execution. An explicit count greater than CUDA visibility fails with an actionable error. Auto mode uses four workers when Slurm exposes four cards, even on an eight-card node; it never changes allocation boundaries.

Each worker loads a complete PD-MC pipeline, so host RAM usage grows with worker count. Request resources for these copies when submitting a new job, for example `--gres gpu:8 --cpus 32 --mem 128G`; include space for any RAM-backed model cache. The coordinator divides the CPU thread budget among workers. `--local` resource flags cannot expand an existing allocation.

Per-worker output is written to `OUTPUT/workers/<execution-id>/worker-00.log`, etc.; the runner prints each path. `execution.json` records devices, CPU threads, status, and exit codes. A failed worker stops its siblings and returns an error without creating a new combined report. Repeating the command resumes completed targets. An OS file lock rejects overlapping evaluation coordinators using the same output directory.

CPU integration tests verify serial/parallel scheduling, unchanged completion artifacts and seeds with substituted model calls, resume, and metric/failure denominators. Actual numerical parity and throughput still require a GPU-node smoke test; scheduling preserves the algorithm and does not promise bitwise equality across different GPU/CUDA implementations.

## Full evaluation and separate preparation jobs

Run a CPU source-image preflight before requesting the GPU allocation:

```bash
python3 prepare_visible_references.py \
  --dataset-root ./datasets/open_world_amodal --check-only
```

Then submit:

```bash
bash run_amodal_slurm.sh \
  --benchmark-root ./datasets/open_world_amodal \
  --cache-dir /scratch/$USER/amodal-cache \
  --output-dir /scratch/$USER/pdmc-evaluation \
  --partition YOUR_GPU_PARTITION --gres gpu:8 \
  --cpus 32 --mem 128G --time 24:00:00
```

Use your cluster's permitted wall time; the full benchmark may need multiple jobs. Repeating the same command resumes reference preparation, completed targets, and metric records. `--run-only` reuses a fully prepared runtime/checkpoint setup. A full metrics run additionally requires `lpips==0.1.4`; omit `--run-only` once to install it into the private environment. No metric model is loaded alongside diffusion models.

To prepare references in a separate job, add `--benchmark-reference-only`. Then pass `--reference-dir ./datasets/open_world_amodal/visible_references` to the evaluation job to reuse them without provisioning/loading LISA. `--benchmark-generate-references` fills missing references in an explicitly supplied directory. `--benchmark-inference-only` retains the option to run PD-MC without reference preparation/scoring.

### Recovered visible-reference procedure

The paper's Section 3.1/configuration appendix and its modified `LISA/app.py` identify **LISA-13B-llama2-v1**. The implementation retains the raw annotated label as query, `llava_v1` conversation, CLIP preprocessing, 1024-pixel SAM resize/normalization/padding, `max_new_tokens=512`, and first predicted mask thresholded at zero. In the released `main.py`, the image and raw mask are placed on a 6× canvas, cropped using a 60-pixel buffer and 150-pixel boundary padding with a 10-pixel boundary gap, then the mask is **eroded once by a 5×5 kernel** before becoming the alpha channel of `sd_img_cut.png`. Visible RGB values are preserved.

The crop calculation was checked against the actual published functions on interior/boundary examples; binary erosion was checked against OpenCV. Generated visible references are not human annotations or hidden-region ground truth. Empty/eroded-away predictions are recorded and excluded from PD-MC failure/appearance denominators rather than replaced with arbitrary masks.

LISA source commit `3cb2d4301f1af4691bd4f3938335ef06e76f155a`, model revision `b89000be11ad0a45512745a15063f2f6af1d9a5c`, and its CLIP vision tower are pinned. Inference uses unquantized FP16 PyTorch with the same model/preprocessing, without Gradio or optional DeepSpeed kernel replacement. Accelerate 0.24.1 dispatches complete decoder layers to multiple GPUs when necessary, with LISA's vision, segmentation, and projection modules kept on logical GPU 0. The device map deliberately omits root/model parent entries that would temporarily copy the whole model to GPU 0. LISA's Transformers 4.31.0 is installed in a private subprocess overlay; PD-MC's 4.30.2 remains unchanged. `--lisa-precision bf16`, `--lisa-model`, and `--lisa-prompt-template` are explicit alternative experiment settings. References record the model, precision, prompt, seed, input hash, and all generated artifact hashes, plus the corresponding execution/placement record.

To score previously generated completions without rerunning inference, use the same output directory with `--benchmark-score-only` and `--reference-dir`. Do not combine score-only and inference-only. Do not run multiple jobs concurrently in the same output directory.

Results are written as:

```text
run_config.json                    # seeds, upstream/model revisions, checkpoint hashes, vocabulary
metric_config.json                 # metric versions and scoring conventions
targets/SOURCE/TARGET/result.json   # target identity, status, iteration count, input/output hashes
targets/SOURCE/TARGET/completion.png
targets/SOURCE/TARGET/completion.jpg
targets/SOURCE/TARGET/mask.png
targets/SOURCE/TARGET/metrics.json
per_target.csv
summary.json                       # per-source and per-metric denominators
table.md
```

Completed targets are reused only when their input and saved completion hashes still match. Previously missing-image targets become pending work when their originals are installed. Changed seeds, MC settings, model/checkpoint contents, vocabulary, or target-selection mode require a new output directory. Reference-selected runs also validate the selection-reference hash during score-only evaluation. Each target has a stable seed derived from its annotation ID, so skipping completed work does not advance another target's random-number sequence. This makes resume order stable; it does not promise bitwise equality across GPU types or CUDA versions.

## Recovering from an oversized Mixed Context crop

The full-run log reached target 265 before failing on `VG/1592205_plane` with `IndexError: index 1200 is out of bounds`. The released Mixed Context function indexes a fixed wallpaper using coordinates from progressively growing object crops. A crop can extend beyond that wallpaper's height or width.

The benchmark adapter now enlarges the wallpaper with aspect-preserving bicubic interpolation only when a background lookup would go out of bounds. All previously valid background lookups and visible query pixels retain their original behavior. The pinned checkout, checkpoint files, seeds, and run configuration remain unchanged; completed target records can be reused.

Copy the updated module from your local project folder:

```bash
scp amodal_benchmark/inference.py daoqm@slurm.csse.rose-hulman.edu:~/Vu_Hung/cv_nangcao/amodal_benchmark/
```

Then rerun on the compute node with the **same output directory**:

```bash
bash run_amodal_slurm.sh --local \
  --benchmark-root ./datasets/open_world_amodal \
  --benchmark-boundary-check released --benchmark-target-selection reference \
  --lisa-device-map auto \
  --cache-dir /dev/shm/daoqm/amodal-cache \
  --output-dir ./output/pdmc-full
```

This historical recovery command explicitly preserves the old run's boundary and selection semantics. Completed targets are reused after hash checks; the interrupted target is retried. No dataset, reference, or model-cache deletion is necessary. The new log records the background enlargement when needed. Use the corrected-run instructions above for a new reproduction experiment.

## Evaluation conventions and limits of exact reproduction

The integration calls `load_models`, `segment`, `QueryObject`, and `run_iteration` from the pinned CS331 upstream commit `bfbfcdec581484586774d9ca53f5c603c97c725b`. Its `main.py` SHA-256 is checked before model loading. The custom Mixed Context Diffusers pipeline, occlusion ordering, SAM, LaMa, crop parameters, and diffusion operations are retained. The default boundary-coordinate repair is documented above; `released` mode retains the original boundary behavior. The model-source substitution uses the accessible SD2 community mirror, pinned to revision `5f74973cbb64c8568780732c17f43eb269d63a0d`.

Defaults retain the released runner's **MC timestep 35** and `max_iter_id=5`. The upstream termination condition is `iter_id > max_iter_id`, so the latter can permit **six** diffusion passes. The original paper identifies timestep 20 as generally favorable; use `--mc-timestep 20` in a separate output directory for that setting. The supplement's three-iteration cap applies to its own proposed method and is not silently imposed on the PD-MC baseline.

The original upstream category vocabulary remains the default. `--classes-file` is available for a separately labelled experiment; expanding to all 553 benchmark classes changes the baseline.

The public release does not include the authors' PD-MC target-selection adapter. Default `auto` resolves to `reference-best`: use the generated **raw LISA mask** to select an existing PD-MC detected instance by maximum IoU (minimum 0.5, stable first-detection tie break). It does not replace the detected mask or class used by PD-MC. Explicit `reference` retains the earlier 0.05 near-tie veto for legacy comparison. `--benchmark-target-selection label` retains word/phrase matching; without references, auto resolves to label mode.

The four metric implementations follow the authors' [released `eva_amodal.py`](https://github.com/saraao/amodal/blob/14d59e09c052585bdb3b0f447078fa877d3d3321/eva_amodal.py):

- **CLIP:** raw image/text logits from `openai/clip-vit-base-patch32`, using the original label and the white-composited completed RGBA object. The model revision is pinned. The released script incorrectly fills its aggregate “Average CLIP Score” with OpenCLIP cosine scores; this integration aggregates the actual CLIP logits.
- **LPIPS:** AlexNet backbone, RGB tensors normalized to `[-1, 1]`.
- **VGG16 feature similarity:** ImageNet VGG16 feature maps, 224×224 preprocessing, channel-wise cosine similarity averaged spatially.
- **SSIM:** the released evaluator's RGB, window-size, and prediction-dependent data-range conventions.

Transparent pixels are composited over white. Default `--benchmark-metric-alignment released` compares the generated `sd_img_cut`-style visible crop with the completed RGBA object, using `eva_amodal.py`'s bounding box of reference `alpha < 255` and crop/resize conventions. The optional `original-frame` mode instead uses the saved crop offsets and a shared canvas; it is a separate scoring protocol. The exact-copy convention is retained: LPIPS exactly 0 becomes 1, VGG similarity exactly 1 becomes 0, and SSIM exactly 1 becomes 0. Tiny/constant images for which the released SSIM formula is undefined cause a scoring error rather than a fabricated score.

Model failures remain in the appearance denominator, with the paper's penalties **LPIPS=1, VGG similarity=0, SSIM=0**. Missing source images are unavailable data, not model failures. Infrastructure/model exceptions stop the job and remain resumable, rather than being counted as unsuccessful completion.

The paper does not specify a CLIP score for failed completions; this integration leaves it null and reports **CLIP over successful outputs, with `CLIP_n` shown explicitly**. This convention is not asserted to reproduce the reported all-instance CLIP row. Overall means are weighted over instances, not averaged over four source means. Every metric's denominator is in `summary.json`.

For reference, the supplied PDF's published PD-MC Table 3 row is:

| Source | CLIP ↑ | LPIPS ↓ | VGG similarity ↑ | SSIM ↑ |
| --- | ---: | ---: | ---: | ---: |
| VG | 28.367 | 0.578 | 0.413 | 0.463 |
| COCO-A | 27.383 | 0.664 | 0.328 | 0.382 |
| Free images | 28.333 | 0.720 | 0.279 | 0.309 |
| LAION | 27.573 | 0.692 | 0.299 | 0.346 |
| Overall | 27.984 | 0.628 | 0.364 | 0.413 |

These are **published reference numbers**, not newly reproduced results. Generated reference masks, reconstructed LAION bytes, the recorded target-selection rule, and the CLIP failure convention can affect numerical agreement. The framework marks `exact_table_reproduction=false` until measured agreement is established; reference preparation itself is implemented and requires no external PNG release. The main CS331 pseudo-occlusion evaluation is a different benchmark.

## Validation performed locally

Archive checks, metadata hashes/counts, complete image decoding, dataset preparation, CPU preflight, and unit/shell regression checks can run locally. PyTorch/CUDA inference and the real pretrained metric networks have **not** run on this Windows host; no new benchmark scores are claimed.

An earlier server log confirmed multi-GPU reference preparation and PD-MC inference through 265 targets; the subsequent full run's reported table contains 1,225 completions. The oversized-background and boundary repairs are tested locally against pinned upstream functions with CPU substitutes for expensive model calls. No corrected GPU run or Slurm job was submitted locally.

Latest local verification on 2026-10-20: **77 Python tests and all four shell regression scripts passed** on WSL, plus Python compilation and Bash syntax checks. This includes paper comparisons, data/reference identities, compatible seed statistics, score-only seed preservation, and chained Slurm submissions with failure propagation. Independent review found no remaining important issues. The new reports audit the archived full runs; no additional GPU run was performed, and CUDA-extension/pipeline import validation still requires the provisioned cluster environment and upstream checkout.

```bash
python3 -m unittest discover -s tests -v
bash tests/test_run_amodal_slurm.sh
bash tests/test_open_world_slurm.sh
bash tests/test_run_modes.sh
bash tests/test_script_directory.sh
```

Primary sources: [open-world authors' repository](https://github.com/saraao/amodal), [official dataset access page](https://forms.gle/AQCPVRnzFfB449KfA), [PD-MC authors' repository](https://github.com/k8xu/amodal), and the two supplied local PDFs.
