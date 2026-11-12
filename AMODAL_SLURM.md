# Running the amodal-completion paper on Slurm

For an expanded detector vocabulary, add `--benchmark-detector-vocabulary annotated` to the dedicated SAM launcher and use a new `--output-dir ./output/pdmc-open-world-sam-annotated`. This retains scene categories and queries missing per-image annotation categories separately. See [the vocabulary experiment and four-worker commands](OPEN_WORLD_REPRODUCTION.md#expand-the-detector-vocabulary-using-image-annotations). The default remains the original detection vocabulary.

For OpenWorld evaluation without LISA, use `bash run_pdmc_open_world.sh --local --cache-dir /scratch/$USER/amodal-cache --benchmark-limit 2` inside a compute allocation. Remove the limit for all targets; omit `--local` to submit a Slurm job. This separate launcher uses released PD-MC with Grounding DINO + SAM, defaults to semantic label matching and automatic backdrop sizing, and saves results and SAM appearance references in `output/pdmc-open-world-sam-semantic-v2`. Corrected matching requires new outputs/references; resume historical SAM experiments with explicit `--benchmark-target-selection label` and their original `--benchmark-background-sizing released|auto`. The shared runner retains its legacy SAM defaults. See [the protocol and full commands](OPEN_WORLD_REPRODUCTION.md#released-pd-mc-with-grounding-dino--sam-only).

For evaluation on the 2,379-image open-world benchmark from `supplement_paper.pdf`, see [OPEN_WORLD_REPRODUCTION.md](OPEN_WORLD_REPRODUCTION.md). The runner accepts `--benchmark-root` and automatically generates visible RGBA references with LISA before PD-MC and scoring. All selected source images are installed; LAION reconstruction details are recorded in the guide. The ordinary `--input-dir` workflow below remains the qualitative entry point.

For controlled boundary/matching experiments, `run_open_world_controls.py` writes a plan by default and submits/runs it only with `--execute`. It requires an explicit existing reference directory, reuses it across inference seeds, and gives each experiment/seed its own output directory. Submitted jobs are chained with `afterok` dependencies to avoid concurrent setup in their shared cache. `compare_open_world.py` audits saved CSVs against summaries and reports paper deltas and compatible seed statistics without loading GPU models. See [the reproduction workflow](OPEN_WORLD_REPRODUCTION.md#paper-comparison-and-controlled-reproduction-runs) for commands, resource settings and interpretation.

Benchmark defaults now repair PD-MC's reversed boundary coordinates (`--benchmark-boundary-check paper`) and use best-IoU reference matching (`--benchmark-target-selection auto`, resolving to `reference-best`). Start corrected runs in `output/pdmc-paper-v2`; retain `output/pdmc-full` for comparison. Resuming the earlier experiment requires explicit `--benchmark-boundary-check released --benchmark-target-selection reference`. The reproduction guide includes transfer and `sbatch` commands. Existing references and model caches are reusable.

Benchmark inference defaults to `--benchmark-workers auto`, running one independent PD-MC process per allocated CUDA-visible GPU. Use `--benchmark-workers N` to choose a smaller count or `1` for serial execution. Workers retain the same experiment settings and per-target seeds, reuse completed targets, and write separate logs under `OUTPUT/workers/<execution-id>/`; the coordinator scores and combines their results after success. Each worker loads a full pipeline, so request host RAM and CPU resources for the worker count (an eight-GPU example is `--gres gpu:8 --cpus 32 --mem 128G`). LISA reference generation remains a separate preceding stage. See the reproduction guide for transfer commands, cancellation behavior, and validation limits.

`run_amodal_slurm.sh` is the entry point for the CVPR 2024 implementation. Run it from a login node; it submits a one-GPU job automatically. The job then clones the pinned upstream commit, creates a private Python 3.10 environment, compiles GroundingDINO inside the GPU allocation, downloads the checkpoints, and runs the official progressive Mixed Context pipeline.

If you are already on the compute node, use `--local` to execute directly without submitting another job, including when `SLURM_JOB_ID` is absent:

```bash
bash run_amodal_slurm.sh --local --input-dir ./input
```

An existing Slurm allocation is also detected automatically. `--local` uses the current allocation; it does not request new GPU, memory, or wall-time resources.

Helpers and default paths are resolved beside `run_amodal_slurm.sh`, even when the allocation's `SLURM_SUBMIT_DIR` points to your home directory. Direct `sbatch` spool copies use the submission directory; self-submitted jobs preserve the project directory explicitly. `AMODAL_SCRIPT_DIR` can override this location. If an older runner reports that `/home/daoqm/prepare_visible_references.py` is missing, copy the updated runner to the project and restart from that directory with the same output path to resume completed targets.

The default input is `./input`, which already contains the sample image. Results are written to `./output`, and Slurm logs are written to `./logs`.

```bash
chmod +x run_amodal_slurm.sh
bash run_amodal_slurm.sh --input-dir ./input
```

The file also contains conservative `#SBATCH` defaults, so `sbatch run_amodal_slurm.sh --input-dir ./input` works on clusters where the default partition is suitable. The `bash` form is recommended when you need to pass the resource options shown below.

Most clusters require a site-specific partition or account. Pass those at submission time:

```bash
bash run_amodal_slurm.sh \
  --partition gpu \
  --account MY_ACCOUNT \
  --gres gpu:1 \
  --cpus 8 \
  --mem 48G \
  --time 24:00:00 \
  --input-dir /scratch/$USER/images \
  --output-dir /scratch/$USER/amodal-output
```

The cache contains the repository, Conda environment, model checkpoints, and Hugging Face files. Put it on persistent scratch when the home directory is small:

```bash
bash run_amodal_slurm.sh \
  --cache-dir /scratch/$USER/amodal-cache \
  --input-dir /scratch/$USER/images \
  --output-dir /scratch/$USER/amodal-output
```

The first job performs setup and can take a while because the InstaOrder checkpoint archive is several gigabytes. To separate setup from inference, submit this once:

```bash
bash run_amodal_slurm.sh --setup-only --cache-dir /scratch/$USER/amodal-cache
bash run_amodal_slurm.sh --run-only --cache-dir /scratch/$USER/amodal-cache --input-dir ./input
```

The runner uses the public `sd2-community/stable-diffusion-2-inpainting` mirror because the original `stabilityai` endpoint is deprecated and returns 401. You can select another accessible SD2 inpainting repository or a complete local Diffusers model directory with `--sd-model` (or `AMODAL_SD_MODEL`). A private repository may need `HF_TOKEN`; it is inherited by the job and is not placed in the command line:

```bash
export HF_TOKEN=hf_...
bash run_amodal_slurm.sh --input-dir ./input
```

If the cluster exposes CUDA through environment modules, set the module names before submission, for example `export AMODAL_MODULES="cuda/11.7 gcc/11"`. The default Mixed Context timestep is 35, matching the released `main.sh`; override it with `--mc-timestep 20` when reproducing the paper's typical appendix setting. The pipeline needs a GPU with roughly 24 GiB of memory for the released configuration.

## Recovering from the GroundingDINO build error

The supplied log fails because the node uses GCC 12.2, while PyTorch 1.13.1's CUDA 11.7 extension builder requires GCC 6.0 through 11.5.0. The runner now selects a compatible GCC/G++ pair and sets both `CC` (also used as nvcc's host compiler) and `CXX`. If none is available, it installs GCC/G++ 10 into the private Conda environment. It also checks that the CUDA compiler, runtime headers/libraries, and Thrust headers form a complete toolkit matching the requested CUDA runtime (11.6 or 11.7).

Re-run the regular command on the cluster, with your usual resource and path options:

```bash
bash run_amodal_slurm.sh --input-dir ./input
```

Do not use `--run-only` for this repair. The updated dependency cache marker triggers installation once without deleting existing checkpoints. Shared constraints stop GroundingDINO's unpinned requirements from upgrading NumPy, SciPy, OpenCV, or PyTorch. Albumentations 0.5.2 and imgaug retain LaMa's `DualIAATransform` API; YAPF 0.32.0 retains the config formatter's `verify` argument. Matching OpenCV distributions satisfy upstream package metadata, with the headless build installed last for compute nodes. User-site Python packages are disabled.

Before downloading checkpoints, startup validation checks package consistency, imports the complete inference entry point, and runs a small GroundingDINO CUDA operator. These checks also run with `--run-only`. The actual CUDA build and model inference must be verified in the cluster's GPU allocation.

## Recovering from Conda's MKL hook error

The newer log stops in `libblas_mkl_deactivate.sh` because its `CONDA_MKL_INTERFACE_LAYER_BACKUP` variable is unset. Installing packages into an active environment makes Conda automatically reactivate it, including hooks from newly installed packages whose activation hook has not run yet. The runner now temporarily disables `nounset` for every Conda call, restores the caller's setting afterward, preserves activation changes in the current shell, and propagates command failures.

Every Conda create/install transaction now includes Python 3.10, NumPy 1.23.5, MKL below 2024.1, the requested CUDA version, and compatible setuptools/wheel/packaging versions. This also prevents the MKL 2024.2 and NumPy 2.2.6 selections shown in that log from causing subsequent PyTorch import or NumPy ABI failures. cuRAND and cuFile are pinned to the requested toolkit's component versions because their older package metadata otherwise allows newer CUDA 12/13 libraries into the environment. GCC 10 uses a glibc 2.17 sysroot so compiled extensions remain compatible with older cluster nodes. Versioned PyTorch and dependency markers trigger repair of an existing environment; checkpoints do not need to be deleted. Re-run the normal command above without `--run-only` once.

Local shell regression checks, including reproduction of the unset MKL backup variable, can be run without a GPU or Conda installation:

```bash
bash tests/test_run_amodal_slurm.sh
```

## Recovering from missing CUDA library headers

The latest log fails while compiling GroundingDINO with `fatal error: cusparse.h: No such file or directory`. PyTorch's `ATen/cuda/CUDAContext.h` includes cuSPARSE, cuBLAS, and cuSOLVER headers, including when compiling GroundingDINO's CPU source. Installing the CUDA runtime, nvcc, and Thrust alone does not provide these library development headers.

The runner now installs matching `libcusparse-dev`, `libcublas-dev`, `libcusolver-dev`, and `cuda-driver-dev` packages. It rejects cached toolkits missing any required header or linker library. Before installing GroundingDINO, both the selected C++ compiler and nvcc compile a small source including PyTorch's CUDAContext and Thrust headers, checking the complete include chain. The v4 cache marker causes an existing setup to repair itself once when run without `--run-only`; checkpoints remain cached.

To separately verify a provisioned environment and checkout on a GPU machine, the integration check rebuilds the extension and exercises its forward/backward CUDA operators without loading model checkpoints:

```bash
bash tests/validate_cuda_extension.sh /path/to/conda-env /path/to/amodal
```

## Recovering from the Segment Anything namespace import error

The next log reaches `CUDA operator OK: groundingdino._C` and then fails with `cannot import name 'SamPredictor' from 'segment_anything' (unknown location)`. The outer checkout directory was being imported as a namespace package before setuptools' editable import finder could load the inner SAM package.

Both preflight and inference now use one shared import-path configuration. It places `Grounded-Segment-Anything/segment_anything` and `diffusers/src` before the surrounding checkout directories. Preflight verifies the SAM package's origin and imports `build_sam`, `SamPredictor`, and `SamAutomaticMaskGenerator`, instead of accepting a successful namespace import.

Full pipeline import validation also found that LaMa needs Kornia's `SamplePadding` API. The runner now pins Kornia 0.5.0, matching the upstream requirement. The v5 dependency revision updates Python packages in a prepared v4 environment without reinstalling PyTorch or rebuilding GroundingDINO. Re-run the normal command with your existing cache directory; `--run-only` still requires that dependency repair and checkpoint downloads have completed.

The namespace regression test and the real pipeline import check can be run separately:

```bash
python3 tests/test_amodal_import_paths.py
bash tests/validate_pipeline_imports.sh /path/to/conda-env /path/to/amodal
```

## Recovering from the InstaOrder Google Drive download error

The `Cannot retrieve the public link` error was caused by a case-sensitive typo in the file ID: `1_GEmCmofLSKJZnidfp4vsQb2Nqq5aqBU` returned 404. The correct ID from the upstream InstaOrder README is `1_GEmCmofLSkJZnidfp4vsQb2Nqq5aqBU` (lowercase `k`). The runner uses the corrected ID with `python -m gdown.cli`; its public ZIP download was verified with the pinned gdown 4.7.3 by reading only the initial ZIP bytes.

Re-run the regular command with the same cache directory and without `--run-only`. Existing dependencies and the already downloaded GroundingDINO/SAM checkpoints are reused.

## Recovering from Stable Diffusion's 401 error

The latest run downloaded the four checkpoints and reached model loading, then failed because `stabilityai/stable-diffusion-2-inpainting` was unavailable and uncached. The default source is now its public community mirror, `sd2-community/stable-diffusion-2-inpainting`. The runner retains the pinned custom Mixed Context pipeline and float16 settings; it replaces only the model-source arguments while executing `main.py`, leaving the upstream checkout unchanged.

Preflight fetches the small model configuration files and verifies the SD2 inpainting architecture (nine UNet input channels, four latent channels, and a 1024-dimensional text encoder) before checkpoint downloads and inference. Model weights are downloaded into the existing Hugging Face cache on first use. Existing environments and non-Hugging-Face checkpoints are reused.

```bash
bash run_amodal_slurm.sh --cache-dir /dev/shm/daoqm/amodal-cache --input-dir ./input
# Or use an already downloaded model:
bash run_amodal_slurm.sh --sd-model /scratch/models/sd2-inpainting --input-dir ./input
```

Model selection, both upstream pipeline branches, local model paths, and config validation have a standalone regression check:

```bash
python3 tests/test_amodal_model_source.py
```
