# Test guide

Run the complete offline checks from Linux/WSL in the CPU environment:

```bash
PYTHON_BIN=/path/to/cpu/environment/bin/python bash scripts/check.sh
```

The script runs Python unittest discovery, all five `test_*.sh` regression scripts and the synthetic CPU smoke. Windows can run the Python suite and smoke; Bash, symlink and POSIX process tests need Linux/WSL.

| Area | Test files |
| --- | --- |
| Dataset identity, target selection, geometry and cache guards | `test_open_world_benchmark.py` |
| Semantic labels and annotated detector queries | `test_semantic_matching.py`, `test_detection_vocabulary.py` |
| Pinned upstream gates and backdrop repair | `test_inference_protocol.py`, `test_mixed_context_background.py` |
| Reference crops, erosion, provenance and LISA placement | `test_visible_references.py`, `test_lisa_placement.py` |
| Worker scheduling, cancellation, serial/parallel parity and resume | `test_parallel_benchmark.py`, `test_parallel_evaluation.py` |
| Configuration, provenance, verification, ledger and smoke | `test_project_workflow.py` |
| Published comparisons and controlled experiment plans | `test_reproduction.py` |
| Model source and editable import paths | `test_amodal_model_source.py`, `test_amodal_import_paths.py` |
| Provisioning helpers, run modes, project paths and benchmark forwarding | All five `test_*.sh` files |

Keep tests flat: unittest discovery and existing Bash commands rely on this layout. Pinned [fixtures](fixtures/README.md) let tests generate synthetic images without downloading the benchmark. Optional historical tests use the original release ZIP or saved reports when present. CPU tests replace expensive model calls and establish control flow and artifact consistency, not model quality.

Real GPU checks remain explicit and require the provisioned runtime and upstream checkout:

```bash
bash tests/validate_cuda_extension.sh /path/to/cache/conda-env /path/to/cache/amodal
bash tests/validate_pipeline_imports.sh /path/to/cache/conda-env /path/to/cache/amodal
```

The extension check builds and exercises a real CUDA operator. These helpers are separate from `scripts/check.sh` and do not establish full evaluation quality or memory fit.
