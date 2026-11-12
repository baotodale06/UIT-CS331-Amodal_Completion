# Fix SAM Evaluation Wrapper Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this bounded repair inline, with one final code review. Steps use checkbox syntax for tracking.

**Goal:** Correct avoidable annotation-to-detection mismatches, retain genuine ambiguity, and report unfinished outputs and failure stages clearly without changing PD-MC inference.

**Architecture:** Add a CPU-only, versioned semantic-label matcher with explicit aliases and exact-mask deduplication. Expose it as `label-semantic`; dedicated SAM entry points select it by default, while the shared runner retains legacy `label` behavior. Record the matcher in inference/reference configurations so corrected runs cannot mix with existing results. Reporting preserves the upstream saved-output success convention and adds completed/iteration-limited output counts and failure stages.

**Tech Stack:** Python 3.10-compatible stdlib, NumPy, Pillow, unittest, Bash/Slurm.

**Spec:** The user-authorized correction follows the confirmed counterexamples and audit in `output/pdmc-open-world-sam-bgfix-audit/failure_diagnosis.json`.

## Global Constraints

- Preserve vocabulary, detected-class diffusion prompts, checkpoint pins, seeds, thresholds, and released boundary behavior.
- Preserve the full completed experiment. Changed matching requires a new output/reference directory and cannot use background-only result import.
- Keep LISA absent from the SAM runner. Use the already validated background sizing repair by default in the dedicated runner.
- Do not guess among distinct same-class objects; do not merge merely overlapping masks.
- Keep historical reports readable and permit explicit legacy label matching when resuming old runs.
- No Git metadata is present; work in place without commits or worktrees.

## Review Focus

- Compound nouns such as hot dog and table lamp must not select dog or dining table incorrectly.
- Distinct masks with identical labels remain ambiguous; identical masks use the highest detector confidence, preserving its original class/prompt.
- Missing SAM masks with detector labels must remain distinguishable from no detector labels.
- Coordinator/worker/score-only configurations and generated references must agree on matcher identity.
- CSV boolean strings must not make false iteration-limit flags truthy; old CSVs must remain valid.

## Tasks

- [x] Add failing CPU matcher regressions; implement aliases, noun-head matching, exact deduplication, and complete candidate diagnostics in `amodal_benchmark/matching.py`.
- [x] Add failing evaluator/worker/resume tests; wire `label-semantic`, configuration versioning, reference identity guards, and corrected standalone defaults.
- [x] Add reporting regressions; expose completion-state and failure-stage counts, CSV flags, and legacy-compatible report loading.
- [x] Update deployment/run documentation and shell forwarding checks. Run the full Python/shell suites in WSL and CPU dataset preflight.
- [x] Review the complete change, address actionable findings, and provide transfer plus a small corrected smoke-run command. GPU evaluation remains a cluster check.

## Execution ledger

- Ruling: keep released boundary behavior for this wrapper fix, isolating the target-matching change; its independently diagnosed boundary bug remains a separately controlled correction.
- Ruling: only exactly equal masks are deduplicated, avoiding an unvalidated IoU threshold that could merge distinct objects.
- Completed dedicated Python/Bash defaults (`label-semantic`, automatic backdrop sizing); the Bash default output is now `output/pdmc-open-world-sam-semantic`. Explicit legacy options and saved score-only settings remain usable.
- Added score-only reference-configuration verification and worker matcher-drift regressions. Legacy outputs cannot be imported into changed matching experiments.
- Final review corrected a report-loader `KeyError` when completion-state columns are removed: the CSV is now checked against the summary's reporting fields and raises a clear mismatch. Historical CSVs remain supported.
- Validation: all 104 Python tests passed in WSL; all five `tests/test_*.sh` suites passed; input preflight passed for all 2,565 targets; standalone two-target check-only passed without creating outputs; Python 3.10 syntax parsed successfully for 24 source/test files.
- Transfer commands, corrected smoke command, legacy resume options and GPU-validation commands are documented in `OPEN_WORLD_REPRODUCTION.md`. Real CUDA-extension/import validation and model inference were not run: the provisioned upstream/model environment is on the cluster.
