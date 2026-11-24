# Annotated Surface-PMC: full-run continuation

**Completed and recovered, 14 November 2026.** All 2,565 targets in both arms
are verified and registered. Surface 39490 timed out; the planned continuation
39491 completed. Finalizer 39492, independent full/unseen audit 39497 and export
39495 completed. The collector recovered all four exports and exited. No study
job needs resubmission. Read the [completed full results](SURFACE_ANNOTATED_FULL_RESULTS_20261114.md)
and [local review](../output/surface-annotated-full-review-20261114/REPORT.md).
The job map and recovery instructions below are retained as historical procedure.

This is the continuation of the frozen 12 November annotated-vocabulary study.
The original 12 November interactive work window was 06:46-11:46 UTC. The independent 192-image /
204-target cohort and the full 2,565-target benchmark are separate deliverables.
Both compare the current annotated PD-MC control with the unchanged transferred
Surface-PMC method. No new parameter fitting or diffusion training is running.

Remote project: `daoqm@slurm.csse.rose-hulman.edu:~/Vu_Hung/cv_nangcao`.
Active study: `output/surface-annotated-20261112-freshrefs`.
Design, source freezes and exact submission commands:
`output/surface-annotated-20261112-design`.

| Job | Purpose | Dependency / behavior |
| --- | --- | --- |
| 39486 | Fresh annotated cohort control | Completed; preserve its snapshot |
| 39487 | Surface-PMC cohort | Verified smoke and matching copied references |
| 39489 | Frozen final-pool diagnostic | After successful cohort |
| 39498 | Independent native cohort audit | After successful cohort |
| 39499 | Independent native pool audit | After diagnostic and cohort audit |
| 39488 | Full annotated control | After cohort and diagnostic scheduling gate |
| 39490 | Full Surface-PMC | After full control references are verified and copied |
| 39491 | Unchanged Surface-PMC timeout continuation | Resume only if 39490 times out; skip if it completes; otherwise fail for diagnosis |
| 39492 | Full comparison, provenance verification, registration | After full control and continuation succeed |
| 39497 | Independent full and unseen-image sensitivity audit | After full finalizer |
| 39495 | Full metadata archive | After independent full audit |

Every GPU evaluation requests four GPUs and four workers. These jobs are already
submitted: do not submit duplicate runs or change their frozen output directories.
Actual completion states belong in the latest saved status, not this job map.
Failed historical-reference attempts remain in the separate original study path.

From the local project directory, inspect current states without reading partial
quality outcomes:

```powershell
python scripts/monitor_annotated_surface.py --compact
```

The hidden `scripts/collect_annotated_surface.py` process recovers only completed
exports, checking independent archive and member SHA-256 values. Its state is
`output/surface-annotated-20261112-recovery/collector-status.json`. If the local
computer stops, remote Slurm jobs continue. Restart the collector only after
checking that an existing instance is no longer running. Individual completed
exports can be recovered with `scripts/recover_annotated_surface.py`.

The paired cohort report uses immutable `baseline-cohort` and `surface-cohort`
snapshots. Continuing `baseline` and `surface` does not replace those snapshots.
Never interpret a cohort or partial run as a full registered experiment. The full
result includes prior controlled-study images; its frozen unseen-image analysis
must be reported separately and does not inherit the cohort's near-copy screen.

When the full jobs finish, inspect `full-comparison/comparison.json`, the separate
full audit and the registry receipt. Recover and check the completed evidence,
then update the research report with full denominators, failure transitions,
per-source results, confidence intervals, compute, and the unseen sensitivity.
Do not refit the selector against this cohort or replace an unfavorable endpoint.

If a job fails rather than times out, retain its artifacts and diagnose the
failure before resuming. In particular, do not relax the exact SAM-reference
guard or overwrite reference bytes to make a run continue. The documented
[reference-runtime amendment](SURFACE_ANNOTATED_REFERENCE_AMENDMENT_20261112.md)
already specifies how the fresh matched comparison differs from historical
annotated outputs.
