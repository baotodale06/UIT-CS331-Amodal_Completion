# Accepted baseline and historical launch evidence

**Job 31548 is complete, verified and registered. Do not resubmit this completed baseline.**

See [REPORT.md](REPORT.md) for the final results and verification evidence. The accepted remote run is `/home/daoqm/Vu_Hung/cv_nangcao/output/pdmc-baseline-provenance-20261105`; the verified local copy is `../../output/baseline-31548/output/pdmc-baseline-provenance-20261105`. The registered entry is `ec823094e31f226957ef0442f0480cf14ae2db91c79550a9701a9e0aef99b07c`.

The baseline used the unchanged `configs/baseline.json`, persistent `.amodal-cache`, four Quadro RTX 6000 GPUs, four workers, 16 CPUs and 64 GiB RAM. All nine workflow stages passed. Cluster elapsed time was 4h 02m 38s, ending at 2026-11-06 09:27:10 UTC.

## Preserved operational evidence

- [Exact successful submission](submission-repaired-20261106.json)
- [Launcher repair, source backup and parser tests](launcher-repair-20261106T052337Z.json)
- [Slurm completion and allocated resources](slurm-accounting-31548.json)
- [Full source/runtime audit](accepted-source-runtime-audit.json)
- [Complete remote file hashes](completed-remote-manifest.json)
- [Recovery receipt](accepted-recovery-receipt.json)
- [Independent local full/provenance verification](full-independent-verification.json)
- [Earlier launch/status notes](LAUNCH-before-acceptance.md), retained as historical evidence

Earlier runtime probes, failed L40S smoke, launcher failure, source backups, submissions and monitoring snapshots remain in this package. The stopped partial bulk archive is transport evidence, not an accepted complete archive. The local watcher was retired after cluster completion and the final manual recovery; it must not be restarted for this completed job.

Future changed experiments need a new output directory, their own explicit contract and provenance, and registration against this baseline. Keep SAM and LISA reference protocols separate.
