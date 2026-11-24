## Fixed final-epoch diagnostic completed: no retained offset (24 November 2026, 12:31 UTC)

Recovery completed all four phases at12:30:09UTC: real GPU preflight, independent
preflight replay, full diagnostic, and independent full native-mask replay/report.
All1,264 objects/217images passed, covering8,848 exported masks (six variants and
the separate zero-offset reference). Original Quadro pixels and final zero-offset
repeats match exactly; recorded epoch20 geometry, hashes, identities, support, crop
bounds, empty masks, monotonic offsets and summaries all pass. Processes5580/28752
have exited. Original failed attempt, policy/source snapshots and parent remain intact.

None of offsets[0,0.5,1,1.5,2] passes both unchanged geometry floors. All fail hidden
IoU: the largest is0.3609508314 at+1, below0.3703334757; that offset also misses the
full floor. No new checkpoint or offset is promoted. Better unshifted Brier/NLL
does not establish hidden geometry retention; probability replay checks bounds and
aggregation only. This is reused validation, not a full-benchmark improvement.

Report: `output/shape-probability-final-recovery-20261124/full-replay.md`.
Assessment/chart: `assessment.json` and `diagnostic.html` in that directory.
Independent replay SHA256:6685eadd221a1ad4d4e849d8325173c15ea6581aab7abc6470d2ac1afea3d254.

Kaggle147113223/version1 remains QUEUED as of 2026-11-24T12:31:38.670884+00:00;
existing observer23064 and three cluster collectors14576/2860/22844 are identity-
verified alive. Current quota:1.492533 GPU-hours used of30, zero reserved. SSH still
cannot resolve the cluster host; an independent public resolver also returned no
host address. Last successful cluster accounting remains23November19:10UTC.
No jobs were resubmitted, canceled or changed. Preserve all three Quadro chains
and the calibrated-initialization Kaggle study. Maintenance is scheduled to end
25November13:00UTC/20:00Vietnam. The >=3% all-four-metric full2,565-target goal
remains unachieved; only eligible verified training may proceed to Quadro admission.

# Fixed final-epoch diagnostic recovery

All seven schema8 parent phases completed successfully at 12:21:19 UTC. All 20 epochs
completed, with 189,880 sampled objects and zero skipped optimizer steps. Independent
selection replay finds no trained epoch meeting both unchanged floors: maximum trained
hidden IoU 0.3476279268 is below 0.3703334757. Selected epoch remains zero. The complete
1,264-object/217-image native replay reproduces original Quadro masks with zero differing
pixels. Parent3532 and trainerWSL502 exited normally; do not restart them. Report:
`output/shape-probability-workers4-20261123/report/REPORT.md`.

Existing diagnostic observer36128 started after parent closure and failed before inference:
its support-only training bundle omitted sam-prediction.png. The failed attempt is preserved.
A separate recovery uses the complete original native validation export, whose result hash
and all 2,528 required mask artifacts match the frozen references. Policy, sources, epoch20
checkpoint and offsets [0,0.5,1,1.5,2] are unchanged. Three-object GPU preflight and its
independent replay passed; full diagnosis/replay is proceeding under WindowsPID5580.
See `output/shape-probability-final-recovery-20261124/execution.json`. Do not duplicate it.

Kaggle147113223/version1 remains QUEUED; observer23064 is alive. Cluster collectors
14576/2860/22844 are alive; fresh SSH fails DNS resolution. No jobs were duplicated or
changed. Maintenance ends 25November13:00UTC. No new benchmark admission or verified
>=3% four-metric full-benchmark result exists. Continue the live diagnostic, Kaggle observer
and preserved Quadro chains. This is iterative research on reused development data.

