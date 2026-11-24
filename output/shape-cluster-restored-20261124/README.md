## SSH restored; GPU maintenance still active (24 November 2026, 12:35 UTC)

SSH and fresh Slurm accounting work again. Existing collectors14576/2860/22844
are identity-verified alive and have each resumed successful polling after12:33UTC.
This supersedes the preceding SSH/DNS failure and stale-accounting statements.
All12 shape-only jobs remain PENDING. Roots45673(v2 full),45677(constrained
preflight),45680(calibrated native) wait for reserved maintenance; downstream
jobs have normal dependencies. Both GPU nodesgebru/gus and all CPU nodes remain
in maint. The active all-node reservation ends25November13:00UTC/20:00Vietnam.
No training/evaluation job or collector was restarted, duplicated or canceled.

V2 retains222 historical target records; no new full benchmark output is ready.
Private Kaggle147113223/version1 remains QUEUED; observer23064 is alive.
The completed local probability training and fixed-offset diagnostic remain closed
with no eligible new checkpoint. Continue the preserved Kaggle study and Quadro
chains; full2,565-target >=3% on allfour metrics remains unachieved.

Evidence: `output/shape-cluster-restored-20261124/restoration.json`.

