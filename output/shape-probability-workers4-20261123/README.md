# Shape-only probability training with four input workers

The execution chain started at 17:26:48 UTC on 23 November after the completed
predecessor passed fresh independent replay and Kaggle probability version 1
still reported QUEUED. Calibration reproduced all 1,264 original Quadro masks;
independent calibration replay and optimizer preflight both passed. Training
started at 17:29:23 UTC. At 17:44 UTC it has completed 2/20 epochs, selected epoch
zero, with epoch 3 running. Actual handles: observer PID 3532, Windows WSL process
2980, WSL trainer 502. Calibration processes 28772/424 are already terminal.
See `launch-request.json`, `execution.json`, `calibration-replay.json` and
`launch-milestone.json`. There is no eligible learned checkpoint yet.

Activated 23 November 2026 at 17:07 UTC as observer PID 3532. Its first poll
identified the original observer PID 21512. After checking that no fallback GPU
phase had started, the retirement helper stopped only PID 21512 and recorded its
startup fence. At 17:08 UTC the replacement entered `waiting_predecessor`, with
original WSL launcher 328 and trainer 495 still live. The old exit-1 notification
is the expected observer retirement, not a training failure.

Full CPU checks passed: 458 tests, 26 optional-dependency skips, six Bash
regressions and CPU smoke. All 19 focused pinned-environment tests passed without
skips. The frozen training archive is
`fe6b8235e4dd4e5908e3452558f4f281d0f18c0fbc392aeb1543f8840a2268f3`;
the 54-file operational manifest is
`9a531b2d0b9ddc938518db4a0ac0ff3cffc694f814eb8fc030bea22a9cfb56fe`.
No new GPU phase or benchmark admission has started at this activation milestone.

This is a separate execution of the schema8 probability-loss experiment. Only
the CPU data-loader worker count changes from two to four. The model, original
v2 initialization, 9,494 training objects, seed, augmentation, loss, effective
batch64/microbatch8, 20 epochs, optimizer, native arithmetic and checkpoint
selection are retained. Existing training and source snapshots remain unchanged.

The read-only CPU diagnostic measured 512 sampled objects in order 2/4/4/2
workers. Mean loading time was36.248s with two workers and22.965s with four,
a1.578x input-throughput ratio. Every sampled tensor byte matched. This is a
concurrent-training CPU microbenchmark, not an end-to-end training speedup.

A separate bounded GPU comparison performed two real optimizer updates for each
worker count. All model parameters/buffers, optimizer tensors, loss, scaler and
CPU/CUDA random-number states matched exactly. Both used all128 sampled objects,
had no skipped updates and preserved the encoder. State fingerprint SHA256:
`3593efc84695777e2ed1d1cd500a4a126cdc25bf3c7eaeb4b324da0eaa6b420a`.
No full-trajectory equivalence or model-quality gain follows from two updates.
Diagnostic evidence is in `../shape-loader-profile-20261123/`.

The new observer must first see the old probability observer exit. It also probes
the real WSL processes for both source directories, requires successful completed
schema6 training and independent native replay, then checks explicit Kaggle
probability version1. Only a fresh QUEUED status permits local training. RUNNING
or COMPLETE skips it; errors do not launch anything. A persistent request prevents
duplicate launches. When superseded, the original observer gets an explicit
retirement record and startup fence; neither its sources nor its original
activation evidence is overwritten.

Before optimization, calibration must reproduce all1,264 original Quadro native
masks exactly. Real optimizer preflight, all20 epochs, both full/hidden geometry
floors, stable checkpoint selection, epoch-zero exclusion, selected native pixel
replay and independent CPU replay/reporting remain mandatory.

Read `derivation.json`, `staging.json`, `cpu-validation.json`,
`operational-manifest.json`, `activation.json`, `queue-status.json`, and eventually
`execution.json` for actual preparation/activation state. Preparation alone does
not mean that this observer or any GPU phase has started.

No benchmark admission exists. A verified eligible nonzero epoch still needs
Quadro admission, smoke, all2,565 benchmark targets, failure penalties, CLIP
coverage, unseen/native audits and at least3% relative improvement on all four
metrics. The goal remains unachieved. Existing calibrated jobs45680-45685 remain
separate and unchanged.
