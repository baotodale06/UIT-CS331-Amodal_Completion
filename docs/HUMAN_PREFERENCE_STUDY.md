# Human preference study

Prepared on 22 November 2026 for the user's own ratings. No human preferences
have been collected or inferred. This is a separate study from the November 8
single-method appearance audit.

## Current round

The user chose the completed full Surface-aware run first, with Shape-aware
added only after its full run finishes. The combined method is reserved for a
later round.

The frozen `surface-full-v1` round compares annotated-vocabulary PD-MC with
Surface-aware on **all 1,387 shared successful outputs** from the same
2,565-target population. Baseline saved 1,394 outputs; Surface saved 1,399.
There are 1,178 excluded targets: 1,159 fail in both arms, seven have only
baseline output, and twelve have only Surface output. Those pipeline outcomes
remain separate from preference scores. No cases were chosen using metrics,
visual quality, or prior judgements.

Study ID: `e1dd2c278789385727448373`.
Contract: [human-preference-20261122.json](../configs/research/human-preference-20261122.json).

Published private reviewer: https://amodal-preference-dv20261122.rompingsalmon2.chatgpt.site .
Select `output/human-preference-20261122/dist` with **Open study folder**.
The fully recovered offline packet is ready through `Start review.cmd`.
All 6,935 image hashes and geometry checks passed, and desktop/mobile browser
QA passed including the hosted folder-required flow. Deployment evidence is
in `output/human-preference-20261122/deployment.json`.

Sources are the registered full `baseline` and `surface` runs under
`output/surface-annotated-20261112-freshrefs/`. This uses the actual annotated
vocabulary control and corrected progressive Surface outputs; it is not the
controlled artificial-occlusion candidate study.

## Rating

Compare each completion with the input scene and visible target. Select the
preferred output overall, judging plausible complete shape, coherent surface
appearance, and preservation of visible identity. Multiple selected options
mean a tie for best. “All equal” selects every option. “Cannot judge” is distinct
from a tie. Skipping requires a reason and remains revisitable.

Overall judgement is required to save a rated case. Shape and surface
preferences, confidence, and notes are optional. Optional answers have their
own denominators. Rater code, research involvement, and previous exposure
accompany exports. As the method's researcher, the user should record that
involvement; this cannot be reported as an independent multi-rater study.

Method labels, metric scores, original target IDs, and the assignment key are
absent from the rating application. The A/B permutation varies by case, with
each method appearing in each position 693 or 694 times. Cases are ordered by
SHA-256 with seed 20261122. The permutation schedule balances positions; it
does not establish psychological blindness to recognizable outputs.

The source scene, SAM visible reference, and both returned RGBA completions
share a crop covering the union of their alpha support in original image
coordinates, including negative offsets and completion outside the source.
Each panel has the same native dimensions and scale. Lossless WebP preserves
decoded pixels; no resizing, sharpening, or independent object normalization
is applied. The complete source image and native-pixel inspection are available.
SAM visible support is a target-identification aid, not hidden-shape truth.

## Save and resume

The website stores responses in this browser, under the exact study ID.
Export a JSON backup after each session or batch of 20. JSON restores progress
on another browser; CSV is a readable export. Browser storage is not a portable
backup. Private-browsing restrictions or storage failures produce a warning.
Concurrent-tab conflicts stop storage overwrites.

The hosted website requires opening the extracted local study folder; those
images stay on the device and are SHA-256 checked when opened. Select the folder
again after reloading the site; browser-saved ratings remain intact. A folder must
contain `study.json` and the complete `assets/` directory. Directly opening
the offline packet's `index.html` uses relative assets without a server.
The local packet's `Start review.cmd` opens `dist/index.html`.

The hosted `out/` directory contains only the reviewer UI and frozen study/hash
manifests. The complete native-image archive exceeded Sites' 512 MiB upload cap,
and its remote build failed with a hosting-service error. All native images
remain in the verified offline packet. The compact site waits for folder
selection before enabling ratings or skips, without requesting absent images.

Imports validate the study, round, case IDs, option IDs, timestamps, required
judgements, duplicates, and completion status before replacing data, and ask
for confirmation. A different round's backup is rejected. No ratings are sent
automatically to a server.

## Add human results beside the defined metrics

Keep the returned human JSON unchanged and use a new output directory:

```powershell
python scripts/summarize_preference_study.py --packet-dir output/human-preference-20261122 --response PATH/returned.json --output-dir output/human-preference-20261122/responses/R01-v1
```

The importer produces `original-response.json`, `summary.json`,
`human-preference.csv`, `per-case-preferences.csv`, and `REPORT.md`.
Incomplete assignments fail by default; `--allow-partial` explicitly produces
an interim report. Drafts, skips, inability to judge, and unanswered optional
criteria never become losing votes or zero-quality scores.

The main human column is **overall preference share (%)**, splitting a tied
top vote equally across selected methods. Report unique wins, tied-top counts,
and the assessable denominator alongside it. Keep shape/surface preferences
secondary. With three or more methods, two nonselected options have unknown
relative order; the importer does not invent a tie between them.

Add this supplement beside LPIPS, SSIM, CLIP, feature similarity, and geometry
metrics. Do not average incompatible measures into a composite score. Report
the actual completed scope and single-rater involvement. Conditional preference
does not establish an all-target improvement, hidden RGB accuracy, independent
consensus, or inter-rater reliability. Repeated objects from a source image can
be dependent; no population confidence interval or inter-rater claim is made.

## Later rounds

Do not change the current packet or pool early two-method votes with a later
three-method choice task. After Shape's **full 2,565-target** run is completed,
verified, and registered, clone the contract to a new filename, choose a new
round ID, and add the full Shape run as a third method. Build to a new packet
directory. The builder rejects incomplete executions, mismatched populations,
changed source/completion hashes, and non-annotated vocabulary. Preserve the
new shared-success denominator, hardware/protocol differences, and all previous
responses. The UI and importer already support two to four methods.

The combined Surface + Shape method belongs to a later separately frozen round.
No Shape/full or combined ratings are fabricated from current data. No automatic
future job, new inference, retuning, or recurring monitoring was scheduled.

## Build and validate

Use the cluster project root for access to the native full outputs. The Python
builder requires only Pillow and standard-library modules and does not load GPU
models or change the pinned inference environment.

```bash
python scripts/prepare_preference_study.py --config configs/research/human-preference-20261122.json --output-dir output/NEW-preference-packet
python scripts/install_preference_ui.py --packet-dir output/NEW-preference-packet
python scripts/validate_preference_study.py --packet-dir output/NEW-preference-packet --receipt output/NEW-preference-packet/validation.json
python scripts/package_preference_study.py --packet-dir output/NEW-preference-packet --archive output/NEW-preference-packet/reviewer.zip
```

Keep `COORDINATOR_ONLY/` out of the website and reviewer ZIP. It contains target
and method mappings, source-run provenance, artifact hashes, and excluded cases.
Only `dist/` is public to the reviewer. Preparation does not generate any ratings.

```powershell
python -m unittest discover -s tests -p test_preference_study.py -v
node --test tests/test_preference_core.cjs
node --check sites/amodal-preference/dist/app.js
```

The optional dependency-free browser smoke uses headless Edge and an isolated
three-case QA packet/profile. It checks desktop/mobile layout, rating/disclosure,
ties, skips, persistence, export/import, folder mounting, and wrong-round
rejection. Its `TEST_ONLY` records use a separate `qa-` study ID and are not
human responses. They must never be included in final research results.
