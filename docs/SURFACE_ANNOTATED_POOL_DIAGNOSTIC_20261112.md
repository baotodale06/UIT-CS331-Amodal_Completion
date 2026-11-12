# Pre-specified annotated final-pool diagnostic

Added 12 November 2026 before annotated Surface-PMC candidate generation or
metric results. This is a secondary mechanism analysis alongside the frozen
204-target progressive comparison; it changes no primary decisions or outputs.

Use all eight saved raw candidates from the last progressive pass of every
successful Surface-PMC cohort target. Resize with the original bicubic/native
JPEG procedure, and attach the corrected final mask. Keep the final geometry,
mask, success/failure status, and every preceding trajectory fixed. Evaluate
the same visible SAM reference and the same four benchmark metrics. Rescore
both original control and corrected output in the same four-GPU job, requiring
absolute cache replay error ≤1e-5. Retain all 204 targets and original penalties.

Compare the frozen raw choice with (a) uniform expectation across all eight
raw images, (b) uniform expectation among originally valid candidates, falling
back to candidate zero if none are valid, (c) candidate zero/seed-slot 11, and
(d) highest category-text cosine among valid candidates, with the same fallback.
Compare corrected output with the selected raw image to isolate the final RGB
correction at fixed masks/trajectory. Report all metrics and 5,000-resample,
source-stratified image-cluster paired intervals. Treat all diagnostic intervals
as nominal/exploratory. The primary 204-target progressive LPIPS test is unchanged.

Compute a separate reference-informed best raw choice for each metric as a
headroom diagnostic. These oracle choices use evaluation references, differ
between metrics, are not deployable selectors, and do not form a jointly
attainable result. No new numerical target is introduced. Report each ceiling
and tradeoffs without tuning against these outputs.

This is a shared saved-pool comparison at the final step. It does not measure
an equal-budget progressive random trajectory, and cannot isolate changes to
candidate masks or earlier passes. Its fixed corrected masks can favor the
selected trajectory. No hidden RGB, human preference, new diffusion generation,
training, or independent confirmation is claimed.
