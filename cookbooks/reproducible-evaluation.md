# Reproducible amodal evaluations

Choose the reference protocol before interpreting scores. The SAM pipeline compares completions against its own initial detections. The LISA pipeline uses independently generated visible references. Neither supplies hidden-pixel ground truth, and their appearance scores should not be pooled.

1. Validate official annotations and original image bytes before model loading.
2. Declare the experiment in an inherited JSON config and record a hypothesis.
3. Run the synthetic CPU smoke, then a separate two-target GPU smoke.
4. Use a new output directory for each changed scientific setting. Reuse the original model cache and, for controlled LISA experiments, fixed references.
5. Resume interrupted runs with the same scientific settings. Worker count can change because each image and target has a stable seed.
6. Verify completions, references, metric caches, summaries and provenance with `amodal verify`. Register full evaluations explicitly with `amodal register`.
7. Keep failed attempts and negative findings. A saved output at the iteration limit is distinct from a completed object; inspect both counts.

CLIP is reported for successful completions only. The four metric denominators can differ. A lower failure rate can change the population contributing CLIP, so a CLIP mean alone cannot establish improvement. Failed completions receive the documented appearance penalties. Empty generated references remain explicit.

For a one-factor comparison, pass `--baseline-dir` when registering an experiment with `experiment.changed_factor`. Both runs must be verified on the same input bytes and target identities. LISA comparisons reject changed reference bytes; SAM comparisons record when their generated reference bytes differ. The ledger never automatically ranks incompatible protocols.

Sources: [PD-MC implementation](https://github.com/k8xu/amodal), [Open-World benchmark](https://github.com/saraao/amodal), and this project's [protocol details](../OPEN_WORLD_REPRODUCTION.md).
