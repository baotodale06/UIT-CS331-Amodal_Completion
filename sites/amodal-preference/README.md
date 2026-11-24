# Amodal Preference Lab

Buildless, private rating website for the full annotated-vocabulary PD-MC versus
Surface-aware study. Shape-aware and the combined method are later rounds.

The scientific contract, data preparation, response importer, and interpretation
are documented in [HUMAN_PREFERENCE_STUDY.md](../../docs/HUMAN_PREFERENCE_STUDY.md).
No human ratings are bundled in the website.

The deployed directory is `out/`, prepared by `scripts/prepare_preference_hosted.py`.
Open the private site and select the verified offline packet's `dist/` folder.
The hosted interface waits for that folder before enabling ratings; images stay
on the device and are checked against the frozen hashes. The full native-image
deployment exceeded the hosting upload limit and its remote build failed.
The offline `dist/` reviewer opens directly without a server or folder selection.
All full-resolution images and the original full source remain preserved.

Do not copy `COORDINATOR_ONLY/`, returned responses, or QA fixtures into this
Site repository. Preserve `.openai/hosting.json` and the private audience.
