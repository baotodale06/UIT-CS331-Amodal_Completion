# Offline regression fixtures

The four `official/` JSON files are byte-for-byte copies of the authors' released annotations. Their SHA-256 values, counts, and public download IDs are pinned in `amodal_benchmark/dataset.py` and `prepare_open_world.py`. They contain metadata, not downloaded images.

`pdmc_main.py` is the unchanged upstream reference from [k8xu/amodal, commit bfbfcdec581484586774d9ca53f5c603c97c725b](https://github.com/k8xu/amodal/blob/bfbfcdec581484586774d9ca53f5c603c97c725b/main.py), already archived by this project. Its checksum is `eed6b9eefe6c8aff9ad3d14046c5df8c04b5bc79cfd8ef586e97e51456331c8e`. Tests parse selected functions from its syntax tree and replace expensive model boundaries; they do not import its GPU dependencies. Preserve upstream attribution and bytes.

All test images are generated in temporary directories. Two optional historical integration tests use the original release ZIP or saved paper reports when installed; they skip when those artifacts are absent. The CPU smoke command does not depend on these fixtures or any downloaded data.
