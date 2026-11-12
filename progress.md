# Repository setup progress

2026-11-12: created `setup/reproducible-development` from GitHub main
`14834b65a09878b20fe32267919a47f16e784c7d`. Added maintained benchmark/research
source, configs, fixtures and checks alongside the original PD-MC release.

Added hashed CPU/build dependency locks, a Python 3.12 installer, GitHub Actions,
collaborator instructions and experiment handoff documentation. Kept the GPU
runtime separate and retained original dependency/install files.

Fresh Linux CPU installation, Python regressions, all five Bash regression
scripts, synthetic smoke, configuration resolution and package builds passed.
The Python suite ran 271 tests with two expected historical-artifact skips.
No new GPU experiment was run for repository setup.
