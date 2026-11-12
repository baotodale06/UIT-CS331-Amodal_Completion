# Repository setup findings

- GitHub already contains the upstream release and two PMC proposal documents;
  the local benchmark project originally had no Git metadata. Importing onto a
  branch preserves the existing history.
- Root `requirements.txt` is the original GPU-era dependency list. CPU
  development uses `requirements/cpu.lock`; the GPU runner provisions its own
  environment. Installing one over the other breaks reproducibility.
- Root `main.py` is retained for reading/development. Benchmark inference uses
  the explicitly pinned cache checkout, so model edits need a deliberate source
  identity and experiment change.
- Exact fixture and frozen research bytes matter. Git attributes preserve those
  files while enforcing Linux line endings for launchers.
- A Git clone supports offline development checks. Historical experiment replay
  additionally needs original data, frozen parameters, saved pools/features and
  the recorded native runtime; these artifacts are not in Git.
- Python version-manager symlinks can produce an incorrect venv stdlib path.
  The installer resolves the base interpreter before creating the environment.
