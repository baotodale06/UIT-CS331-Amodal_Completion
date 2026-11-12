# Dependency environments

| Environment | Installation source | Purpose |
| --- | --- | --- |
| CPU development, Python 3.12 | `build.lock` and `cpu.lock` | Editable package, complete offline regressions, SciPy/scikit-learn research tests |
| PD-MC GPU, Python 3.10 | `run_amodal_slurm.sh` | Pinned PyTorch 1.13.1 / CUDA 11.7 pipeline and models |
| Surface research metrics | `scripts/setup_surface_metrics.sh` | Additional metric environment based on the GPU runtime |
| Historical report generation, Python 3.14 | Root `requirements-analysis.txt` | Recorded reporting versions; separate from CPU development and GPU inference |
| Original release | Root `requirements.txt`, `install.sh` | Preserved upstream setup; not the supported benchmark environment |

`scripts/setup_cpu.py` installs all transitive dependencies from hashed locks
using binary wheels, then installs this repository editable with dependency
resolution and build isolation disabled. It checks installed dependency
consistency. Python 3.12.12 is the tested interpreter in `.python-version`.
The package metadata retains its wider Python compatibility; other versions
are outside the locked development setup.

The CPU environment is not a promise of bit-exact historical numerical replay.
Historical research freezes also pin their native runtime, inputs, features and
source hashes. GPU packages must never be upgraded from these CPU lock files.

Locks were generated with **uv 0.9.5**, targeting Python 3.12 across platforms.
uv is needed only to regenerate locks, not for a normal clone/install:

```bash
uv pip compile requirements/cpu.in --python-version 3.12 --universal \
  --generate-hashes --no-emit-index-url --output-file requirements/cpu.lock
uv pip compile requirements/build.in --python-version 3.12 --universal \
  --generate-hashes --no-emit-index-url --output-file requirements/build.lock
```

Edit the `.in` files deliberately, keep runtime ranges in `pyproject.toml`
compatible, review transitive changes, and create a new environment with
`python3.12 scripts/setup_cpu.py --venv .venv-cpu-next`. Activate it and run
`bash scripts/check.sh` and `python -m build --no-isolation` before committing.
Use `--upgrade` with uv only when intentionally refreshing locked dependencies.

Installation needs package-index access. Once installed, the offline checks
need no network, GPU, dataset, model downloads or cluster credentials.
