# UIT CS331: reproducible amodal completion

This repository retains the original PD-MC implementation and adds the team's
Open-World benchmark, explicit experiment configs, regression tests, provenance
verification and Surface-PMC research tooling.

**Start here: [collaborator setup guide](docs/GETTING_STARTED.md).**
The supported development environment is Python 3.12 on Linux/WSL; CPU
installation and checks do not require CUDA, datasets or model downloads.

```bash
git clone --branch setup/reproducible-development \
  https://github.com/baotodale06/UIT-CS331-Amodal_Completion.git
cd UIT-CS331-Amodal_Completion
python3.12 scripts/setup_cpu.py
source .venv-cpu/bin/activate
bash scripts/check.sh
```

The installer verifies exact dependency hashes and uses a dedicated `.venv-cpu`.
GitHub Actions runs the same offline suite and package build. Two optional tests
need historical artifacts and skip in a fresh clone. For a personal branch:

```bash
git switch -c feature/my-change
```

| Next step | Guide |
| --- | --- |
| Contribute and open a PR | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Dataset, GPU provisioning and first experiment | [GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Experiment configuration and verification | [BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md) |
| GPU runtime details | [AMODAL_SLURM.md](AMODAL_SLURM.md) |
| Evaluation protocol and limits | [OPEN_WORLD_REPRODUCTION.md](OPEN_WORLD_REPRODUCTION.md) |
| Dependency locks | [requirements/README.md](requirements/README.md) |
| Research notes, code map and original proposals | [docs/README.md](docs/README.md) |

Datasets, checkpoints, caches and generated research artifacts are external to
Git. The CPU environment is separate from the pinned Python 3.10 / PyTorch 1.13.1 /
CUDA 11.7 GPU environment provisioned by `run_amodal_slurm.sh`. Keep the two
separate. Root `requirements.txt` and `install.sh` below belong to the original
release; use the collaborator guide for the maintained benchmark workflow.
The runner uses a pinned model checkout under its cache, so editing root
`main.py` alone does not change benchmark inference.

## Original upstream README

The original authors' setup, usage and citation follow for reference.

## Amodal Completion via Progressive Mixed Context Diffusion (CVPR 2024)
### [Project Page](https://k8xu.github.io/amodal/) | [Paper](https://arxiv.org/pdf/2312.15540.pdf) | [arXiv](https://arxiv.org/abs/2312.15540) | [Bibtex](#bibtex)

[Katherine Xu](https://k8xu.github.io)$^{1}$, [Lingzhi Zhang](https://owenzlz.github.io)$^{2}$, [Jianbo Shi](https://www.cis.upenn.edu/~jshi)$^1$<br>
$^1$ University of Pennsylvania, $^2$ Adobe Inc.

![teaser](images/teaser.png)
Our method can recover the hidden pixels of objects in diverse images. Occluders may be co-occurring (a person on a surfboard), accidental (a cat in front of a microwave), the image boundary (giraffe), or a combination of these scenarios.
The pink outline indicates an occluder object.

We use pretrained diffusion inpainting models, and no additional training is required!


## 🚀 Updates
- Stay tuned for our code release!


## Table of Contents
* [Requirements](#requirements)
* [Setup](#setup)
* [Citation](#citation)


## Requirements
* Python 3.10
* Docker


## Setup

1. Clone this `amodal` repository, and run `cd Grounded-Segment-Anything`.

2. In the Dockerfile, change all instances of `/home/appuser` to your path for the `amodal` repository.

3. Run `make build-image`.

4. Start and attach to a docker container from the image `gsa:v0`. Then, navigate to the `amodal` repository.

5. Run `./install.sh` to finish setup and download model checkpoints.


## Dataset

1. Run `./download_dataset.sh` to download the COCO dataset.


## Usage

### Progressive Occlusion-aware Completion Pipeline

1. In `./main.sh`, modify `input_dir` to your folder path for the images.

2. Run `./main.sh`. You may need to use `chmod` to change the file permissions first.


## Citation

If you find our work useful, please cite our paper:
```
@inproceedings{xu2024amodal,
  title={Amodal completion via progressive mixed context diffusion},
  author={Xu, Katherine and Zhang, Lingzhi and Shi, Jianbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9099--9109},
  year={2024}
}
```
