# Documentation index

The [24 November 2026 results index](../results/README.md) links the published
full-benchmark score tables, reports, figures, and verification receipts, and
separates them from unfinished experiments and validation diagnostics.

Start with the root [README](../README.md) for current commands. The detailed setup and protocol guides stay at their established paths for existing users and cluster uploads.

| Document | Purpose |
| --- | --- |
| [Codebase map](CODEBASE.md) | Entry points, module responsibilities, test locations and file retention |
| [Completed annotated full benchmark, 2026-11-14](SURFACE_ANNOTATED_FULL_RESULTS_20261114.md) | Verified/registered 2,565-target comparison: LPIPS +1.191%, frozen unseen +0.801%, 4.733× allocation cost; original primary cohort remains inconclusive |
| [Scientific research briefing, 2026-11-11](../output/research-briefing-2026-11-11/briefing.md) | Revised full project review: 75 slides, 57 equations, derivations, fitted parameters and statistical evidence; [PDF](../output/research-briefing-2026-11-11/presentation.pdf), [equation index](../output/research-briefing-2026-11-11/equations.md) |
| [Latest team manager briefing, 2026-11-11](../output/manager-briefing-2026-11-11/briefing.md) | 40-slide review and detailed notes through completed diagnostics: baseline, audits, controlled confirmation, full-target shortfall and next decisions; [PDF](../output/manager-briefing-2026-11-11/presentation.pdf) |
| [Historical team manager briefing, 2026-11-05](../output/manager-briefing-2026-11-05/briefing.md) | Preserved 16-slide snapshot before artifact recovery and the new studies; [PDF](../output/manager-briefing-2026-11-05/presentation.pdf) |
| [Project setup](../SETUP_NEW_COMPETITION.md) | CPU installation and the research workflow |
| [Evaluation protocol](../OPEN_WORLD_REPRODUCTION.md) | Dataset identity, reference protocols, evaluation and historical findings |
| [Cluster setup](../AMODAL_SLURM.md) | Pinned GPU environment and Slurm usage |
| [PD-MC paper summary](PAPER_SUMMARY.md) | Explanation of the completion method |
| [Surface-Consistent PMC research reference](SURFACE_CONSISTENT_PMC.md) | Retained proposal: appearance-based candidate selection, formulas, source extract, evaluation and open questions |
| [Surface consistency experiment](SURFACE_CONSISTENCY_EXPERIMENT.md) | Implemented observable-only scorer, theory limits, controlled hidden-pixel experiment and prespecified replication |
| [Surface consistency results](SURFACE_CONSISTENCY_RESULTS.md) | Completed 159-test-image study: full rule inconclusive, exploratory local benefit, exact replay, explanations and report |
| [Annotated-vocabulary Surface-PMC results](SURFACE_ANNOTATED_RESULTS_20261112.md) | Frozen 204-target transfer: primary LPIPS inconclusive, secondary CLIP gain, fixed-pool diagnostic, full-run continuation |
| [Annotated transfer mathematical methods](SURFACE_ANNOTATED_METHODS_20261112.md) | Exact selector, historical ridge training, corrections, metric operations and paired uncertainty |
| [Surface consistency derivations](SURFACE_CONSISTENCY_THEORY.md) | Conditional risk, identifiability bound, filtering/ranking decomposition and assumptions |
| [Surface experiment cookbook](../cookbooks/surface-consistency.md) | Preparation, four-GPU candidate pools, frozen development choices, evaluation and replay commands |
| [Paper-to-code map](PAPER_CODE_MAP.md) | Paper components mapped to local functions and pinned upstream source |
| [Test guide](../tests/README.md) | Offline regressions, fixtures and separate GPU checks |
| [Evaluation cookbook](../cookbooks/reproducible-evaluation.md) | Reproducible experiment procedure |
| [Recovery cookbook](../cookbooks/cluster-recovery.md) | Transfers, failed jobs and resuming evaluations |
| [Archived plans](archive/README.md) | Completed implementation records retained for context |

The root `plan.md`, `progress.md`, `discoveries.md`, `closed_ideas.md` and `session.md` remain the live local research record. They are not part of the source upload. Downloaded upstream source under `datasets/open_world_amodal/references/` and scripts stored with historical outputs remain with their provenance.
