# Local output cleanup review — 16 November 2026

The initial review below proposed cleanup without deleting anything. After the user approved removal of failed transfers, **group B was completed: 10 files totaling 396,648,697 bytes (378.27 MiB) were removed**. Groups A and C remain proposals and were not removed. The cluster was not accessed.

The [cleanup receipt](OUTPUT_CLEANUP_RECEIPT_20261116.json) records the exact deleted paths, sizes, SHA-256 hashes and execution timestamps. All ten paths were confirmed absent afterward, and the required completed archive, recovered result directories and recovery receipts were confirmed present. Parent directories, logs and other artifacts were retained.

The scan found 201,433 regular files under `output/`, totaling 29.761 GiB by path. Accounting for shared NTFS hard links gives 28.498 GiB of distinct file data. These are file-content sizes, not exact allocated filesystem space. One inaccessible WSL `validation-env/lib64` entry was excluded. C: had about 21.6 GiB free at the initial check.

The original candidates below totaled **4,306,280,107 bytes = 4.011 GiB**; after group B, **3,909,631,410 bytes = 3.641 GiB** remain proposed in A and C. None of the selected regular files is hard-linked. Sizes exclude filesystem overhead. Keep the extracted contents and all verification/provenance receipts when removing an archive.

## A. Duplicate packages and transfer archives: 3.542 GiB

Every regular member of these six archives was compared with its existing extracted counterpart using SHA-256 during this review. All matched; no missing or differing members were found.

| Archive, relative to `output/` | Size (MiB) | Matching files | Extracted contents to retain |
| --- | ---: | ---: | --- |
| `human-review-20261108/census-1390/human-review-1390-cases.zip` | 1,831.73 | 6,955 | `human-review-20261108/census-1390/FOR_RATER/` and `COORDINATOR_ONLY/` |
| `human-review-20261108/audit-160/human-review-160-cases.zip` | 216.36 | 805 | `human-review-20261108/audit-160/FOR_RATER/` and `COORDINATOR_ONLY/` |
| `surface-annotated-20261112-recovery/surface-annotated-20261112-freshrefs-cohort-native/evidence.tar.gz` | 985.62 | 5,577 | `surface-annotated-20261112-freshrefs/baseline-cohort/` and `surface-cohort/` |
| `surface-followup-20261110-design/fresh-four-results.tar.gz` | 215.96 | 2,000 | `surface-followup-20261110/full/` |
| `surface-followup-20261110-design/fresh-extra-results.tar.gz` | 215.73 | 2,000 | `surface-followup-20261110-design/extra-pools-166/full/` |
| `appearance-expanded-20261107/appearance-expanded-delivery.zip` | 162.08 | 1,562 | The remaining contents of `appearance-expanded-20261107/` |

Tradeoff: removing these loses the ready-made shareable/restore bundles and an extra local copy. The extracted reviews and research evidence remain usable. ZIPs would need rebuilding before sharing them again; a rebuilt archive may have a different archive checksum. For human review, package only `FOR_RATER/`, never private `COORDINATOR_ONLY/` material.

## B. Completed removal: failed transfers and redundant transfer chunks, 378.27 MiB

The following named files were removed with the user's approval; their parent directories were retained:

- `baseline-establishment-20261105/accepted-baseline-20261106T092725Z.tar.gz` — 243.53 MiB. This archive is truncated (`unexpected end of data`). All 3,588 complete readable members match `baseline-31548/` by SHA-256. The existing `accepted-recovery-receipt.json` records the completed independently verified recovery. Keep `baseline-31548/` and that receipt.
- `artifact-recovery-2026-11-05/transfers/pdmc-open-world-sam-annotated.tar.gz.partial` — 23.00 MiB.
- `artifact-recovery-2026-11-05/transfers/pdmc-open-world-sam-semantic-v2.tar.gz.partial` — 21.00 MiB. The recovery report explicitly identifies these stopped archives as transport diagnostics; completed recovered artifacts are under `downloads/`.
- `surface-followup-20261110-design/fresh-controlled-evidence-legacy.tar.gz.partial` — 11.50 MiB.
- `surface-followup-20261110-design/fresh-controlled-evidence.tar.gz.partial` — 16.91 MiB. The complete archive and successful download receipt are present.
- The four `surface-followup-20261110-design/evidence-chunks/fresh-controlled-evidence.part-aa`, `.part-ab`, `.part-ac`, and `.part-ad` files — 62.34 MiB combined. Concatenating them produces exactly the SHA-256 of the retained complete `fresh-controlled-evidence.tar.gz`.
- `surface-annotated-20261112-freshrefs/baseline-cohort/visible_references/VG/tmpxhvjs4ir.partial` — zero bytes. This adds no meaningful saving.

Tradeoff: removing these discards failed transport fragments and duplicate transfer chunks, not the completed recovered results. Keep the small logs and receipts.

## C. Rebuildable validation environments: 101.05 MiB

- `appearance-failures-20261107/validation-env/` — 89.00 MiB of inventoried regular files; an isolated WSL Python 3.14 environment.
- `appearance-expanded-20261107/validation-env/` — 12.05 MiB; an isolated Windows Python 3.14 environment.

These are CPU validation installations excluded from the delivery packages. Their removal would require recreating the environments before using those exact paths for validation. Keep the reports, reproducibility source snapshots, dependency/runtime evidence, and validation logs. Confirm no process is using them when deletion is actually requested. The inaccessible WSL `lib64` entry requires link-aware handling; do not follow it recursively.

## Keep

- Actual run outputs, native images/candidate pools, references, metrics, source/configuration freezes, provenance, registry entries, reports, annotations, and verification receipts. An older experiment is not automatically unnecessary.
- `artifact-recovery-2026-11-05/downloads/` (11.09 GiB by path) and `baseline-31548/` (2.76 GiB): these contain recovered research evidence, not disposable download caches.
- `artifact-recovery-2026-11-05/legacy-verification/`: it appears to hold about 1.16 GiB, but almost all of that is already shared through hard links. Deleting this view would save very little while removing a useful verification layout.
- The COCOA annotation JSON copies in the two appearance folders also share hard links; deleting one set would not reclaim their listed 108.93 MiB.
- `surface-followup-20261110-design/fresh-controlled-evidence.tar.gz`: 1,193 members matched, but the archived `report/gallery/index.html` differs from the current file. It is not a fully redundant snapshot. Its duplicate transfer chunks are covered by B.
- `appearance-failures-20261107/external/cocoa_annotations.tar.gz`: despite its extension, it is an uncompressed tar containing six files, including BSDS annotations beyond the three extracted COCOA JSON files. It was not established to be wholly redundant.
- Human-review coordinator records and extracted reviewer packets. No human ratings are present in the documented handoff; deleting a packet would be a separate decision about future review work.

Group B is complete. A remains optional if ready-made archive bundles are unnecessary, and C remains optional if rebuilding the old CPU validation environments is acceptable. No additional deletion is authorized by this report or the user's approval for B.
