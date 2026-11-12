# Independent human review of visual outputs

Prepared 2026-11-08. **No human ratings have been collected.** The existing appearance reports contain assistant judgments. This protocol prepares one independent human to evaluate the expanded-vocabulary run; it does not claim that the requirement has already been satisfied.

## Which packet to use

The ready-to-share packets are under [output/human-review-20261108](../output/human-review-20261108/). Send only the reviewer ZIP for the agreed assignment. Keep each `COORDINATOR_ONLY` folder, the existing report and prior annotations away from the reviewer until they submit their ratings.

- **Initial audit:** `audit-160/human-review-160-cases.zip`, 160 random saved outputs, eight batches of 20, roughly 3–6 hours plus breaks. This is the prepared default because a full census is a substantial assignment for one person. The user may choose the full census instead.
- **Full census:** `census-1390/human-review-1390-cases.zip`, every saved output, 70 batches (the last has ten), roughly 23–46 hours plus breaks.

Choose the scope before viewing human responses. The audit is the first 160 cases in the same fixed random order as the census. Do not ask the person to rate both as independent datasets. The packets have different IDs and deliberately cannot import one another's backups. If the scope must change after work begins, preserve the first response and arrange an explicitly documented continuation; do not discard or silently relabel it.

The reviewer extracts the entire ZIP and opens `index.html`. No Python, account, installation, server or internet connection is needed. The ZIP includes the guide, neutral overview panels, full original source images, native visible-target PNGs, returned completion PNGs and raw generated JPEGs. Native files are byte-identical to the recorded sources. The overview is a recompressed cropped preview with identifying/metric header rows removed. No earlier labels, evidence notes, scores, category frequency tables or report conclusions are included.

## Recruitment and independence

Recruit one person who did not develop the method, choose the analyzed outputs or produce the earlier annotations, and who has not read the case ratings or conclusions. Give only the rubric and reviewer packet before submission. Explain the task using the hypothetical examples in the guide; do not train the person to reproduce the assistant's answers.

The form records a reviewer code, prior involvement, exposure to prior results, a disclosure note if needed, and an independent-work declaration. These are self-reports, not proof of independence. Resolve any disclosure before calling the reviewer independent in the report. Store identifying information separately if needed; a pseudonymous code is sufficient in the packet.

No messages or invitations have been sent. Suggested invitation:

> Please independently review the visual outputs in this packet using its guide. Extract the ZIP, open index.html and work in the assigned order. Do not consult our report, prior scores/ratings, other people or AI while judging cases. Tell me before starting if you helped develop/select/label these outputs or have seen the earlier findings. Use U when evidence is unclear. Export a JSON backup after each batch, and return the final JSON and CSV. Please ask about unclear rubric wording; I will avoid suggesting ratings for individual cases.

## Frozen population and selection

The population is the 1,390 saved outputs from the historical `pdmc-open-world-sam-annotated` run, recovered under `output/artifact-recovery-2026-11-05/downloads/`. It has 2,565 targets; 1,175 have no visual output and remain separate pipeline failures. They receive no invented visual rating.

Selection orders every saved target by SHA-256 of `appearance-human-v1:20261108:<target-id>`, with target ID as tie breaker. The first 160 form a fixed, approximately simple random sample without replacement; each output has inclusion probability 160/1,390. There is no selection by score, appearance label, severity, category, occlusion or dispute. The full census has inclusion probability one. Ordering, target mapping, artifact hashes, protocol and delivery receipts are retained privately.

The 160-case audit is suitable for a manageable independent check of overall appearance judgments. It is **not powered or balanced to validate rare failure types, every category, or each true-occlusion bin**. Do not replace random cases with dramatic examples, present the sample as a 1,390-output human census, or treat its frequencies as the exact full-census counts. Outputs sharing a source image may be dependent. This preparation makes no power or confidence-interval claim.

## Rubric and response collection

The six criteria are texture, color, pattern, semantic identity, newly generated-region consistency and added secondary objects. Each uses severity 0–3 or U; pattern alone also allows NA. Unknown and NA are separate from negative judgments. All six answers, an evidence note and an explicit confound choice are required before a case is marked complete. Positive added-object labels also require a subtype and location relative to the returned mask.

The full source is available for every case because pre-existing context can otherwise be mistaken for generated objects. Mask errors, target switches, lost visible support, low resolution, incomplete geometry and ambiguous residue are recorded separately. There is no hidden RGB ground truth. The map's orange area is predicted expansion, not true occlusion.

The form saves drafts locally and exports JSON/CSV. Require a JSON export after each batch; browser local storage is not a portable backup. Import validates packet identity and completion fields and requires confirmation before replacing local state. Partial work remains explicitly incomplete. The blank CSV is an emergency manual alternative; it must be checked and transcribed into the JSON schema with reviewer disclosures before automated import. Preserve that original CSV and document the transcription.

## When responses return

Keep the original response unchanged. Use a new output directory for every submitted version:

```powershell
python scripts/summarize_human_review.py --packet-dir output/human-review-20261108/audit-160 --response PATH/returned-review.json --output-dir output/human-review-20261108/received-R01-v1
```

The default rejects incomplete assignments, wrong packet IDs, duplicates, missing evidence and invalid labels. `--allow-partial` permits an explicitly marked interim summary; unanswered/draft cases are excluded rather than assigned zero. Do not present an incomplete or selectively completed subset as the full random sample.

The importer writes separate human annotations, per-criterion positive/assessable counts, severity and U/NA counts, original response content, provenance receipts and a disagreement table. Human–assistant binary agreement, Cohen's kappa and exact severity agreement use only jointly assessable cases; undefined kappa is null. These compare **one human with one assistant**, not two human raters. Rare labels and prevalence imbalance can make kappa unstable. No multi-human reliability or consensus claim is possible.

Freeze the independent human submission before sharing assistant labels or discussing disagreements. Any later adjudication must be a separate record with reasons; keep both original judgments. Do not automatically overwrite the existing assistant census or its report. After collection, add the human review as a separately identified audit/supplement with its actual scope and limitations.

Suggested report wording to fill only after collection:

> One independent human rater, [relevant background], reviewed [n] randomly selected saved outputs / all 1,390 saved outputs, with prior assistant labels and metric scores hidden. The rater used six appearance criteria with severity 0–3, U for uncertainty and NA for inapplicable patterns. [n_completed]/[n_assigned] cases were completed. We report criterion-specific positive/assessable counts and retain unknown, inapplicable and pipeline-failure counts separately. Human–assistant agreement is descriptive; a single human rater does not establish inter-human reliability or hidden-appearance accuracy.

## Rebuild and validate

Python with Pillow and the existing project CPU dependencies suffices. Do not modify the pinned GPU environment.

```powershell
python scripts/prepare_human_review.py --analysis-dir output/appearance-expanded-20261107 --output-dir output/NEW-human-audit --count 160
python scripts/prepare_human_review.py --analysis-dir output/appearance-expanded-20261107 --output-dir output/NEW-human-census --count 0
python scripts/validate_human_review.py --packet-dir output/NEW-human-audit
python -m unittest discover -s tests -p test_human_review.py -v
```

The builder refuses to overwrite a packet. The validator checks coverage, allowed public fields/files, blank score cells, native evidence hashes, image decoding and exact ZIP-member hashes. Its success does not validate any human judgment.
