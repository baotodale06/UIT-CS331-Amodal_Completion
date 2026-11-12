# Active session: Surface-PMC on annotated vocabulary (12 November 2026)

User work window: 06:46-11:46 UTC. Frozen design: docs/SURFACE_ANNOTATED_20261112.md.
Primary: 192 image-disjoint source images / 204 targets, annotated K=1 control versus
unchanged trained corrected K=8 Surface-PMC. No refitting. Both arms use annotated
scene queries and identical canonical annotated reference bytes; candidate masks
retain target-only segmentation. Exclude all 477 prior controlled-study images;
one near-copy was screened out before generation. All failures remain included.

Local source/protocol freeze passed; 54 source/design files deployed with backup and
SHA-256 receipt. Both reference copies passed all 10,261 file hashes. Jobs
39473/39474 run the control/surface cohorts, 39475/39476 their full continuations,
39477 the unchanged surface run after a timeout, and 39478 full verification and
registration. Diagnostic 39479 runs after the surface cohort, before its full
continuation. Exact commands/dependencies are in the design submission receipts.
Preserve all previous runs and keep AGENTS.md local.

07:13 UTC: control cohort has 89/204 target records with no worker tracebacks.
GPU CUDA-extension, FFT and pipeline-import checks passed. All 268 final CPU
tests, five shell regressions and synthetic smoke passed without skips. Additional
cohort/diagnostic focused checks passed. No paired quality result exists yet.

The cohort will be independently snapshotted and compared before unchanged full
2,565-target continuation. Full runtime is expected to exceed this work window;
leave valid jobs running and report their IDs/status. Do not reinterpret partial
results as full results or tune against the frozen cohort.

---

# Current session handoff

Updated 2026-11-11. Research runs complete; scientific briefing revision delivered.

## Scientific revision after the user's feedback (11 November)

The user found the manager deck insufficiently scientific and asked where the
formulas were. The corrected [research presentation](output/research-briefing-2026-11-11/presentation.pdf)
retains the complete project history and integrates 35 scientific slides into
the main narrative: **75 slides, 57 rendered equations**, and approximately
19,300 words of detailed explanation. [Offline HTML](output/research-briefing-2026-11-11/presentation.html),
[notes](output/research-briefing-2026-11-11/briefing.md),
[equation index](output/research-briefing-2026-11-11/equations.md), and
[exact ranker parameters](output/research-briefing-2026-11-11/ranker-parameters.csv)
are included in the portable package.

Coverage includes formal notation, PMC context mixing, surface and local losses,
validity/normalization, conditional-risk and identifiability arguments, exact
gain/regret decompositions, all 22 fitted features and coefficients, ridge target
and derivation, affine/consensus corrections, cohort lineage, metric protocols,
paired bootstrap, practical thresholds, real gains/losses, compute and the saved-pool
oracle bound. Every equation has explanatory text and evidence links. Original
proposal, implementation, conditional theory and measured evidence are distinguished.

Validation covers rendered mathematics, all slide bounds, navigation/Notes,
all PDF pages/equation labels, links, parameter equality, scoped numeric claims,
source hashes and ZIP payloads. PDF formula pages and contact sheets were visually
reviewed. Both original manager briefings and 215 inventoried source/test/configuration/
helper/old-brief files are preserved. This is a reporting correction: no new
experiment, remote operation or neural-metric rerun; existing conclusions stand.

## Latest reporting task: updated manager briefing (11 November)

The user requested the same format as the 5 November manager brief. Delivered
[40-slide PDF](output/manager-briefing-2026-11-11/presentation.pdf),
[offline presentation](output/manager-briefing-2026-11-11/presentation.html),
and [detailed briefing / speaker notes](output/manager-briefing-2026-11-11/briefing.md).
The notes contain approximately 10,300 words, displayed tables and evidence links.
The package includes the unchanged eight-run historical CSV, a 28-row scoped
results appendix, source hashes, and a portable ZIP.

The story covers engineering, dataset/protocols, historical recovery, accepted
baseline, both appearance audits, pending human ratings, initial surface study,
fitted follow-up, independent confirmation, full progressive results and final
diagnostics. It preserves the controlled confirmation's positive scope and the
unmet full 3% target. All research jobs remain complete; no new experiment,
neural-metric computation or remote operation was performed for this briefing.

Delivery checks cover every slide's bounds, all embedded images, navigation,
Notes mode, PDF page count/text, evidence links and archive hashes. The original
manager brief and 181 inventoried maintained-source/test/configuration/helper
and old-brief files remain unchanged. Read the package's validation.json and
delivery-receipt.json for the final checks and ZIP identity.

## Latest: final-RGB diagnostics completed and audited (11 November, 12:01 UTC)

Cluster access is restored. Jobs **39404** (7m28s) and **39407** (31s) completed
successfully. All 2,565 ablation records and 958 raw derivations were recovered
and independently audited. Raw-final versus matched control gains: CLIP **1.423%**,
LPIPS **0.596%**, VGG **2.121%**, SSIM **1.087%**. Against corrected output, raw-final
improves CLIP 0.293% but worsens LPIPS 0.240%, VGG 0.650%, SSIM 0.683%. Original
score replay drift is at most 1.79e-7. The full 3% objective remains unmet.

Submitted and completed frozen headroom diagnostic **39411** (23m36s): all eight
saved final candidates, 958 pools / 7,664 scored RGB images, with corrected final
masks, geometry, preceding trajectories and all failures fixed. No diffusion or
fitting. Individual reference-informed raw-pool ceilings versus control: CLIP
**2.817%**, LPIPS **0.740%**, VGG **3.257%**, SSIM **1.407%**. These use DIFFERENT
oracle choices and are not a combined attainable output or deployable selector.
Three ceilings below 3% rule out final raw RGB reranking alone on these saved
outputs. New generated pools, candidate masks and earlier trajectories are outside
these bounds. Do not launch an expensive raw full run based solely on controlled
gains or assume final correction caused the full shortfall.

Read [the combined report](output/surface-lastpass-review-20261111/REPORT.md),
[headroom audit](output/surface-lastpass-review-20261111/headroom-audit.json), or
[portable package](output/surface-lastpass-review-20261111/surface-followup-diagnostics.zip).
The ZIP is 569,403 bytes, all nine payload hashes/CRC pass, SHA256
6590b4eab32ae1fef70f844eb7235b4c50522015d6a8539ddcbefae25760454a.
The ablation audit verifies 2,577 recovered members and 7,047 original files;
headroom audit verifies 2,582 recovered members, 2,565 parent links, 7,664 metric
tuples, 3,832 oracle choices, every mean and paired image-cluster interval.
Native PNGs remain on the cluster; local neural metrics were not rerun.

Headroom freeze: output/surface-candidate-headroom-20261111-design/freeze.json,
SHA256 dc0f22bfdf1ff72e31f53400a520ffc303c3a55a353a6ce0a1cd81b732ac7cd7.
Exact command is in submission.json. Preflight verifies nine sources, all 2,565
parent records and native selected-RGB serialization. Both GPU integration checks
pass. All selected image hashes and metric scores reproduce exactly, with zero
score drift. Seven new focused audit/headroom regressions pass. Source freezes
and previous scientific implementations were preserved.

Both collectors finished, including headroom session82848; local audits8453/10560
also finished. No jobs need resuming and no inference jobs remain active. Do not
resubmit completed work. The original five-hour deadline was exceeded. The
independent 128-image raw-selector confirmation remains positive within artificial
occlusion; the full progressive goal remains unmet. Further work needs a separately
frozen generation/trajectory intervention or another explicitly isolated mechanism,
while preserving these negative results and all original denominators.

## Latest: full result verified; final-pass diagnostic queued (11 November, 11:00 UTC)

Full surface continuation **39387** completed successfully at 10:17:13 UTC;
finalizer **37814** completed at 10:24:23 UTC. Both runs have all 2,565 targets,
passed full/provenance verification and were registered. The old evidence
collector finished recovery; do not restart it.

Full paired gains: CLIP **1.123%** [0.768,1.480], LPIPS **0.835%**
[0.520,1.173], VGG feature similarity **2.789%** [1.911,3.764], SSIM **1.782%**
[0.934,2.694]. The full 3% objective is NOT met. CLIP uses 946 shared successes;
other metrics retain all failure penalties. No natural hidden RGB MAE exists.
Saved outputs: control 951 / surface 958; shared 946, gained 12, lost 5. The independent
128-image raw-selection confirmation remains a separate positive result.

Read output/surface-progressive-review-20261111/REPORT.md or REPORT.pdf; the
portable surface-progressive-review.zip includes the audit and all paired records.
Local audit independently reproduces means, intervals, failures and denominators,
verifies 14,123 recovered members and 4,921 pre-resume record hashes (2,565 control
plus 2,356 surface). Native PNGs remain on the cluster; neural metrics were not
rerun locally. The old timed-out execution snapshot remains unfinished, while its
successful continuation and pre-resume records are verified separately.

The sampler ran on 966 targets; 1,599 never reached it. Shared-success LPIPS gains
4.165%, versus 0.835% over all targets. Do not substitute that diagnostic subset
for the frozen primary denominator or change detector vocabulary to claim a
surface gain.

New frozen diagnostic: output/surface-lastpass-ablation-20261111-design/freeze.json,
SHA256 4866d3f2a1a26e207b45a101ef8e3c0316cc18af828188b1edf017229a7210c8.
It uses each existing final raw candidate selected by the frozen ranker, retains
the corrected run's final mask, frame, geometry, failure status and preceding
trajectory, and removes final RGB corrections/restoration. Baseline, corrected
surface and raw-final images are rescored together. This is exploratory, NOT an
independent holdout or an end-to-end raw progressive method. No new diffusion,
ranker fitting or hyperparameter search is performed.

Four-Quadro/four-worker measurement job **39404** is queued (two-hour limit).
CPU finalizer **39407** waits on afterok:39404 and produces analysis/REPORT.md,
analysis/summary.json and a hashed evidence archive. Exact commands and uploaded
source hashes are in the design submission.json and finalizer-submission.json.
The last start estimate was11:37 UTC /18:37 Vietnam; a test-only gpu-partition
probe was later and created no submitted job. Preserve these valid jobs.

New read-only collector is alive in unified execution session **62314**:
`.venv/Scripts/python.exe -u scripts/collect_surface_lastpass_evidence.py --measurement-job 39404 --finalizer-job 39407`.
It writes output/surface-lastpass-ablation-20261111-recovery/status.json and
recovers completed numerical evidence with independently obtained hashes. Do not
start a duplicate. It stops on integrity or job failure; network errors retry.
The previous network interruption was resolved when the user reconnected.

Four focused derivation regressions pass on Windows and pinned WSL Python3.10.
Remote preflight verifies frozen hashes and one real candidate's crop, selected
identity, alpha, and JPEG/RGBA serialization. The queued job performs the existing
GPU integration checks before scoring. No frozen scientific implementation was
edited. Corrected accidental spaces in four confirmation handoff paths.
Original five-hour deadline exceeded; no full metric success is claimed.

Next: check 39404/39407, recover and inspect raw-final versus corrected and matched
control comparisons, including original-score replay drift. A final-pass result
cannot establish what a wholly raw progressive trajectory would do. Do not start
an expensive new full run based solely on the already-known controlled gains.

## Current result and resumed full run (2026-11-11)

Cluster access works again after the user reconnected. Confirmation job **37843**
completed successfully in 1h20m33s. Its independently frozen 128-image test meets
the four predeclared controlled endpoints versus expected uniform selection from
the same eight candidates: MAE +7.644% [4.287,11.291], object LPIPS +11.328%
[8.021,14.828], DreamSim +16.766% [11.808,21.583], CLIP cosine +4.068%
[3.504,4.653]. Intervals are individual nominal 95% paired source-stratified
bootstrap intervals, not simultaneous coverage. Zero failures in 1,024 candidates.
SSIM improves only 1.668%; do not claim every recorded metric clears 3%.

Local audit passes 2,631 recovered archive member hashes, 1,024 candidate identities,
128 rank decisions, exact selected RGB/masks, selection-before-scoring timing,
all summary means/intervals, and independent selected-image hidden MAE for all
128 cases. Native generation-node replay passed; local audit did not rerun neural
metrics. Read output/surface-confirmation128-20261110/report/REPORT.md and REPORT.pdf.
All 128 gallery images load; filters show 64 VG, 64 COCOA, 86 gains and 42 losses,
with no horizontal overflow. PDF layout was visually inspected. This is one-step
artificial occlusion, not a completed full progressive benchmark improvement.

Full job **37812** reached its 24-hour limit after 2,356/2,565 target records.
Continuation **39387** was submitted at 07:17:49 UTC on 11 November, with four
Quadro GPUs, four workers and 12 hours, using the same freeze/output directory.
The one incomplete target was archived before retry; completed results were kept.
Exact command, archive hash and dependency update are in
output/surface-followup-20261110-design/resume-20261111-submission.json.
Finalizer **37814** now correctly waits on afterok:39387. At 07:42:53 UTC, control
has all 2,565 records and surface has 2,383; the job is RUNNING. Its corrected
policy differs from the independently confirmed raw selection policy.

The read-only collector now watches **39387/37814/37843**, using
`.venv/Scripts/python.exe -u scripts/collect_surface_job_evidence.py --progressive-job 39387`.
Existing Windows process pair 40212/37892 is one collector (venv launcher/child),
not duplicate collectors. Do not launch another while it is alive. Read
output/surface-job-recovery-20261110/status.json for current progress. Five collector
regressions pass, including replacement-job monitoring. Earlier session 20571 and
the queued/unmeasured confirmation statements below are historical.

Remaining work: recover finalizer outputs, review full denominators/failures and
paired comparisons, and report the full result separately. Preserve the frozen
method and previous primary/secondary labels; do not tune on confirmation data.
The original five-hour deadline was missed. Leave valid inference running.

Portable delivery: output/surface-confirmation128-20261110/START_HERE.md links
surface-confirmation-delivery.zip (5,341,427 bytes; 443 members; all 442 payload
hashes and ZIP CRC pass). SHA-256:
e988278dbe6f8bc45ed7336d230c2deffbc1a47a1b135497b40e49e70a70b924.
The earlier 49 MB reading-package.zip is superseded; it also contains redundant
native selected PNGs. Final delivery omits those images but includes every gallery
panel, all metric records/decisions, frozen source and audit receipts.

Final health check at 07:49:16 UTC: 2,394/2,565 surface records; two workers have
finished and both remaining logs are actively updating, with no recent traceback.
Both remote frozen-policy hashes remain exact. Finalizer dependency is still
afterok:39387. Receipt: design/resume-20261111-health.json (under the original
surface-followup-20261110-design directory). The five collector regressions pass
on Windows Python 3.14 and pinned WSL Python 3.10. Do not interpret the unfinished
full comparison as either a positive or negative quality result.

## Active five-hour follow-up (2026-11-10)

05:19 UTC: goal marked **blocked awaiting external Slurm results**, after the
same missing-result dependency persisted across more than three goal turns and
all independent review/recovery preparation was completed. This does not stop
any experiment. Last live collector output:37812 RUNNING,1,076/2,565 control
records;37814/37843 PENDING. Collector20571 is live, polling and ready to recover
finished evidence. The metric objective remains unverified; resume assessment
when confirmation and full-comparison artifacts exist or a job requires repair.

05:17 UTC CURRENT: all independent implementation/review work is complete;
remaining quality verification requires externally produced Slurm artifacts.
37812 remains RUNNING with 1,068 control records; 37814 and 37843 are PENDING.
There are zero confirmation candidates and neither final result artifact exists.
The same external result dependency has persisted across at least three live
goal checks. No method change is justified by incomplete test outputs.

Added and started a read-only evidence collector in execution session **20571**:
`scripts/collect_surface_job_evidence.py`. It polls all three jobs every 60 seconds,
recovers the finished confirmation archive and full progressive reports/JSON
metadata, validates independent remote checksums, and preserves differing local
evidence. It never mutates remote files or jobs. Four focused safeguards pass on
Windows and pinned WSL Python 3.10; immutable confirmation preflight passes.
Read output/surface-job-recovery-20261110/README.md and status.json. Poll session
20571 rather than launching another collector. Old monitor77472 still exists.
The correct full result filename is progressive-comparison/comparison.json,
not summary.json. After artifacts exist, review the recovered reports and
receipts before deciding whether the original metric objective is achieved.

05:07 UTC CURRENT: capacity review found no faster validated execution route.
Local RTX 3060 has 6 GiB VRAM; the full runtime does not fit its validated
configuration. A read-only `sbatch --test-only --partition=gpu --gres=gpu:4`
probe returned 21:48:49 UTC, later than pending 37843's 09:45:49 UTC estimate.
The test-only probe created no job. Preserve all existing allocations. Current
37812 control has 963/2,565 records; all four worker logs updated within ten
seconds and have no traceback in their last 200 lines. Surface has not started;
37814 waits afterok and 37843 has zero candidate records. No method, source,
cohort, or scientific measurement changed. Live receipts and jobs.json are
updated; see confirmation design/status-20261110T050720Z.json. Monitor 77472
still runs for 37812/37814; query 37843 separately. The queued confirmation is
estimated for 16:45:49 Vietnam time, subject to scheduler changes. Remaining
work depends on these valid jobs finishing, then artifact recovery and reporting.

04:17 UTC CURRENT: independent selection-only confirmation is frozen and
submitted as Slurm **37843**, debug-gpu, four GPUs/four workers, four-hour limit.
Read docs/SURFACE_CONFIRMATION_20261110.md. New experiment is
output/surface-confirmation128-20261110; design is the same name plus -design.
All 128 cases (64 VG, 64 COCOA) exclude all 349 previously used images. Eligibility
was 861; decoded and aligned near-copy audits pass. Freeze SHA is
711f49db26e19d0655c64b59682d9ae50c0841d38071096ad9dc7f83d733d4c4.
The parent ranker/seeds are unchanged; correction, consensus, and visible RGB
restoration are disabled. This independently confirms the previously secondary
raw component; it does not replace the earlier corrected primary. No outcomes
from these 128 have been generated or measured. Remote preflight passed hashes.
The source/input archive SHA is c07874df475cece5ec04acf9ed7189ca2cafd4090f9f178c02f78e19dbd93038.
New scripts prepare_surface_confirmation.py, finalize_surface_confirmation.py,
and run_surface_confirmation_slurm.sh are frozen with their dependencies.
248 CPU tests: 242 passed, six optional-dependency skips; shell/smoke pass. The
two new regressions also pass in the pinned replay environment. No parent-frozen
scientific source was edited. Job 37843 runs GPU checks, all 1,024 candidates,
features, observation-only decisions, metrics, analysis, exact same-node replay,
and creates analysis/CONFIRMATION.md plus design/confirmation-evidence.tar.gz.
Download the archive and independent .sha256 after completion; preserve raw
data and report whether every point estimate and/or lower bound exceeds 3%.
Submission receipt and exact command are design/submission.json. Actual squeue
estimates 09:45:49 UTC; cluster display is EST (04:45:49). Estimates can change.
Do not submit duplicate work. Full progressive 37812 remains running (823
control records at 04:50 UTC); 37814 waits afterok. All four worker logs update
without a recent traceback. This continuation is a verified wait. Current
control uses about10 GB per allocated24 GB GPU; reported batch peak host RSS
is46 GB against64 GB allocated. Receipt: design/runtime-capacity-20261110T0432Z.json
under the original followup design directory. This is current control capacity,
not a certification of full surface-stage peak memory. No scientific code,
policy, allocations, or measurements changed. Monitor session77472 still
works for those two jobs only; query 37843 separately. The original five-hour
deadline is already missed. Keep the goal active until measurements exist.

03:59 UTC historical: read-only SSH monitor is running in unified exec session77472.
Poll it with write_stdin(session_id=77472, chars='', yield_time_ms<=60000); it
queries authoritative sacct states and counts both run directories every60seconds.
It stops on a terminal primary failure or terminal finalizer, without changing jobs.
Its latest live output:37812 RUNNING,37814 PENDING, control302 targets, surface0.
Current receipt: design/progressive-status-latest.json. This goal turn was a
verified wait; the control advanced without requiring code or experiment changes.
All four worker logs had updated within9-20seconds at03:52; none of their last
200 lines contained a traceback. The recent slower minute was not a stalled job.
This continuation is a verified wait; no code, policy or scored results changed.
If monitoring disconnects, re-query those same Slurm job IDs; a dead monitor does
not prove the GPU job stopped. Do not start duplicate inference or reset outputs.
No scientific work remains independent of the pending full comparison; keep the
goal active while these valid jobs advance. The delivered controlled result and
frozen parameters remain unchanged.

03:36 UTC historical: verified live job37812 advanced to151/2,565 control targets;
37814 is still pending. Storage has28TB available. Independent inspection of
the final progressive smoke verifies19 artifact hashes, eight distinct derived
seeds, the target seed and diagnostic linkage, selection index1 under the frozen
ranker (maximum loss discrepancy1.11e-16), and exact preservation of81,241 observed
pixels in selected.png. Receipt: design/progressive-smoke-artifact-audit.json.
This audit is one smoke pass, not full-run validation or another quality result.
No new method changes or experiments were made; the full comparison is pending.

03:32 UTC historical: job37812 is live; all four control workers are generating.
Matched control has127/2,565 recorded targets (37 saved,90 failures); full surface
starts after control verification.37814 remains pending on37812. This is progress
and a verified wait, not a stalled process or a completed full comparison.
Improved the pending comparator to show separate overall means/denominators,
per-source saved/failure counts, shared-success CLIP, and immutable analysis-source
copies. A synthetic changed-coverage case exposed a zero-baseline bootstrap crash:
full-run analysis now reports the relative interval unavailable, counts undefined
draws, and retains the absolute interval; it never drops those draws. Empty shared
CLIP is similarly unavailable. No fresh controlled metrics or method changed.
Nine focused tests pass locally and under pinned cluster Python3.10. Four uploaded
file hashes match; all17 frozen scientific sources remain byte-identical. Receipt:
design/progressive-report-deployment.json. Remote prior reporting sources backed
up in design/progressive-report-source-before-20261110T0331Z. The delivered reading
ZIP is immutable; these full-comparison reporting changes are separate.

03:26 UTC historical: controlled deliverable is complete and locally verified.
Read output/surface-followup-20261110/START_HERE.md. Three-page report PDF,
all166 searchable panels, numerical records, choices, frozen source and receipts
are packaged in surface-followup-reading-package.zip (8,839,075bytes;555 members;
SHAe57f9a6779584853201154746923558b8ba1dece5fbcd6390c18bdb3161e1e05).
All ZIP member hashes pass. Fresh evidence65,366,265bytes matches remote SHA
3bddef2f178b29db9979fa1869be258a99ff91b1cc2f0472fffc742c658b949d and is extracted.
Report validator recomputes primary/secondary means and checks332 selected
artifacts. Browser confirms166 loaded panels,120 MAE gains/46 losses, correct
source/search filters and no horizontal overflow; PDF is visually reviewed.
The control-comparison table now explicitly labels gains of the primary over
each named control. Frozen scientific files and all166 decisions are unchanged.
Job37812 passed its final eight-target/four-worker progressive smoke, including
a saved surface completion, and provenance verification. At03:25:48UTC its full
matched control has82 target records; full method has not started yet. Valid job
37812 remains running on four Quadros;37814 waits afterok to verify/compare/register.
Current remote snapshot is design/progressive-status-delivery.json; smoke receipt
matches independent remote SHA. No new studies should be tuned on fresh outcomes.
The original five-hour deadline was missed. Primary CLIP misses3%; secondary
selection-only clears all four point estimates, with MAE/CLIP intervals below3%.
Full progressive results remain outstanding. Do not mark the ultimate metrics
objective achieved from controlled gains or terminate valid long-running jobs.

03:13 UTC historical: fresh166 scoring finished. Corrected primary gains are
MAE9.086%, LPIPS12.560%, DreamSim18.553%, CLIP2.422% (CLIP target missed).
The pre-specified selection-only component trained_surface_raw reaches
MAE5.845%, LPIPS6.754%, DreamSim12.764%, CLIP3.075%; all point estimates exceed3%,
but raw MAE/CLIP intervals extend below3%. Keep this SECONDARY; no post-test
refreeze or substitution of the primary. Summary/csv are local and archive SHA
verified in fresh-summary-download-receipt.json. Zero candidate failures.
Job37785 failed strict primitive-score replay after successful GPU measurement.
Cross-CPU Lab histogram boundary changes persist even with pinned packages;
CPU37798 on floyd also failed. Original generation node gus matches exactly.
Native CPU-only job37799 is RUNNING there (~14min, healthy four-core compute).
Dependent37812 has four Quadros for final smoke then full matched control/method;
37813 packages all controlled evidence after37799; 37814 finalizes full outputs.
Old37795 was cancelled/superseded. None of the17 frozen scientific files changed.
Wrapper run_surface_final_slurm.sh gained SURFACE_PROGRESSIVE_ONLY=1, requiring
a complete matching replay before skipping already-finished controlled work.
New wrappers isolate user-site packages. Failed runner snapshot is preserved.
Extra candidate transfer63363 FINISHED, SHA91a8ea4661a00f08bc504ffd3f2fd8f7db7d3791d5a96337318eb7ac8c643f3e.
Extracted to design/extra-pools-166/full; receipt saved. Both four/eight native
candidate pools are now local. Evidence archive from37813 is not ready yet.
Once replay succeeds, download design/fresh-controlled-evidence.tar.gz and
independent .sha256, verify then extract at repo root (this archive has full
project-relative paths). Rebuild report with --reuse-panels using cookbook,
run validate_surface_followup_report.py, inspect PDF/gallery, update findings,
and run package_surface_followup.py. START_HERE.md and report builder already
include primary/secondary distinction, exact values and elapsed deadline.
Full GPU job remains valid long work; do not mark goal complete before required
results exist or promise that the5h target/time was met. Original deadline passed.

02:51 UTC: final job37785 is healthy. Observation-only inference finished all166
at02:48:59UTC, before fresh scoring. Quadro CUDA extension, real FFT, full imports
and four observable-feature audits pass (zero feature difference). Complete CPU
suite passes241 tests plus shell regressions/smoke. Dependent CPU job37795 is
queued afterok:37785 for scripts/finalize_surface_progressive_slurm.sh. Extra
candidate SCP session63363 remains active; do not hash/unpack its partial archive.
Need retrieve completed fresh metrics/inference/analysis/runtime evidence, build
report and all166-case gallery, record measured results and any missed targets.

2026-11-10 02:43 UTC continuation: original five-hour deadline HAS PASSED, do not
claim on-time completion. User explicitly kept goal active. Job37568 finished
all349 eight-candidate pools and fresh observable features at20:43:57UTC Nov9.
No fresh quality metrics were opened before final freeze. Eight-pool source-CV
selected ridge100 all-appearance+CLIP(-16), affine1, consensus0.5/sigma4. CV gains
MAE11.415/LPIPS14.378/DreamSim19.042/CLIP3.150%; unconfirmed development estimates.
Final freeze: output/surface-followup-20261110-design/final-method-freeze.json,
SHAa0a626fcfddec095a1d7b677b496309fccb79ff6b4e446f5127b69368f3358e1.
All17 frozen source files are deployed and hash-checked remotely. Fresh truth
archive uploaded with matching SHA199c8c6a697404b91a7821bd5871d8bfa9b1ec9a3256180984403f38f1b7fdfe.
Final Slurm job37785 submitted debug-gpu/fourGPUs/24h (Gus busy, Quadros available).
It selects all166 observation-only outputs, computes all raw/refined/calibrated
metrics, analyzes with cached paired bootstrap, independently replays scores and
MAE, runs frozen8-pool progressive smoke, then matched full baseline and surface.
Full baseline path is now pdmc-matched-control-20261110 (not L40-specific).
Main log design/final-37785.log. Fresh outputs metrics-eight/analysis-eight/
inference-eight under output/surface-followup-20261110. Preserve frozen method;
no tuning after fresh metrics. Report scope accurately, never claim controlled
results prove full progressive improvement.
Extra166 candidate bulk download running exec63363: fresh-extra-results.tar.gz,
expected SHA91a8ea4661a00f08bc504ffd3f2fd8f7db7d3791d5a96337318eb7ac8c643f3e.
Dev87 JSON archive downloaded/extracted and SHAverified (4f20c250e732d3ad843de359a43fc01f8dd6459716941f27f110cad6e7a25295).
Four eight-pool search jobs finished; best linear beat all nonlinear alternatives.
Analysis caching change passes exact source-stratified CI regression; no model
math changed after freeze. Need final artifact verification/report/gallery and
monitor legitimate long full runs; goal not yet achieved.


19:59 UTC update: all 183 old images have eight generated candidates. Development
87 metrics finishing; ensemble87 correction measurement ~52/87 cases. Extra166
fresh generation starts next. Four-pool nonlinear comparison finished, best still
ridge (MAE +2.674%, LPIPS +9.999%, DreamSim +13.621%, CLIP +2.462% in source-weighted
leave-source-out dev CV). Final eight-pool grid will include the same modest tree
family; no fresh outcomes viewed. New portable tree scorer in surface_refinement.py;
CPU-only sklearn1.2.2 installed in WSL pinned environment, no GPU dependency changes.
Nonlinear ordering/export test passes. Full pre-freeze suite passes238, shell
regressions and synthetic smoke; new validator and deployment-hash tests also pass.
WARNING final deployment needs uploading new surface_refinement.py, surface_policy.py,
measure_surface_followup.py, select_surface_pool.py, develop_surface_multimetric.py,
freeze_surface_followup.py and final runner before final freeze/evaluation. Existing
remote running development jobs use the earlier source (same corrections/ridge math).
Final freeze should be made in WSL; generator now stores POSIX source-hash keys.
Fresh metric and observation-only scripts now verify every frozen source before
proceeding, so stale runtime cannot silently consume a nonlinear frozen model.
Full comparison runner prepared, not submitted. Both matched-runtime control and
surface development smokes complete and provenance-verified. The large ensemble96
archive transfer was intentionally stopped (contained images); JSON-only archive
is downloaded/extracted with matching independent SHA948e8d50c67ba3bfeb47beb0521da70240a98782969aa5c9f54087aa2600becc.


19:50 UTC update: 37568 still generating extra seeds on the 87 old images (~240/348
records); 96 extra pools, features, metrics and joint correction pools complete.
Fresh truth remains unmeasured. Final selector NOT frozen. Fresh-four archive is
fully downloaded/extracted and hash-verified; prior SCP finished.
Progressive smoke step 37568.7 failed DINO/InstaOrder utils collision. A scoped
module-cache isolation fixes this; six runtime tests pass. Step 37568.8 succeeds,
all four workers, 1/8 saved completion, scored and provenance verification passed.
Output: pdmc-surface-progressive-smoke-v2-20261110. A matched L40 baseline smoke
runs in exec 37076. The runner now supports baseline invocations (without surface
cache CLI flag). Failed first smoke is preserved. New final workflow script is
prepared but NOT submitted: run_surface_final_slurm.sh will select observation-only,
measure 166 fresh cases, analyze/replay, final-policy smoke, then matched L40 full
baseline and full progressive surface run (each 2,565). Longer runs may continue.
Local ensemble-eight-96 metadata transfer uses exec 13446; verify completion and
SHA 5d5466e8cc8164ef6cb5b76a3c5d4881947f4f62ecbe399191cef16147e5b8e2 before use.


19:24 UTC update: job 37568 is running extra four seeds (first cohort 96 nearly
finished). Both four-pool ensemble metric cohorts finished in 37568.2; metadata
archive hash verified locally. Correction strength=1,sigma=4 reduces dev selected
MAE .165210 -> .157371, with small CLIP loss. Final eight-pool method NOT frozen.
New primary development utility matches known fresh source counts 64/64/38/0;
unweighted sensitivity remains. No fresh truth metric has been opened.

Opt-in progressive adapter now in surface_runtime.py and evaluate_open_world.py,
with run_surface_progressive_slurm.sh. It retains original measured support
across iterations, explicitly seeds candidates, restores original visible pixels
at native resolution before final segmentation, and retains upstream stopping.
Four focused tests pass; full WSL check passes 233 tests, all shell regressions,
and synthetic smoke. Additional source-weight arithmetic test passes. Runtime
feature-provider equivalence audit runs in 37568 via exec session 60380; logs in design/runtime-feature-audit-logs. No GPU progressive
smoke or full surface run has been launched yet. Source/evaluator uploaded.

Fresh four-pool bulk SCP started 19:05 still running (Windows scp PID 83196);
local fresh-four-results.tar.gz is partial until transfer completes. Do not unpack
or claim verified until remote independent SHA256 matches.


User requests 3–5% relative MAE and paper-metric improvement, continuing the
surface-consistency direction. Window: 2026-11-09 18:07–23:07 UTC. Goal is active.
See `docs/SURFACE_FOLLOWUP_20261110.md`. New cohort `output/surface-followup-20261110`
contains 166 images (64 VG, 64 COCOA, 38 free-images), disjoint by identity and
bytes from all 183 earlier cases. LAION eligibility is exhausted. All old 183
cases are explicitly development now, without rewriting their old test results.

19:03 UTC continuation: all 183 development images now have paper-style black-
background LPIPS/DreamSim/CLIP measurements and observation-only deep features.
The balanced multi-metric ridge grid is in `development-multimetric-v2`; four
candidates have only 3.15% maximum development CLIP headroom, and forcing CLIP
hurts MAE. Additional seeds 55,66,77,88 for all 349 images are queued in Slurm
**37568** after **37566**, with three cohorts 96/87/166. Script
`scripts/run_surface_extra_slurm.sh` generates each cohort, extracts eight-pool
features, and measures OLD development cohorts only. Fresh truth is still local
and unopened by any metric job. Never tune on the fresh 166 outcomes.

Development metadata archives are locally verified against remote SHA-256.
`provisional-development-eight.json` is remote and deliberately provisional.
Eight-pool outputs go to design `extra-pools-{96,87,166}/full`; eight-pool
development metrics/features go under design `eight/`. Primary protocol now uses
eight candidates with equal-budget validity/consensus/semantic/deep controls.
Final controlled parameters are NOT frozen yet.

Full cached postprocessing is separately frozen in `full-calibration-freeze.json`
(affine strength 1, max shift 32, exact observed-pixel restoration, unchanged
alpha). Remote `output/pdmc-surface-calibrated-20261110` has all 951 outputs and
1,614 original failures. Paired scoring in allocation 37566 completed. Finalizer
exec session **26847** verifies/aggregates all 2,565 targets; read log
`full-calibration-finalize.log`. The finalizer uses source-image cluster bootstrap.
Do not change this frozen calibration after seeing full metrics. Four-source
visual audit is saved locally as `full-calibration-visual-audit.jpg`.

An extra consensus correction is being developed before fresh-test exposure:
`amodal_benchmark/surface_ensemble.py` adds bounded smooth median-candidate RGB
residuals only on mask-agreement support, keeping chosen masks and visible pixels.
CPU grid runs locally in exec **35609**, writing `ensemble-development-four.json`
on completion and progress to its `.log`. It is roughly halfway done. Its five
core tests (four ensemble, one clustered bootstrap) pass; two deployable-policy
tests also pass. Optional SciPy 1.10.1 is now installed only in the isolated WSL
`surface-replay-py310` environment. All six ensemble/policy tests pass there.

New `surface_policy.py` and `scripts/select_surface_pool.py` can run the final
method from observation manifests without evaluation data. Metric evaluator
supports `--extra-run-dir`, `--features-dir`, variable seeds, optional ensemble
correction, calibration-only ablations and explicit valid-pool controls.
Feature audit passed four GPU cases: hidden-observation randomization invariance,
zero identical-visible distances, and official DreamSim forward equivalence.
Latest full CPU log: `logs/surface-followup-check-extended-20261110.log`: 226 tests,
five optional SciPy skips (subsequently checked separately), all Bash regressions
and synthetic smoke. PowerShell incorrectly reports native stderr as wrapper exit
1; inspect completed logs/native exit codes, do not misreport test failures.

New module `amodal_benchmark/surface_refinement.py` implements bounded visible
residual calibration and a small regularized within-pool ranker. Development
leave-source-out search currently selects ridge=10 and LPIPS objective, giving
exploratory cross-validated MAE reduction 4.995% and masked LPIPS 5.748%; these
are not fresh-test results. Correction grid is running locally in exec session
19286; outputs go to `output/surface-followup-20261110-design/development`.

Quadro job 37564 was pending until after the deadline and was cancelled by us.
L40S smoke 37565 passed CUDA extension forward/backward, real FFT, full imports,
and all eight generated/masked candidates. A separate cuFFT 10.9.0.58 library
at `.amodal-cache/surface-cufft-cu118` is loaded with LD_PRELOAD only for this
study. The original pinned library remains unchanged. Cross-runtime RGB is
not byte-identical; MAE differences are recorded in `smoke-audit.json`.
Fresh 664-candidate generation is job **37566**, four GPUs/four workers on gus.
Log directory: `output/surface-followup-20261110-design/job-logs`.

An isolated `surface-metrics-env` inherits the pinned runtime and adds DreamSim.
Setup initially found missing wcwidth, then peft 0.2's custom-model config error;
script is updated to wcwidth 0.2.13 / peft 0.5.0, pending rerun. Weights are cached.
Local smoke download matches independent remote SHA-256; four seed-11 images
were visually inspected. Do not overwrite accepted baseline or historical studies.
Freeze the method before evaluating fresh truth; then run controlled metrics and
the separately identified full-baseline surface-postprocessing comparison.

## Surface-PMC task completed (2026-11-08)

The requested theory-to-evidence work is complete within the 07:42-12:42 UTC
window. Start with [the handoff](output/surface-consistency-20261108/START_HERE.md),
[13-page report](output/surface-consistency-20261108/report/REPORT.pdf), or
[durable results note](docs/SURFACE_CONSISTENCY_RESULTS.md).

All 732 candidates for 183 images are available locally and on the cluster.
Development has 24 images; pilot test 72; image-disjoint replication 87.
Primary full-surface pooled gain is +0.004310 [-0.000512, +0.009074], inconclusive.
The fixed local variant has exploratory pooled gain +0.007187
[+0.002183, +0.012296], with supporting LPIPS, but no established superiority to
full or visible-preservation selection. The 51/159 fixed-support truth
counterexamples, filtering/ranking decomposition and qualitative losses explain
the limited compatibility assumption. Independent human preference is absent.

Both cohorts pass exact prepared-pixel/scorer/metric/policy replay in isolated
WSL Python 3.10, NumPy 1.23.5, Pillow 9.5.0. Full CPU checks passed 212 tests,
five Bash regressions and smoke. The final report validator and browser checks
pass; all 159 cases and 318 panels are present. Frozen analysis hashes match.
Full/local decisions are unchanged in the separate Windows numerical-portability
audit; one boundary-plus-color pilot decision changes.

Generation and LPIPS jobs 36891/36894/36895/36897/36898 completed. Validation job
36958 failed immediately because scripts/prepare_surface_experiment.py was not
deployed remotely; this is recorded, not hidden. Successful local exact replay
supplies the validation. Cluster diagnostics completed separately and match the
local pinned diagnostic summary. No requested training/inference job is pending.
Original cluster archives and SHA-256 receipts are preserved. Do not duplicate
completed transfers or resume old session IDs in the historical notes below.

Scientific source and both development/test decisions remain frozen. The report
builder, validator, portability audit and delivery packager are reusable; commands
are in cookbooks/surface-consistency.md. Future scorer changes require new
validation/test data and a separate output directory. The original accepted
baseline and unrelated user jobs were preserved.

## Historical Surface-PMC execution notes (superseded by completion above)

Continue theory to explainable measured results within the user's five-hour
window. Goal is active. Protocol: `configs/research/surface-consistency-v1.json` and
`docs/SURFACE_CONSISTENCY_EXPERIMENT.md`. Artifacts:
`output/surface-consistency-20261108/`. Ninety-six prepared cases, 24 development
and 72 test, 24 per source, one target per image identity. Evaluation truth stays
local and separate; the generator receives only `inputs/`, `protocol.json` and
`generation_manifest.json`. Seeds 11,22,33,44; MC 35/50; one-step shared pools.

Implemented `amodal_benchmark/surface.py`, preparation and GPU candidate scripts,
and `scripts/run_surface_slurm.sh`; 11 focused tests passed. Smoke job **36891**
requests four GPUs on debug-gpu, four workers, first four cases and first two
seeds. It runs real CUDA/FFT and pipeline import checks first. Remote/local paths
are both `output/surface-consistency-20261108`; remote logs in `job-logs/`.
Source is unchanged for the accepted baseline. Unrelated user jobs exist and
must be preserved. Next actions are smoke inspection, full shared-pool generation,
development-only normalization freeze, held-out analysis and honest report.

08:03 UTC update: smoke 36891 completed (8/8 valid generated outputs; runtime
checks passed). Full 384-candidate job **36894** runs; LPIPS **36895** waits
after it; replication **36897** waits after metrics. Replication output root is
`output/surface-consistency-replication87-20261108`, 24/24/24/15 new cases by
VG/COCOA/free-images/LAION, all test and disjoint by image ID and bytes. The
96-case attempt exhausted LAION eligibility and is retained incomplete at
`output/surface-consistency-replication-20261108`; it was never generated.
Use `analyze_surface_experiment.py --stage freeze` on the pilot only, then
`--stage test` on the pilot and `--stage test --freeze-from PILOT/analysis/development_freeze.json`
on replication. Test analysis must wait for the freeze. Existing census scoring
is running locally in exec session 70278; its log is `logs/surface-census-20261108.log`.
Full WSL check passed 203 tests and all Bash/smoke checks. Source snapshots and
exact-hash recovered original preparation source are retained under pilot/source/.

08:37 UTC continuation: development was frozen on the cluster at 08:22:29 UTC;
source hashes match locally and the selected exploratory development rule is
`texture`. Pilot test analysis completed at 08:27:47 UTC: 72 cases, 288 generated
outputs, full-surface gain +0.002972 (95% CI -0.004722 to +0.011277), local-seam
+0.011096 (+0.003282 to +0.019391). These are pilot findings only. Original full
pilot archive transfer is still active (Windows SCP PID 68588). Replication
36897 is running, followed by metric job 36898. Exec session **82966** waits for
both, analyzes replication with the frozen pilot lock, packages artifacts and
starts SCP to the replication folder. Do not duplicate these operations.

Added post-hoc explanations without changing frozen science: exact filtering /
ranking / oracle-regret decomposition, within-image pair ordering, and a
fixed-support truth-versus-selected compatibility diagnostic. Declaration is
`posthoc_explanation_declaration.json` (08:30:53 UTC, after pilot errors and before
replication errors). New derivations are in `docs/SURFACE_CONSISTENCY_THEORY.md`.
The independent validator now recomputes every hidden metric, policy and summary
from candidate artifacts, and rejects corrupt or nonfinite saved results.
Full CPU checks are running in `logs/surface-explanation-check-wsl-20261108.log`.
Historical census associations are complete; full-surface AUC is 0.522 on 1,102
assessable valid cases against prior assistant labels, and is descriptive only.

## One-human visual review prepared (2026-11-08)

The user requested a handoff for the report's independent human rater. Preparation is complete at [output/human-review-20261108/README.md](output/human-review-20261108/README.md). The default is a random 160-case audit (eight batches; about 3–6 hours); a full 1,390-case census alternative (70 batches; about 23–46 hours) is also packaged. An optional workload question was asked; no answer was received before preparing both options. Choose one scope before collecting responses.

Both reviewer ZIPs contain an offline HTML form, guide, blank CSV, neutral preview panels and byte-identical native source/visible/completion/raw images. Previous assistant labels, numeric metrics and private identity mappings are excluded. The form requires six ratings plus evidence/confounds, handles U/NA, records independence disclosures, saves drafts and exports/imports JSON. Reviewer data is not uploaded anywhere.

Source helpers: `scripts/prepare_human_review.py`, `scripts/validate_human_review.py`, `scripts/summarize_human_review.py`; instructions: [docs/HUMAN_VISUAL_REVIEW.md](docs/HUMAN_VISUAL_REVIEW.md). The importer writes a new human annotation set, per-criterion counts and human–assistant agreement without overwriting the assistant census. It rejects incomplete assignments unless explicitly asked for an interim summary.

Validation: 15 focused tests passed, including synthetic end-to-end import. Both ZIPs match their exact public manifests; all 800/6,950 images decode; 640/5,560 native evidence hashes match. Browser tests covered offline opening, blank defaults, validation, autosave/reload, import/export and confound handling. Synthetic browser data was cleared. No real human responses, external messages, inference, cluster operation or changed experiment configuration occurred.

Audit ZIP SHA-256: `519ab277cb4a40dbbb8a3742fba65a0b9782219a5456e61de4c3f7e504704cf2` (226,870,269 bytes). Census ZIP SHA-256: `e78b58b30ff903613bb1f1786385438807a9c7a4ed0bde7394fead3e174e34cc` (1,920,708,510 bytes). Actual independent human evaluation remains pending; no human-consensus claim is supported.

## Expanded-vocabulary appearance census completed (2026-11-08)

The user requested the Person 2 deliverable on expanded vocabulary, with all outputs if possible and a three-hour budget. The resulting [13-page PDF](output/appearance-expanded-20261107/REPORT.pdf), [report](output/appearance-expanded-20261107/REPORT.md), [taxonomy](output/appearance-expanded-20261107/TAXONOMY.md) and [filterable census viewer](output/appearance-expanded-20261107/review/index.html) use the recovered `pdmc-open-world-sam-annotated` run. The earlier baseline sample below remains separate.

- All 2,565 records were analyzed; all 1,390 saved outputs received explicit single-assistant visual labels. The 1,175 no-output failures are accounted for separately. The intermediate 552-case sample is archived and superseded by the final census with unit inclusion probabilities.
- At least one observed appearance defect: 652/1,390 (46.9%). Positive/assessable counts: texture 578/1,304; color 398/1,345; pattern 311/601; semantic drift 6/1,350; hidden-region inconsistency 559/1,113; co-occurrence 11/1,199. Maximum severity: 347 mild, 243 moderate, 62 severe. Labels overlap and U/NA do not count as passes.
- All eleven confirmed co-occurrence cases are in raw RGB only; extras are excluded from returned masks. Full-source checks removed six false positives involving existing context, and one ambiguous bus/car-like depiction became unknown. Twenty-four explicit adjudications preserve the initial census labels and evidence. Sixteen output target-switch tags are tracked separately from semantic drift.
- True COCOA occlusion is available for 321 targets, including 181 saved outputs. Requested 20-80% bins contain 114 targets and 71 saved outputs. Availability by bin: 24/30, 10/24, 12/24, 9/15, 13/16, 3/5. No monotonic or causal performance claim is justified. Independent mask decoding reproduces every bin.
- Generated expansion remains a separate diagnostic: Spearman rho=0.297 on 181 paired outputs, median discrepancy 12.21 percentage points. Category tables cover 284 saved-output analysis categories. Hidden RGB ground truth and independent human adjudication are absent.
- Fifteen focused appearance/census tests pass. Artifact/report validation reproduces all metric bins and label counts, passes 9,300 source-artifact hash checks, verifies all 1,390 viewer panels and the 13-page PDF. Browser failure filters return exactly the recorded positive counts. A strict confound-exclusion sensitivity analysis is included.

Final labels: `output/appearance-expanded-20261107/review/annotations.csv`, SHA-256 `ebf4f6b036d8cf592fe115e2e4811da891d29e1fe56a37a3da0ac7090b718062`. New helpers compile explicit batch/adjudication records, build the expanded report, and independently validate its artifacts. Existing generic report rendering now accepts census counts and an explicit footer. Historical execution provenance was not fabricated. No inference, remote operation, scientific configuration change or experiment registration occurred. The requested analysis is complete; independent human review is a follow-up recommendation.

The [portable delivery ZIP](output/appearance-expanded-20261107/appearance-expanded-delivery.zip) contains 1,562 files, including every census panel, and is 169,950,750 bytes. ZIP CRC verification passed; SHA-256 is `ec61b2b0558504b0740f895b1fb00a1b566caf2b023b0b43daa1dcaf2c521797`. The package omits models, original datasets, external annotation JSONs, environments and duplicate contact sheets. Its manifest and receipt are retained beside the archive. Final PDF SHA-256: `989ce2552337741e6c4492d8b6edb7cd4e86b749014d6166f62dc6b9ffe51184`.

## Appearance-failure analysis completed (2026-11-07)

Completed the user's Person 2 deliverable using the accepted baseline, without rerunning inference. Start with [the 12-page PDF](output/appearance-failures-20261107/REPORT.pdf), [full report](output/appearance-failures-20261107/REPORT.md), [taxonomy](output/appearance-failures-20261107/TAXONOMY.md), or [offline audit viewer](output/appearance-failures-20261107/review/index.html).

- All 2,565 targets retain their original recorded scores/status; 951 saved outputs have aligned mask/color diagnostics. The baseline passes full/provenance verification.
- External COCOA annotations uniquely match 321 targets by image identity and exact region name. Independent mask decoding reproduces the supplied occlusion ratios. The requested [20,80)% bins contain 114 targets and 44 saved outputs; 2,244 benchmark targets have no matched ratio.
- A deterministic sample of 120 saved outputs was reviewed by one assistant across eight expansion-proxy strata. Final labels are authoritative in `review/annotations.csv`; initial labels and same-reviewer revisions are retained. This is not independent human consensus.
- Raw positive/assessable counts: texture 54/112; color 51/113; pattern 25/45; semantic drift 4/114; hidden-region inconsistency 59/100; co-occurrence 11/105. Counts overlap. Severity, unknown/NA counts and exploratory stratum-weighted estimates are included.
- Expansion is not true occlusion: Spearman rho=0.133 on 142 matched saved outputs. No monotonic performance trend or hidden-appearance ground-truth claim is supported.
- Existing validation: 167 Python tests, five Bash regressions and CPU smoke passed in isolated WSL CPU tooling. Final report validation reproduces all bin means/denominators, sample identities and raw/weighted taxonomy, verifies local links/images and checks the 12-page PDF. Browser filters and all 120 panels work.

New offline tools are `amodal_benchmark/appearance.py`, `scripts/analyze_appearance_failures.py`, `scripts/audit_appearance_evidence.py`, `scripts/build_appearance_report.py`, and `scripts/validate_appearance_report.py`, with focused regression coverage. The report builder deliberately rejects a different baseline/review identity because its narrative describes this frozen audit. Runtime/model code, accepted artifacts, historical outputs and the registry remain unchanged. The shareable ZIP excludes the validation environment and external annotation archive. No task work remains; independent human review and controlled surface/context experiments are recommendations.

## Full baseline accepted (2026-11-06)

The requested full baseline is complete. Slurm job **31548** completed with exit code 0 in **4h 02m 38s** on four Quadro RTX 6000 GPUs. GPU inventory/FFT, CUDA extension build/forward/backward, full pipeline imports, all source-image checks, the four-worker scored smoke, full evaluation, provenance-required verification and registration all passed. The unchanged scientific contract is `configs/baseline.json` (SAM references, semantic matcher v2, released vocabulary, seed 0).

**2,565 targets: 951 saved completions and 1,614 failures (62.92%).** Saved outputs comprise 796 non-cap termination states and 155 iteration-limited outputs; neither category certifies visual completeness. There are zero missing images and zero reference failures. Overall CLIP is **28.393098 (n=951)**; LPIPS **0.735009**, feature similarity **0.242636** and SSIM **0.303397** each use **n=2,565**, including the configured failure penalties.

Recovered **16,814 files (2,960,571,039 bytes)**. Every file matches a fresh independently computed remote SHA-256, and the recovered run passed local `verify --require-full --require-provenance` against the existing dataset. All 36 captured project source files match the maintained source. See the [accepted report](output/baseline-establishment-20261105/REPORT.md) for per-source results, failure reasons, provenance, recovery evidence and interpretation limits.

- Remote immutable run: `/home/daoqm/Vu_Hung/cv_nangcao/output/pdmc-baseline-provenance-20261105`.
- Recovered immutable run: `output/baseline-31548/output/pdmc-baseline-provenance-20261105`.
- Local copy of the cluster ledger: `output/baseline-31548/experiments.csv`; it is a separate snapshot, not an additional experiment or a replacement for a local ledger.
- Registered entry: `ec823094e31f226957ef0442f0480cf14ae2db91c79550a9701a9e0aef99b07c`.
- Execution: `20261106T053037.969259Z-fdce84ff`; workflow: `20261106T052433Z-c6cfb8b1`.
- `output/baseline-establishment-20261105/status.json`, `watcher-status.json` and `final-acceptance.json` record completion. The watcher and its slow bulk transfer were stopped after successful cluster completion; exact-byte recovery replaced the transfer. No baseline job or watcher needs resuming. Do not submit the finished baseline again.

Historical semantic-v2 comparison: identical scientific/metric configurations, target/input identities and recorded seeds, but six gains and one loss among saved completions; all 945 shared completion PNGs and 32 initial reference masks differ. The historical environment was not preserved, so this is not a causal improvement or exact historical reproduction claim.

The baseline task is finished. The expanded annotated vocabulary was already evaluated on all 2,565 targets: 1,390 saved completions and 1,175 failures, with independently verified recovered artifacts in `output/artifact-recovery-2026-11-05/downloads/pdmc-open-world-sam-annotated`. Reuse those historical results; no routine rerun is needed or planned. Their missing historical execution/source/environment snapshots prevent provenance-required registration and an isolated comparison with the newly validated baseline. A fresh run would only address a separate explicit need for current-runtime provenance. Surface-PMC remains a retained, unimplemented proposal. Earlier job-state entries below are historical and are superseded by this acceptance record.

## Active task: full baseline (2026-11-05)

The user requested a complete baseline with the current provenance workflow and validated GPU runtime, with a three-hour working window beginning about 07:05 UTC. Longer valid inference may continue. This is pretrained PD-MC evaluation, not training.

Prepared [the exact launch and acceptance procedure](output/baseline-establishment-20261105/LAUNCH.md), `scripts/run_baseline_slurm.sh`, `scripts/run_baseline_workflow.py`, `configs/baseline-gpu-smoke.json` and six regression tests. The unchanged full contract is `configs/baseline.json`; new full output is `output/pdmc-baseline-provenance-20261105`, with a separate `-smoke` directory. GPU validation runs both required integration checks and exercises all four devices. Smoke must generate at least one verified completion with all four metrics, then full scoring, provenance-required verification and registration proceed sequentially. There is no session timeout that kills a valid job.

Validation: all 155 Python tests pass under WSL Python 3.10 without skips, all five Bash regressions and synthetic CPU smoke pass. Source-image preflight passes all 2,565 targets with zero missing images. Reviewed 88-file source-only upload excludes AGENTS.md/data/results/notes/ledger; hashes are in `prepared-source-manifest.json`. Windows planning was fixed to retain Linux executable paths and rechecked, including the focused WSL tests.

Read-only SSH confirms other user jobs belong to `playground-series-s6e11`; none were modified. The remote amodal project lacks the new provenance modules. The compute-node cache `/dev/shm/daoqm/amodal-cache` cannot be checked on the login node; direct SSH to gebru is denied by cluster policy. Do not work around that restriction. Validate within the normal submitted allocation. Local GPU is a 6 GB RTX 3060 and cannot support the documented four-worker baseline.

The user subsequently explicitly authorized agent-operated transfers/submissions and requested AGENTS.md updated; this is now persistent project permission and must not be requested again. All 88 source files are uploaded and match local SHA-256. The previous remote source is backed up at `.deployment-backups/baseline-20261105T094541Z.tar.gz` (hash recorded in the launch package). AGENTS.md remains local.

At 10:13 UTC, the persistent runtime is validated on Quadro RTX 6000, including the added real FFT round trip (job 30435, successful). Job 30425 had repaired the cache and passed the original CUDA/import/pinned-package tests. Scored smoke 30428 ran on L40S, wrote seven expected selection-failure records, then failed during LaMa with CUFFT_INTERNAL_ERROR. Isolated FFT diagnostic 30440 reproduced the same error on L40S without loading models. Preserve this failed attempt; its execution/source/environment/artifacts and logs are downloaded under `output/baseline-establishment-20261105/cluster-evidence/`. No scientific settings or pinned package versions were changed to work around the hardware incompatibility.

Added FFT checks to `tests/validate_cuda_extension.sh` and the baseline launcher GPU inventory, reran the six workflow regressions, uploaded both files and verified their hashes. Full baseline 30426 is back in the Quadro-only `debug-gpu` partition, with a valid dependency on new Quadro scored smoke 30436; it no longer depends on failed 30428. Smoke 30436 waits for one full GPU, and the full baseline waits for it plus four full GPUs. The full job retains four workers, 16 CPUs, 64G, 24 hours, its own eight-target/four-worker smoke, and provenance verification/registration gates. Leave these valid queued jobs in place. No full baseline is accepted yet. Current machine-readable state and all receipts are in the launch package.

Prior job 30424 failed immediately because the `/dev/shm/daoqm/amodal-cache/conda-env/bin/python` executable was absent. Pending job 30423 was cancelled and replaced to use the persistent cache; no running inference was stopped. A rejected CRLF stdin submission created no job; subsequent sbatch scripts are sent as LF bytes. Receipts, scripts and status are under `output/baseline-establishment-20261105/`. Monitor `logs/pdmc-runtime-30425.out`, `logs/pdmc-runtime-smoke-30428.out` and `logs/pdmc-baseline-30426.out`. Preserve failed attempts, update dependencies if repair is needed, and do not edit scientific settings or import historical results. Earlier sections describe previous tasks and their deployment restrictions are superseded by the current AGENTS.md.

## Experiment artifacts recovered and verified (2026-11-05)

The user's one-hour recovery task completed in 57.1 minutes. Recovered 70,012 files (11.90 GB) into `output/artifact-recovery-2026-11-05/downloads/`; every file matches an independent remote SHA-256. See [the recovery report](output/artifact-recovery-2026-11-05/REPORT.md).

Recovered semantic v1/v2, annotated, paper-v2 and shared historical LISA references. Six full historical runs and both 20-target runs pass CPU artifact verification. The three older LISA reports use separate `legacy-verification/` views with documented metadata additions; original reports, target records, scores and references are unchanged. These views use hard-linked targets and must remain read-only.

An additional 275 previously absent target records, including 82 successful completions and their references, were recovered through the documented older SAM-to-bgfix reuse chain. This isolated 295-target archive is unscored; no original learned-metric caches were synthesized.

The user explicitly authorized downloading the listed cluster folders for this task. Remote reads used shared run locks; no remote files were written and no jobs were submitted. Slow bulk streaming was replaced by local hash deduplication and exact PNG reconstruction from recorded masks/RGB using the pinned CPU environment. All 23,724 original experiment files and the maintained-source manifests remain unchanged.

New paired artifact evidence: all 946 semantic-v2 successes remain saved in annotated, with 444 additional successes. Exactly 941 shared completion PNGs are byte-identical; five completion changes match the five changed metric tuples. Reference artifacts change for 622 targets. This strengthens the historical coverage finding without establishing a pure generator improvement.

Historical execution/source/environment provenance was absent from the recovered run directories and was not fabricated. No GPU inference, learned-metric recomputation or experiment registration was performed. The next research gate remains a new provenance-backed baseline. Original report-only folders remain unchanged; use the recovered paths and reference directory documented in the report. Earlier statements that artifacts were missing describe the pre-recovery state.

## Manager review and presentation

Completed the requested team review, building on the eight-report audit and fresh CPU checks. Deliverables: [16-slide PDF](output/manager-briefing-2026-11-05/presentation.pdf), [offline presentation](output/manager-briefing-2026-11-05/presentation.html), and [speaker notes / complete results / evidence index](output/manager-briefing-2026-11-05/briefing.md).

The paired historical comparison is the central finding: annotated vocabulary recovered 444 outputs relative to semantic v2, lost none, and retained identical reported metrics for 941 of 946 shared outputs. Aggregate improvement primarily reflects coverage recovery under failure penalties. Surface-PMC remains unimplemented and unvalidated. Semantic v1/v2 and annotated detailed artifacts are absent locally, so report arithmetic is verified but full image-level audit remains pending. SAM-bgfix passed the fresh artifact verifier, with historical execution provenance absent. No local experiment ledger exists.

The proposed next gates are artifact recovery, a provenance-backed baseline under the current workflow, and a shared-candidate boundary-scoring experiment. These are recommendations, not completed experiments or newly authorized remote work. All 16 slide layouts fit, all four embedded example images load, local briefing links resolve, and the exported PDF has 16 pages. The 59-file maintained source/config/test/helper manifest remained unchanged. No GPU run, remote operation or registration was performed.

Earlier handoffs follow; their task descriptions are historical.

## Current task and result

The user asked to read `ChatGPT - Assess Reproduction Feasibility.html`, understand specifically Surface-Consistent PMC, and retain the knowledge for future questions. Completed as a documentation task.

- Read [docs/SURFACE_CONSISTENT_PMC.md](docs/SURFACE_CONSISTENT_PMC.md) for the research question, equations, example boundary scorer, integration, evaluations, limitations, optional extensions and unresolved choices.
- Read [the focused source extract](docs/research/SURFACE_CONSISTENT_PMC_SOURCE.md) for original Vietnamese wording and pseudocode. The original HTML remains unchanged; SHA-256 is `a398b6d068bd7abb2e7aee9453c6796f2c486d9b5143624b2c0112f624eeed3f`.
- Local `AGENTS.md` instructs future work to read the reference when answering Surface-PMC questions. Durable recall depends on access to the project files.
- Surface-PMC is an unvalidated proposal: sample K PMC completions and minimize boundary/color/texture inconsistency relative to visible object appearance. No code or configuration implements it as a result of this task, and no improvement was measured.
- Preserve the distinction between pseudo-occlusion hidden-ground-truth evaluation and this project's SAM/LISA visible-reference protocols. Further implementation requires explicit coordinate handling, original-visible anchoring, valid-candidate fallback and a fair sampling-budget control.

The baseline/annotated evaluation plan is unchanged. No remote operations or GPU runs were performed. Earlier repository-organization work and its validation are retained below.

---

# Previous session handoff: repository organization

Updated 2026-11-05.

## Current task and result

Review and organize the codebase, cleaning unnecessary files with extreme caution. Conservative cleanup is complete. Scientific source, entry points, configurations, tests, original papers/release ZIP, datasets, inputs, outputs and the existing environment are preserved.

Documentation starts at [docs/README.md](docs/README.md), with a [codebase map](docs/CODEBASE.md) and [test guide](tests/README.md). The root README links these guides. The paper summary moved to `docs/PAPER_SUMMARY.md` with only its relative link corrected. Both completed plans moved from `docs/superpowers/plans/` to `docs/archive/plans/` with identical bytes. The historical root `run_amodal.log` moved to `logs/run_amodal.log` with identical bytes.

## Cleanup and rollback

Removed from the workspace: three source/test `__pycache__` directories (76 bytecode files), generated `amodal_benchmark.egg-info` (six files), and the empty former `docs/superpowers/` tree. The generated files total 939,305 bytes. Automatic approval review blocked permanent deletion with the generic reason "blocked by policy"; all these items were instead moved into a rollback directory. Nothing was permanently deleted.

Rollback location: `C:\Users\daovu\AppData\Local\Temp\cv-nangcao-cleanup-20261105-0yh43rr4`.

- `source/` and `source-manifest.json` preserve the 84 original source/documentation files and their hashes.
- `run_amodal.log` preserves the original root log; `moves.json` records all four file relocations.
- `generated/`, `generated-manifest.json` and `removed-from-workspace/` preserve the generated files, sizes and hashes. The latter also contains the empty old documentation tree.
- `protected-inventory.json` records paths, sizes and modification times for all 29,318 protected files. It is an inventory, not a backup of the dataset or environment.

To undo a move, first check that its original destination is absent, then move the file back using `moves.json`. Restore the paper summary from `source/paper_summary.md` to retain its original relative link. Restore edited README/research notes from `source/` only after checking for later edits. Generated caches need no restoration; the normal Python/install workflow recreates them. Keep the rollback directory until the organization is accepted.

## Validation

- Before cleanup, Windows Python 3.14 ran all 149 tests successfully with nine expected Linux-only skips: `logs/cleanup-baseline-windows-2026-11-05.log`.
- After cleanup, WSL Python 3.10.22 ran all 149 tests without skips, all five Bash regressions and synthetic CPU smoke: `logs/cleanup-check-wsl-2026-11-05.log`.
- Installed Windows `amodal` CLI and CPU smoke passed after root egg-info/cache removal. The editable environment's own distribution metadata remains installed.
- Source/result transfer helpers passed `-WhatIf`; no host was contacted. All 123 local links checked in the root README, documentation tree and test guide resolved.
- SHA-256 checks confirm all 63 original code/configuration/test files are unchanged. Protected paths, sizes and modification times match the pre-cleanup inventory; this did not re-hash every historical image or result.

No GPU code or runtime was changed. No GPU validation, remote transfer, Slurm submission or experiment registration was performed. The research next steps below remain applicable.

---

# Previous session handoff

Updated 2026-11-04. Preserved from the preceding rebuild.

## Task

Adapt the codebase to `SETUP_NEW_COMPETITION.md` within the user's one-hour budget, retaining only the workflow needed for this amodal research project.

## Current state

Implementation, documentation and local validation are complete. No remote transfer, Slurm submission, GPU inference or experiment registration has been performed.

## Constraints

- Preserve dataset bytes, scientific protocols, pinned GPU dependencies, prior outputs and resume compatibility.
- The user performs SSH/SCP/Slurm operations. Keep `AGENTS.md` local.
- Full cluster runs request four GPUs before the script name and four workers after it.
- New scientific settings require a new output directory. Historical CLI-only runs remain supported.

## Checks and evidence

- Linux/WSL Python 3.10.22, NumPy 1.23.5 and Pillow 9.5.0: 149 tests passed, all five Bash regression scripts passed, and synthetic CPU smoke passed (`logs/check-py310-final.log`). This is CPU compatibility coverage using the runtime's Python/NumPy/Pillow versions, not a GPU execution.
- Windows Python 3.14.6: 149 tests ran successfully, with nine expected Linux-only skips (`logs/tests-windows-final.log`).
- WSL Python 3.14 also passed its 147-test suite before the final two protocol regression cases were added (`logs/check-wsl-final.log`).
- The installed `amodal` command and smoke work from outside the repository in the Python 3.10 environment.
- Source-image preflight passed for all 2,565 targets; configured two-target SAM preflight passed.
- Transfer helpers were exercised with `-WhatIf`; neither contacted the cluster.
- The printed baseline plan requests `--gres=gpu:4` and `--benchmark-workers 4` in the correct positions; its CPU allocation is also forwarded to the runner's thread budget.
- The final reference-protocol preflight regression passed (`logs/protocol-preflight-final.log`).
- After the final planner CPU-budget fix, all 25 focused workflow tests passed again (`logs/workflow-py310-final.log`).
- Earlier logs retain intermediate fixture/environment failures. Use the final logs above for the delivered state; the shell fixture's temporary runtime was corrected before the final full checks.

## Environment locations

The local editable CPU installation is `.venv` on Windows. WSL verification used `/home/hungvudao/.cache/cv-nangcao-cpu` (Python 3.14) and a separate `/home/hungvudao/.cache/cv-nangcao-py310-rebuild` (Python 3.10). No GPU environment was upgraded.

## Next action

Follow the cluster section of `README.md`: preview and upload source, run GPU extension/import checks inside an allocation, then a separate two-target smoke and full baseline. Reuse the existing remote dataset/cache. Verify and register the full result before comparing the annotated-vocabulary experiment.
