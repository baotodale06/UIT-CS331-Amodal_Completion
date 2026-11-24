# Annotated Surface-PMC: reference-runtime amendment

Recorded on 12 November 2026 before any Surface-PMC candidate generation or
paired quality analysis. The frozen 192-image / 204-target sample, primary
failure-inclusive LPIPS endpoint, intervals, trained method, seeds, source quotas,
exclusions and secondary pool diagnostic remain unchanged.

The initial control job 39473 failed when its regenerated SAM mask disagreed
with a copied historical annotated reference. An unchanged retry, 39484,
failed at the same target. An isolated fresh-reference probe, 39485, found
exactly one changed binary pixel out of 698,368 pixels for
`free_images/pexels-kelly-1179532-28287998_electricguitar`: historical area
346,523 versus current area 346,522, mask IoU 0.9999971142. The input hash,
selection class/index, detection seed, crop geometry and run configuration agree.
The 5x5 reference erosion propagates the difference to five alpha pixels.
The exact source of this numerical drift has not been isolated.

The original strict mismatch guard remains intact. The fresh experiment uses
`output/surface-annotated-20261112-freshrefs/`; the failed attempts remain in
`output/surface-annotated-20261112/`. No old completion is imported into the
fresh comparison and no target is excluded for the mismatch.

The new annotated control generates references under the current pinned
runtime. Its complete cohort is verified and snapshotted, then all cohort
reference files are copied with independent SHA-256 checks into the new
Surface-PMC directory before Surface-PMC starts. Surface-PMC must regenerate
the exact same target masks, and paired analysis requires identical reference
hashes, inputs, model checkpoints, configurations apart from the surface policy,
and installed package inventories. Reference generation does not read quality
metrics or choose between candidate outputs.

The full annotated control likewise finishes and is verified before its
references are copied into the full Surface-PMC continuation. Existing copied
reference bytes cannot be replaced. A successful full control is therefore an
additional dependency for the full Surface-PMC job. Cohort and full results
remain distinct; the complete 2,565-target result is allowed to run beyond the
interactive work window.

This is a documented deviation from the original requirement to reproduce
historical reference bytes. The comparison now estimates the transferred
Surface-PMC method against a fresh annotated-vocabulary baseline in the same
runtime. Historical scores are context, not the matched control. There is no
refitting, outcome-driven threshold change, new training, or weakening of the
paired-reference checks. Native historical/current reference differences will
be enumerated for the completed cohort as an audit diagnostic.
