# Surface consistency as a decision rule: assumptions and testable consequences

Continuation recorded 2026-11-08. These derivations are additional reasoning,
not claims from the PMC paper or empirically proven guarantees. The implemented
study and frozen protocol are in [SURFACE_CONSISTENCY_EXPERIMENT.md](SURFACE_CONSISTENCY_EXPERIMENT.md).
Measured results belong in the accompanying experiment report.

## The quantity a selector actually needs

Let `X` contain observed RGB, the original visible mask, known occlusion and the
query. Let `C = {(I_k, M_k) : k=1,...,K}` be one shared candidate pool. Define the
evaluation error on a **fixed**, unknown-at-inference support `H`:

    e_k = mean_{p in H, channel} |I_k(p) - Y(p)| / 255.

The optimal rule for this particular loss would minimize the conditional risk
`E[e_k | X,C]`. The proposed surface loss is a proxy for that risk. It compares
visible evidence with a candidate's generated surface; it does not observe `Y`.
Weights and normalization make the components comparable numerically. They do
not establish that compatibility is calibrated to reconstruction error.

This separates three questions that are often conflated:

1. Does the generator produce a useful candidate at all?
2. Can the observable score order the candidates **within the same image**?
3. Does the winning decision improve a held-out outcome at the required cost?

A low average surface loss answers none of these by itself. A correlation across
unrelated objects can reflect category or difficulty. Within-pool concordance,
paired selection gains, and oracle headroom address the three questions directly.
The oracle is an evaluation device; its hidden-pixel error cannot be used by the
inference selector.

## An identifiability limit

Suppose two possible objects yield the same observation `X` but different hidden
surfaces `Y_1` and `Y_2` on the same evaluation support. Couple their sampling noise
so the candidate pool is also the same. Any observation-only decision returns the
same reconstruction `A` in both worlds. The triangle inequality gives

    d(A,Y_1) + d(A,Y_2) >= d(Y_1,Y_2),

where `d` is the normalized hidden-region L1 distance. Thus its worst-world error
is at least `d(Y_1,Y_2)/2`; under equal world probabilities, its expected error is
also at least that quantity. The argument holds in expectation for randomized
decisions as well. This is a conditional impossibility example, not a numerical
lower bound estimated for the benchmark.

Examples include a different material continuing behind a seam, an unseen logo,
or the back of a multicolored object. No improvement to histogram matching alone
can remove this ambiguity. Same-surface stationarity is a prior that can be useful
on some images, not a universal law of physical objects.

## An exact decomposition that can be measured

Let `A` be exactly the full-surface-valid candidates. If it is empty, define the
effective set as `{seed 11}` according to the frozen fallback. Let `s` be the
surface-selected candidate, `u` the all-candidate mean error, `u_A` the valid-set
mean, `o` the minimum all-candidate error and `o_A` the valid-set minimum.

    total_gain       = u - e_s
                     = (u - u_A) + (u_A - e_s)
                     = filter_gain + ranking_gain.

    total_regret     = e_s - o
                     = (e_s - o_A) + (o_A - o)
                     = ranking_regret + exclusion_regret.

    total_gain       = (u - o) - total_regret.

The first identity prevents giving the appearance score credit for a validity
filter's benefit. The second shows when guards exclude a good reconstruction.
The third distinguishes weak sampling from weak ranking. These are arithmetic
identities for the saved pool, without a causal assumption about how diffusion
generated the pixels. Filter and ranking gains can both be negative. Regrets are
nonnegative when the selection lies in its effective valid set.

Uniform control is evaluated by its exact expected error, rather than one noisy
random draw. This estimates the policy's mean outcome; it is not a deployable RGB
image formed by averaging the candidates. Every method uses the same generated
images and the same evaluation support.

## A sufficient ranking condition, not an established guarantee

For a particular pool, suppose the observable loss satisfies

    e_k = a L_k + b + delta_k,   a >= 0,   |delta_k| <= epsilon

for every candidate in `A`, with a common `a` and `b`. Because `s` minimizes `L`,

    e_s - o_A <= 2 epsilon,
    e_s - o   <= 2 epsilon + (o_A - o).

Proof: if `j` attains `o_A`, then `L_s <= L_j`, so
`e_s - e_j = a(L_s-L_j) + delta_s-delta_j <= 2 epsilon`.
The second inequality adds the exact exclusion regret.

The experiment does **not** establish a uniform residual bound or estimate this
as a certified guarantee. It explains why a well-scaled score can still fail:
normalization controls numerical units, while a useful residual bound requires
the compatibility prior to predict the actual hidden appearance. An unobserved
material transition, shape change or mask shortcut can violate that assumption.

## Why local evidence is a distinct hypothesis

Global color histograms discard arrangement. Global gradient histograms retain
some orientation and strength information but can still miss pattern phase.
Mean colors on two seam bands can agree even if their corresponding stripes do
not. The constructed counterexamples exercise these failures explicitly.

Pairing nearby observed and generated pixels preserves part of the spatial
relationship. It may improve a repeated surface continuing across the seam.
It still cannot prove correct distant structure, resolve all legitimate material
changes, or identify the correct completion when all candidates are poor. A local
score is therefore a different observable prior with a narrower claim, not a
solution to the identifiability limit.

A useful future model could represent both continuation of the same surface and
a change of material/part, with uncertainty between them. A learned mixture or
abstaining selector would need training data and a fresh evaluation cohort. No
such trained model or improvement is established by this note.

## A direct diagnostic for a wrong prior

After freezing all selections, compare the compatibility of the actual held-out
source RGB and the selected RGB using **the identical original SAM support for
both**. If the erroneous selected RGB has lower surface loss than the source,
the prior can prefer an incorrect reconstruction even without a changing mask.
This is stronger evidence of proxy mismatch than showing that a selected image
has positive pixel error alone.

The diagnostic uses evaluation-only truth and support and can never become an
inference feature. It is post-hoc, does not change a selected seed, and does not
identify the precise visual cause without inspecting the example. Original SAM
support remains a proxy; this does not provide ground truth for naturally hidden
parts of the entire object. The check was declared after pilot test results and
before replication errors were inspected, with its timestamp and source hash
saved in `posthoc_explanation_declaration.json`.

## How the evidence should change the theory

- Retain the primary full-surface result even if a component or local variant
  performs better. The fixed ablations and new explanatory contrasts are
  exploratory; selecting the best test result does not validate a new method.
- Report pilot and replication separately, then the predeclared pooled estimate.
  A pooled benefit alone can conceal a failed replication or source dependence.
- Report ranking gain beyond validity guards, paired comparisons with visible
  preservation, fixed-support pixel error, perceptual diagnostics and failures.
- Treat the historical single-candidate assistant-label audit as a separate
  descriptive test. Cross-image AUC is not within-pool reranking accuracy and is
  not independent human agreement.
- Preserve all frozen test decisions. Any changed scorer requires new validation
  and test data. A well-explained negative or mixed result narrows the hypothesis
  and is a valid research outcome.
