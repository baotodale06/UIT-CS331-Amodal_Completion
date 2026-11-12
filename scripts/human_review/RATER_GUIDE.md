# Independent visual review

You will compare an object's visible appearance with a generated completion. There is no preferred outcome. Use only the images and this rubric. Work alone, without AI assistance or discussion of individual cases. Do not read the project report, earlier annotations, example failure lists or model scores until your ratings are submitted. Disclose any earlier involvement or exposure; the coordinator will decide how to describe independence.

## Before starting

Use a desktop or laptop at a comfortable zoom. Keep normal display settings and avoid night-mode color filters. Enter a pseudonymous reviewer code (for example R01). Read this guide, then review cases in the assigned order. Take a break after each 20-case batch. Aim for about 1–2 minutes per case, longer when needed. You may revise your own ratings before submission. Export a JSON backup after each batch. The form sends nothing over the internet.

## What to inspect

1. Read the target description and identify it in the full original image. The visible mask is a detector estimate, not perfect ground truth.
2. Inspect the overview: aligned original context, visible target, returned completion, support map and raw generated RGB. The first four columns have the same crop and scale. The overview is a reduced JPEG preview; use the native images for fine detail.
3. Open the native visible target and returned completion when assessing texture, color and patterns. Their original coordinate frames/canvas sizes may differ due to cropping and outpainting; the overview supplies alignment. Transparent areas are outside the returned mask. Ignore the white display background.
4. Compare the full original with the full raw output before deciding whether a secondary object was introduced. Some generated content lies outside the returned object mask.
5. Rate all six criteria, record confounds, and write a short visual evidence note. A note should say where and what you observed, or why evidence was insufficient. Do not guess a causal mechanism.

The support map uses blue for retained visible support, orange for predicted support outside the initial visible mask, and pink for lost visible support. Orange is **not** a true hidden-region annotation: it can also reflect segmentation errors or outpainting. There is no true hidden RGB image to compare against. Judge consistency with visible evidence, not recovery of unknowable pixels.

## Rating scale (each criterion separately)

| Code | Meaning |
| --- | --- |
| 0 | Assessable; no clear observed defect of this type. This does not certify that the whole output is correct. |
| 1 | Mild, localized inconsistency; overall surface or object identity remains coherent. |
| 2 | Clear, consequential mismatch over a meaningful region. |
| 3 | Extensive/dominant corruption, or a strongly misleading change of identity/content. |
| U | Uncertain or unassessable because of resolution, ambiguity, missing visible evidence or confounds. Explain in the note. |
| NA | Only for pattern: no applicable repeated motif/print/graphic is visible. If a pattern might exist but cannot be assessed, use U. |

Empty means not yet answered. Do not use 0 for uncertainty. Several criteria can be positive in the same case.

## Six criteria

| Criterion | Count | Do not count automatically |
| --- | --- | --- |
| Texture/material | Fur, grain, gloss, material or fine surface detail conflicts with the visible target; substantial unsupported smoothing | Shape-only defects; plausible unseen material differences |
| Color | Unsupported hue, brightness, saturation or color leakage conflicts with visible evidence | Ordinary shading, plausible illumination or reflection |
| Pattern | Stripes, motifs, print, lettering, graphics or repeated key layouts break spacing, scale, direction or continuity | A merely unknown hidden design; physical limb/slat geometry alone |
| Semantic identity | Content becomes another object/part identity or a recognizable hybrid | Missing limbs, holes or distorted shape alone; a final mask selecting another pre-existing object |
| Newly filled region consistency | Newly generated regions conflict with the visible object's material, appearance or part content | Exact hidden-pixel accuracy; merely missing geometry. Use U when a new region cannot be identified or judged |
| Added secondary object | A recognizable unsupported secondary object/accessory appears in raw RGB | A pre-existing scene object, surviving occluder residue, or an ambiguous blob. Compare the FULL source first; use U if origin is unclear |

If an added secondary object is rated 1–3, select its subtype and location: inside/overlapping the returned mask, outside only, or unclear. An outside-only addition is still counted under this raw-output criterion. Use 'other/unclear' for uncertain subtype rather than inventing one.

## Confounds and examples

Mark any mask contamination/undercoverage, target switch, visible-support loss, low resolution/small target, incomplete geometry or ambiguous occluder residue. If none is apparent, explicitly choose 'None observed'. A confound does not force every criterion to U: judge what is assessable.

These are hypothetical rubric examples, not rated study images:

- A red shirt becomes strongly blue where its visible surface was red: assess color severity; do not also mark semantic drift just because its outline changed.
- A chair loses a leg but its wood appearance remains coherent: note incomplete geometry; that alone does not establish an appearance failure or identity change.
- A second cup in the raw output was already in the full original: it is not a newly added secondary object. If you cannot determine whether it was present, use U.

Before starting, make sure you understand: U is not a pass; orange is not true occlusion; the full source is necessary for added-object judgments. Ask the coordinator about rubric wording if needed, without asking them to suggest case ratings.

## Saving and returning

The form autosaves on this browser when supported. Export JSON regularly and before closing; it is the portable backup and includes drafts. Import JSON to resume on another browser/computer. Import replaces the form state only after checking packet identity and asking you to confirm; export current work first.

When every assigned case is complete, download the final JSON and CSV and return both. Do not change anonymized case IDs. If using the blank CSV instead, fill every criterion and provide a reviewer code, evidence, confounds and applicable secondary-object details. Also provide whether you helped develop/select/label outputs and whether you saw prior scores/ratings/conclusions.
