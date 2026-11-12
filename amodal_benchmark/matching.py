"""Conservative, versioned annotation matching for the SAM-only evaluator.

Aliases affect target identity only. The original detected class remains the
PD-MC prompt, and genuinely different instances are never resolved by score.
"""
import hashlib
import json
import math
import re

import numpy as np

MATCHING_VERSION = 2
_CLASSES = (
    "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|"
    "stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|"
    "backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|"
    "baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|"
    "bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|"
    "potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|"
    "oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|tooth brush|"
    "man|woman|baby"
).split("|")
_ALIASES = {
    "airplane": ("plane", "aeroplane"), "motorcycle": ("motorbike", "motor bike"),
    "couch": ("sofa",), "dining table": ("table",), "tv": ("television", "tv set", "television set"),
    "potted plant": ("plant",), "cell phone": ("phone", "mobile phone", "cellphone", "smartphone"),
    "tooth brush": ("toothbrush",), "hair drier": ("hair dryer", "hairdryer"),
    "refrigerator": ("fridge",), "donut": ("doughnut",), "remote": ("remote control",),
    # Keep toy variants explicit: stripping "toy" generally would turn a
    # "dog toy" into a live dog and lose the annotation's object identity.
    "teddy bear": ("teddybear", "teddy bear toy", "toy bear"),
    "surfboard": ("surfing board",), "snowboard": ("snow board",),
}
_PROTECTED = ("table lamp", "floor lamp", "sea lion")


def _normalize(text):
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


def _plural(phrase):
    prefix, _, word = phrase.rpartition(" ")
    if word in ("sheep", "skis", "scissors", "broccoli"):
        return phrase
    irregular = {"person": "people", "man": "men", "woman": "women", "mouse": "mice", "knife": "knives"}
    if word in irregular:
        word = irregular[word]
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        word += "es"
    elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        word = word[:-1] + "ies"
    else:
        word += "s"
    return (prefix + " " + word).strip()


_CANONICAL = {}
for _class in _CLASSES:
    for _phrase in (_class, *_ALIASES.get(_class, ())):
        for _form in (_phrase, _plural(_phrase)):
            _CANONICAL[_normalize(_form)] = _class
for _phrase in _PROTECTED:
    _CANONICAL[_phrase] = _phrase
    _CANONICAL[_plural(_phrase)] = _phrase

MATCHING_CONFIG = {
    "version": MATCHING_VERSION,
    "aliases_sha256": hashlib.sha256(json.dumps(_CANONICAL, sort_keys=True).encode()).hexdigest(),
    "annotation_rule": "longest-object-suffix-before-spatial-description-v1",
    "deduplication": "exact-binary-mask",
    "duplicate_representative": "highest-detector-score/stable-first-index",
}


def _annotation(label):
    normalized = _normalize(label)
    head = re.split(r"\b(?:on|in|at|under|over|above|below|near|beside|behind|with|without|of)\b",
                    normalized, maxsplit=1)[0].strip()
    endings = [phrase for phrase in _CANONICAL if head == phrase or head.endswith(" " + phrase)]
    phrase = max(endings, key=lambda item: (len(item.split()), len(item))) if endings else None
    return head, _CANONICAL[phrase] if phrase else None


def choose_semantic_target(masks, classes, label, scores=None, diagnostics=None):
    """Select one semantic object region; reject distinct-instance ambiguity."""
    details = diagnostics if diagnostics is not None else {}
    masks = np.asarray([] if masks is None else masks, dtype=bool)
    scores = [0.] * len(classes) if scores is None else [float(score) for score in scores]
    if len(scores) != len(classes) or any(not math.isfinite(score) for score in scores):
        raise ValueError("Detection scores must be finite and match the class count")
    if len(masks) and (masks.ndim != 3 or len(masks) != len(classes)):
        raise ValueError("Detection masks and classes must have matching N,H,W dimensions")
    head, annotation_class = _annotation(label)
    candidates, matching = [], []
    for index, name in enumerate(classes):
        normalized = _normalize(name)
        canonical = _CANONICAL.get(normalized, normalized)
        matched = bool(normalized) and (head == normalized or (
            canonical == annotation_class if annotation_class is not None else head.endswith(" " + normalized)))
        mask = masks[index] if len(masks) else None
        rows, cols = np.where(mask) if mask is not None else ([], [])
        pixels = len(rows) if mask is not None else None
        candidates.append({"index": index, "class": name, "canonical_class": canonical,
            "detector_score": scores[index], "label_matched": bool(matched), "mask_pixels": pixels,
            "bbox_xyxy": [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]
                         if pixels else None})
        if matched and pixels:
            matching.append(index)
    details.update(policy=f"semantic-label-v{MATCHING_VERSION}", annotation_head=head, annotation_class=annotation_class,
                   candidates=candidates, groups=[], duplicate_count=0)
    if not len(masks):
        details["reason"] = "segmentation_unavailable" if classes else "no_detections"
        return None, "not_detected"
    groups = []
    for index in matching:
        for group in groups:
            if np.array_equal(masks[index], masks[group[0]]):
                group.append(index)
                break
        else:
            groups.append([index])
    for group in groups:
        representative = max(group, key=lambda index: (scores[index], -index))
        details["groups"].append({"members": group, "representative": representative})
    details["duplicate_count"] = len(matching) - len(groups)
    if not groups:
        details["reason"] = "label_not_matched" if not any(c["label_matched"] for c in candidates) else "matching_masks_empty"
        return None, "not_detected"
    if len(groups) > 1:
        details["reason"] = "multiple_distinct_instances"
        return None, "ambiguous_target"
    selected = details["groups"][0]["representative"]
    details.update(selected_index=selected, reason="unique_semantic_instance")
    return selected, "semantic_label_match"
