"""Opt-in annotation vocabulary expansion for Grounding DINO + SAM.

Keep scene categories for occluder reasoning and query each added annotation
category separately. Never send the full benchmark label list as one caption.
"""
import hashlib
import json

import numpy as np

from .matching import _annotation


DETECTION_CONFIG = {
    "version": 1,
    "mode": "annotated",
    "queries": "base-scene-vocabulary-plus-per-image-missing-annotation-categories",
    "base_chunking": "tokenizer-budget-preserve-order-v1",
    "extra_chunking": "one-category-per-query",
    "max_text_tokens": 256,
    "query_gate": "base-plus-added-categories-and-their-returned-phrases",
    "deduplication": "semantic-matcher-exact-masks-only",
}


def annotation_queries(rows, base_vocabulary):
    """Use all official rows so limits and worker partitioning cannot change queries."""
    base = {word.casefold().strip() for word in base_vocabulary}
    per_image = {}
    for row in rows:
        head, canonical = _annotation(row["label"])
        phrase = canonical or head
        if not phrase:
            raise ValueError(f"Empty annotation detector query for {row['id']}")
        if phrase not in base:
            per_image.setdefault(row["image"], set()).add(phrase)
    return {image: sorted(phrases) for image, phrases in sorted(per_image.items())}


def query_digest(queries):
    return hashlib.sha256(json.dumps(queries, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def caption_chunks(vocabulary, tokenizer, max_tokens=256):
    """Count the exact caption including punctuation and special tokens."""
    chunks, pending = [], []

    def fits(words):
        caption = ". ".join(words).lower().strip()
        if not caption.endswith("."):
            caption += "."
        return len(tokenizer(caption, truncation=False)["input_ids"]) <= max_tokens

    for word in vocabulary:
        if not word.strip():
            raise ValueError("Detection vocabulary contains an empty category")
        if fits(pending + [word]):
            pending.append(word)
        else:
            if not pending or not fits([word]):
                raise ValueError(f"Detector category exceeds {max_tokens} text tokens: {word}")
            chunks.append(pending)
            pending = [word]
    if pending:
        chunks.append(pending)
    return chunks


def expanded_segmenter(segment, base_vocabulary, extra_queries):
    """Wrap initial and completion segmentation with the same image vocabulary.

    Returned detector phrases and scores stay unchanged. Phrases returned by
    added queries join the eligibility vocabulary, avoiding rejection of a
    valid shorter phrase such as 'plate' returned for 'white plate'.
    """
    def expanded(gdino, sam, checkpoint, pil, tensor, classes, device="cuda"):
        limit = min(DETECTION_CONFIG["max_text_tokens"], int(getattr(gdino, "max_text_len", 256)))
        base_chunks = caption_chunks(base_vocabulary, gdino.tokenizer, limit)
        extra_chunks = [chunk for phrase in extra_queries
                        for chunk in caption_chunks([phrase], gdino.tokenizer, limit)]
        masks, names, scores, extra_names, trace = [], [], [], [], []
        rgb = np.asarray(pil).copy()
        for index, chunk in enumerate(base_chunks + extra_chunks):
            image, detected_masks, detected_names, detected_scores = segment(
                gdino, sam, checkpoint, pil, tensor, chunk, device=device)
            if detected_masks is None:
                if detected_names:
                    raise RuntimeError("SAM returned no masks for an expanded-vocabulary detector query")
                detected_masks = []
            if len(detected_masks) != len(detected_names) or len(detected_names) != len(detected_scores):
                raise ValueError("Expanded detector masks, classes and scores do not agree")
            if image is not None:
                rgb = image
            if len(detected_masks):
                masks.extend(detected_masks)
                names.extend(detected_names)
                scores.extend(detected_scores)
            added = index >= len(base_chunks)
            if added:
                extra_names.extend(detected_names)
            trace.append({"categories": list(chunk), "annotation_query": added,
                          "returned_phrases": list(detected_names), "detections": len(detected_names)})
        expanded.extra_detected_classes = list(dict.fromkeys(extra_names))
        expanded.query_trace = trace
        array = np.asarray(masks, dtype=bool) if masks else np.empty((0, pil.height, pil.width), dtype=bool)
        return rgb, array, names, scores

    expanded.extra_detected_classes = []
    expanded.query_trace = []
    return expanded
