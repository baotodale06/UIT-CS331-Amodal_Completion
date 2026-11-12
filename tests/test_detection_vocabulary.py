"""CPU checks for added detector categories and text-budget handling."""
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amodal_benchmark.detection import annotation_queries, caption_chunks, expanded_segmenter, query_digest


def tokenizer(caption, **kwargs):
    return {"input_ids": [0, *re.findall(r"\w+|[^\w\s]", caption), 1]}


class DetectionVocabularyTests(unittest.TestCase):
    def test_image_queries_use_annotation_heads_aliases_and_stable_order(self):
        rows = [{"id": str(i), "image": image, "label": label} for i, (image, label) in enumerate([
            ("a", "white plate on a table"), ("b", "television"), ("a", "white plate"),
            ("a", "bathtub"), ("c", "black floor lamp")])]
        expected = {"a": ["bathtub", "white plate"], "c": ["floor lamp"]}
        self.assertEqual(annotation_queries(rows, ["tv", "dining table"]), expected)
        self.assertEqual(annotation_queries(list(reversed(rows)), ["tv", "dining table"]), expected)
        self.assertEqual(query_digest(expected), query_digest(dict(reversed(list(expected.items())))))

    def test_chunks_cover_every_category_without_exceeding_special_token_budget(self):
        vocabulary = ["plate", "tennis racket", "white cabinet", "guitar", "bathtub"]
        chunks = caption_chunks(vocabulary, tokenizer, max_tokens=8)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([word for chunk in chunks for word in chunk], vocabulary)
        for chunk in chunks:
            self.assertLessEqual(len(tokenizer(". ".join(chunk) + ".")["input_ids"]), 8)
        self.assertEqual(caption_chunks(vocabulary, tokenizer), [vocabulary])

    def test_single_overlong_category_and_empty_category_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            caption_chunks(["one two three four five"], tokenizer, 5)
        with self.assertRaisesRegex(ValueError, "empty category"):
            caption_chunks([""], tokenizer)

    def test_added_queries_preserve_returned_phrases_scores_and_all_instances(self):
        calls = []
        def segment(model, sam, checkpoint, pil, tensor, classes, device="cuda"):
            calls.append(list(classes))
            names = ["plate", "plate"] if classes == ["white plate"] else list(classes)
            masks = np.zeros((len(names), pil.height, pil.width), dtype=bool)
            for i in range(len(names)):
                masks[i, i:i+2, i:i+2] = True
            return np.asarray(pil), masks, names, [.7] * len(names)
        expanded = expanded_segmenter(segment, ["cat", "dog"], ["white plate", "bathtub"])
        model = SimpleNamespace(tokenizer=tokenizer, max_text_len=256)
        for size in (12, 18):  # Initial image and a later completion crop.
            image = Image.new("RGB", (size, size))
            rgb, masks, names, scores = expanded(model, None, "sam", image, None, ["unused"])
            self.assertEqual(names, ["cat", "dog", "plate", "plate", "bathtub"])
            self.assertEqual(scores, [.7] * 5)
            self.assertEqual(masks.shape, (5, size, size))
            self.assertEqual(expanded.extra_detected_classes, ["plate", "bathtub"])
            self.assertEqual([t["annotation_query"] for t in expanded.query_trace], [False, True, True])
        self.assertEqual(calls, [["cat", "dog"], ["white plate"], ["bathtub"]] * 2)

    def test_empty_base_detections_do_not_prevent_added_category_detection(self):
        def segment(model, sam, checkpoint, pil, tensor, classes, device="cuda"):
            if classes == ["cat"]:
                return None, None, [], []
            return np.asarray(pil), np.ones((1, pil.height, pil.width)), ["plate"], [.8]
        expanded = expanded_segmenter(segment, ["cat"], ["plate"])
        model = SimpleNamespace(tokenizer=tokenizer)
        result = expanded(model, None, "sam", Image.new("RGB", (12, 12)), None, [])
        self.assertEqual(result[2:], (["plate"], [.8]))

    def test_sam_errors_and_inconsistent_results_are_not_silently_dropped(self):
        model = SimpleNamespace(tokenizer=tokenizer)
        image = Image.new("RGB", (12, 12))
        for masks, names, scores, error in [(None, ["cat"], [.8], RuntimeError),
                                          ([], ["cat"], [.8], ValueError)]:
            def segment(*args, **kwargs):
                return np.asarray(image), masks, names, scores
            with self.assertRaises(error):
                expanded_segmenter(segment, ["cat"], [])(model, None, "sam", image, None, [])


if __name__ == "__main__":
    unittest.main()
