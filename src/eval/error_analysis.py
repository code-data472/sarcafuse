"""Systematic error categorisation for misclassified instances."""
import numpy as np
from collections import Counter

SARCASM_TYPES = ["explicit_contrast", "hyperbolic_praise", "understatement", "situational_irony"]
ERROR_CATEGORIES = ["implicit_no_contrast", "ambiguous_polarity", "context_dependent", "domain_specific"]

class ErrorAnalyzer:
    def __init__(self, annotations_path=None):
        self._type_annotations = {}
        if annotations_path:
            self._load_annotations(annotations_path)

    def _load_annotations(self, path):
        import json
        with open(path) as f:
            self._type_annotations = json.load(f)

    def categorise_errors(self, texts, y_true, y_pred, graph_data=None):
        errors = []
        for i, (t, yt, yp) in enumerate(zip(texts, y_true, y_pred)):
            if yt == yp:
                continue
            cat = self._infer_category(t, graph_data[i] if graph_data else None)
            errors.append({"text": t, "true": yt, "pred": yp, "category": cat})
        return errors

    def _infer_category(self, text, graph_info):
        if graph_info and graph_info.get("contrast_edges", 0) == 0:
            return "implicit_no_contrast"
        return "unknown"

    def summarise(self, errors):
        cats = Counter(e["category"] for e in errors)
        total = len(errors)
        return {k: {"count": v, "pct": v / total * 100} for k, v in cats.most_common()}
