"""Unit tests for multi-lexicon annotation."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

class TestTokenAnnotation:
    def test_polarity_vec_shape(self):
        from src.channels.prg.lexicon_annotator import TokenAnnotation
        ann = TokenAnnotation(token="good", index=0, polarity_vec=np.array([1, 1, 1], dtype=np.float32), category="sentiment")
        assert ann.polarity_vec.shape == (3,)

    def test_category_values(self):
        valid = {"sentiment", "negation", "intensifier", "neutral"}
        from src.channels.prg.lexicon_annotator import TokenAnnotation
        for cat in valid:
            ann = TokenAnnotation(token="x", index=0, polarity_vec=np.zeros(3), category=cat)
            assert ann.category in valid
