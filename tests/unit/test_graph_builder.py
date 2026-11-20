"""Unit tests for PRG construction."""
import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

class TestPRGBuilder:
    def test_adjacency_edges_count(self):
        from src.channels.prg.graph_builder import PolarityReversalGraph
        from src.channels.prg.lexicon_annotator import TokenAnnotation
        builder = PolarityReversalGraph(modifier_window=5, include_contrast=False)
        anns = [TokenAnnotation(token=f"w{i}", index=i, polarity_vec=np.zeros(3), category="neutral") for i in range(5)]
        emb = torch.randn(5, 768)
        g = builder.build(anns, emb)
        adj_count = (g["edge_type"] == 0).sum().item()
        assert adj_count == 8  # 4 pairs * 2 directions

    def test_modifier_edges_window(self):
        from src.channels.prg.graph_builder import PolarityReversalGraph
        from src.channels.prg.lexicon_annotator import TokenAnnotation
        builder = PolarityReversalGraph(modifier_window=2)
        anns = [
            TokenAnnotation("not", 0, np.zeros(3), "negation"),
            TokenAnnotation("a", 1, np.zeros(3), "neutral"),
            TokenAnnotation("good", 2, np.array([1, 1, 0], dtype=np.float32), "sentiment"),
            TokenAnnotation("day", 3, np.zeros(3), "neutral"),
            TokenAnnotation("bad", 4, np.array([-1, -1, 0], dtype=np.float32), "sentiment"),
        ]
        emb = torch.randn(5, 768)
        g = builder.build(anns, emb)
        mod_count = (g["edge_type"] == 1).sum().item()
        assert mod_count >= 1
