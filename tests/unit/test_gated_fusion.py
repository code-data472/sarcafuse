"""Unit tests for cross-channel gated fusion."""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

class TestGatedFusion:
    def test_output_shapes(self):
        from src.fusion.gated_fusion import CrossChannelGatedFusion
        fusion = CrossChannelGatedFusion(channel_dim=256, num_classes=2)
        h_prg = torch.randn(4, 256)
        h_llm = torch.randn(4, 256)
        logits, h_fuse, g = fusion(h_prg, h_llm)
        assert logits.shape == (4, 2)
        assert h_fuse.shape == (4, 256)
        assert g.shape == (4, 256)
        assert (g >= 0).all() and (g <= 1).all()
