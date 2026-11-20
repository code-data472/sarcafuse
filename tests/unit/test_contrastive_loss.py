"""Unit tests for supervised contrastive alignment loss."""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

class TestContrastiveLoss:
    def test_loss_positive(self):
        from src.fusion.contrastive_loss import SupervisedContrastiveAlignmentLoss
        loss_fn = SupervisedContrastiveAlignmentLoss(temperature=0.07)
        h1 = torch.randn(8, 256)
        h2 = torch.randn(8, 256)
        labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
        loss = loss_fn(h1, h2, labels)
        assert loss.item() > 0

    def test_loss_decreases_with_alignment(self):
        from src.fusion.contrastive_loss import SupervisedContrastiveAlignmentLoss
        loss_fn = SupervisedContrastiveAlignmentLoss(temperature=0.07)
        labels = torch.tensor([0, 0, 1, 1])
        h_aligned = torch.randn(4, 64)
        l1 = loss_fn(h_aligned, h_aligned + 0.01 * torch.randn_like(h_aligned), labels)
        h_random = torch.randn(4, 64)
        l2 = loss_fn(h_aligned, h_random, labels)
        assert l1.item() < l2.item()
