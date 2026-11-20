"""Polarity-weighted graph readout with learnable query."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolarityWeightedReadout(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(node_dim))

    def forward(self, node_embeddings, batch_index=None):
        scores = (node_embeddings * self.query.unsqueeze(0)).sum(-1)
        if batch_index is not None:
            from torch_scatter import scatter_softmax, scatter_sum
            weights = scatter_softmax(scores, batch_index, dim=0)
            return scatter_sum(weights.unsqueeze(-1) * node_embeddings, batch_index, dim=0)
        else:
            weights = F.softmax(scores, dim=0)
            return (weights.unsqueeze(-1) * node_embeddings).sum(0, keepdim=True)
