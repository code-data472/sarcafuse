"""Encode LLM reasoning outputs into fixed-dim incongruity vectors."""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class IncongruityFeatureEncoder(nn.Module):
    def __init__(self, channel_dim=256, sbert_name="all-MiniLM-L12-v2", sbert_dim=384):
        super().__init__()
        self.num_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, channel_dim // 2), nn.ReLU(),
        )
        self.chain_proj = nn.Linear(sbert_dim, channel_dim // 2)
        self.sbert = SentenceTransformer(sbert_name)
        for p in self.sbert.parameters():
            p.requires_grad = False

    def forward(self, numerical_scores, reasoning_chains):
        h_num = self.num_mlp(numerical_scores)
        with torch.no_grad():
            chain_embs = self.sbert.encode(reasoning_chains, convert_to_tensor=True, show_progress_bar=False)
        h_chain = self.chain_proj(chain_embs.to(h_num.device))
        return torch.cat([h_num, h_chain], dim=-1)
