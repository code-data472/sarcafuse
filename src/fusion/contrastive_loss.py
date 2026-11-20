"""Supervised contrastive alignment loss across channels."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_prg, h_llm, labels):
        B = h_prg.size(0)
        h_prg_n = F.normalize(h_prg, dim=-1)
        h_llm_n = F.normalize(h_llm, dim=-1)
        loss = 0.0
        for h_a, h_b in [(h_prg_n, h_llm_n), (h_llm_n, h_prg_n)]:
            sim = torch.mm(h_a, h_b.t()) / self.temperature  # (B, B)
            mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            mask_pos.fill_diagonal_(0)
            mask_neg = 1 - mask_pos
            mask_neg.fill_diagonal_(0)
            for i in range(B):
                pos_idx = mask_pos[i].nonzero(as_tuple=True)[0]
                if len(pos_idx) == 0:
                    continue
                denom = torch.logsumexp(sim[i][mask_pos[i].bool() | mask_neg[i].bool()], dim=0)
                numer = sim[i][pos_idx]
                loss += -(numer - denom).mean()
        return loss / (2 * B)
