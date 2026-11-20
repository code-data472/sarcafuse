"""Cross-channel gated fusion with classification head."""
import torch
import torch.nn as nn

class CrossChannelGatedFusion(nn.Module):
    def __init__(self, channel_dim=256, num_classes=2, cls_hidden=128, cls_dropout=0.3):
        super().__init__()
        self.gate_proj = nn.Linear(2 * channel_dim, channel_dim)
        self.classifier = nn.Sequential(
            nn.Linear(channel_dim, cls_hidden), nn.ReLU(), nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def forward(self, h_prg, h_llm):
        g = torch.sigmoid(self.gate_proj(torch.cat([h_prg, h_llm], dim=-1)))
        h_fuse = g * h_prg + (1 - g) * h_llm
        logits = self.classifier(h_fuse)
        return logits, h_fuse, g
