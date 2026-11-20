"""
Type-specific Graph Attention Network encoder.
Applies separate attention parameters per edge type (adj/mod/con).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TypedGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_edge_types=3, num_heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.num_edge_types = num_edge_types
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
            for _ in range(num_edge_types)
        ])
        self.a = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, 2 * out_dim))
            for _ in range(num_edge_types)
        ])
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for w in self.W:
            nn.init.xavier_uniform_(w)
        for a in self.a:
            nn.init.xavier_uniform_(a.unsqueeze(-1))

    def forward(self, x, edge_index, edge_type):
        N = x.size(0)
        h_all = torch.zeros(N, self.num_heads, self.out_dim, device=x.device)
        attn_weights_all = torch.zeros(edge_index.size(1), self.num_heads, device=x.device)

        for t in range(self.num_edge_types):
            mask = (edge_type == t)
            if not mask.any():
                continue
            ei = edge_index[:, mask]
            src, dst = ei[0], ei[1]
            Wt = self.W[t]  # (H, in, out)
            at = self.a[t]  # (H, 2*out)
            h = torch.einsum("ni,hio->nho", x, Wt)  # (N, H, out)
            cat_feat = torch.cat([h[src], h[dst]], dim=-1)  # (E_t, H, 2*out)
            e = (cat_feat * at.unsqueeze(0)).sum(-1)  # (E_t, H)
            e = self.leaky_relu(e)

            # sparse softmax per dst node
            alpha = torch.zeros(N, self.num_heads, device=x.device).scatter_add(
                0, dst.unsqueeze(1).expand_as(e), e.exp()
            )
            alpha_norm = alpha[dst] + 1e-12
            attn = e.exp() / alpha_norm
            attn = self.dropout(attn)

            msg = attn.unsqueeze(-1) * h[src]  # (E_t, H, out)
            h_all.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)
            attn_weights_all[mask] = attn

        out = h_all.view(N, -1)  # (N, H*out)
        return F.elu(out), attn_weights_all


class TypedGATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=256, num_heads=4, num_layers=2, num_edge_types=3, dropout=0.1, residual=True):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else num_heads * hidden_dim
            self.layers.append(TypedGATLayer(d_in, hidden_dim, num_edge_types, num_heads, dropout))
        if residual:
            self.res_proj = nn.ModuleList([
                nn.Linear(in_dim if i == 0 else num_heads * hidden_dim, num_heads * hidden_dim)
                for i in range(num_layers)
            ])
        self.out_proj = nn.Linear(num_heads * hidden_dim, out_dim) if out_dim != num_heads * hidden_dim else nn.Identity()

    def forward(self, x, edge_index, edge_type):
        h = x
        all_attns = []
        for i, layer in enumerate(self.layers):
            h_new, attn = layer(h, edge_index, edge_type)
            if self.residual:
                h_new = h_new + self.res_proj[i](h)
            h = h_new
            all_attns.append(attn)
        return self.out_proj(h), all_attns
