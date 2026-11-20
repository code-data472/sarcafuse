"""
Polarity Reversal Graph (PRG) constructor.
Builds heterogeneous directed graph with adjacency, modifier
and contrast typed edges from annotated token sequences.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
from .lexicon_annotator import TokenAnnotation

EDGE_TYPE_ADJ = 0
EDGE_TYPE_MOD = 1
EDGE_TYPE_CON = 2

class PolarityReversalGraph:
    def __init__(self, modifier_window: int = 5, include_contrast: bool = True, clause_boundary_constraint: bool = False):
        self.modifier_window = modifier_window
        self.include_contrast = include_contrast
        self.clause_boundary_constraint = clause_boundary_constraint
        self._boundary_tokens = {",", ".", ";", "!", "?", "but", "and", "or", "yet"}

    def _has_boundary(self, annotations: List[TokenAnnotation], i: int, j: int) -> bool:
        if not self.clause_boundary_constraint:
            return False
        lo, hi = min(i, j), max(i, j)
        for k in range(lo + 1, hi):
            if annotations[k].token.lower() in self._boundary_tokens:
                return True
        return False

    def build(self, annotations: List[TokenAnnotation], roberta_embeddings: torch.Tensor) -> Dict:
        n = len(annotations)
        node_features = []
        edge_index = [[], []]
        edge_types = []

        for ann in annotations:
            cat_onehot = np.zeros(4, dtype=np.float32)
            cat_map = {"sentiment": 0, "negation": 1, "intensifier": 2, "neutral": 3}
            cat_onehot[cat_map[ann.category]] = 1.0
            feat = np.concatenate([
                roberta_embeddings[ann.index].detach().cpu().numpy(),
                ann.polarity_vec,
                cat_onehot
            ])
            node_features.append(feat)

        # adjacency edges (bidirectional)
        for i in range(n - 1):
            edge_index[0].extend([i, i + 1])
            edge_index[1].extend([i + 1, i])
            edge_types.extend([EDGE_TYPE_ADJ, EDGE_TYPE_ADJ])

        # modifier edges
        mod_cats = {"negation", "intensifier"}
        sent_indices = [a.index for a in annotations if a.category == "sentiment"]
        for ann in annotations:
            if ann.category not in mod_cats:
                continue
            for si in sent_indices:
                if abs(ann.index - si) <= self.modifier_window and ann.index != si:
                    if not self._has_boundary(annotations, ann.index, si):
                        edge_index[0].append(ann.index)
                        edge_index[1].append(si)
                        edge_types.append(EDGE_TYPE_MOD)

        # contrast edges
        if self.include_contrast:
            pos_indices = [a.index for a in annotations if a.category == "sentiment" and np.sum(a.polarity_vec) > 0]
            neg_indices = [a.index for a in annotations if a.category == "sentiment" and np.sum(a.polarity_vec) < 0]
            for pi in pos_indices:
                for ni in neg_indices:
                    src, tgt = (pi, ni) if pi < ni else (ni, pi)
                    edge_index[0].append(src)
                    edge_index[1].append(tgt)
                    edge_types.append(EDGE_TYPE_CON)

        return {
            "node_features": torch.tensor(np.stack(node_features), dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_type": torch.tensor(edge_types, dtype=torch.long),
            "num_nodes": n,
        }

    def edge_stats(self, graph_dict: Dict) -> Dict[str, int]:
        et = graph_dict["edge_type"].numpy()
        return {
            "adjacency": int(np.sum(et == EDGE_TYPE_ADJ)),
            "modifier": int(np.sum(et == EDGE_TYPE_MOD)),
            "contrast": int(np.sum(et == EDGE_TYPE_CON)),
            "total": len(et),
        }
