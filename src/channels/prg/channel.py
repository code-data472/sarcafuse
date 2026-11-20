"""End-to-end PRG channel: annotation -> graph -> GAT -> readout."""
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from .lexicon_annotator import MultiLexiconAnnotator
from .graph_builder import PolarityReversalGraph
from .gat_encoder import TypedGATEncoder
from .readout import PolarityWeightedReadout

class PRGChannel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(config.roberta_name)
        self.roberta = RobertaModel.from_pretrained(config.roberta_name)
        for p in self.roberta.parameters():
            p.requires_grad = False

        self.annotator = MultiLexiconAnnotator(
            nrc_path=config.nrc_path, huliu_path=config.huliu_path,
            senticnet_path=config.senticnet_path, negation_path=config.negation_path,
            intensifier_path=config.intensifier_path,
        )
        self.graph_builder = PolarityReversalGraph(
            modifier_window=config.modifier_window,
            include_contrast=config.include_contrast,
            clause_boundary_constraint=config.clause_boundary_constraint,
        )
        node_dim = 768 + 3 + 4
        self.gat = TypedGATEncoder(
            in_dim=node_dim, hidden_dim=config.gat_hidden,
            out_dim=config.channel_dim, num_heads=config.gat_heads,
            num_layers=config.gat_layers, dropout=config.gat_dropout,
        )
        self.readout = PolarityWeightedReadout(config.channel_dim)
        self.proj = nn.Linear(config.channel_dim, config.channel_dim)

    def forward(self, input_ids, attention_mask, token_strings, return_graph=False):
        with torch.no_grad():
            roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = roberta_out.last_hidden_state

        batch_graphs = []
        batch_h = []
        for b in range(input_ids.size(0)):
            tokens = token_strings[b]
            emb = embeddings[b, :len(tokens)]
            annotations = self.annotator.annotate(tokens)
            graph = self.graph_builder.build(annotations, emb)
            node_emb, attns = self.gat(
                graph["node_features"].to(input_ids.device),
                graph["edge_index"].to(input_ids.device),
                graph["edge_type"].to(input_ids.device),
            )
            h = self.readout(node_emb)
            batch_h.append(h.squeeze(0))
            if return_graph:
                batch_graphs.append({"graph": graph, "attns": attns})

        h_prg = torch.stack(batch_h, dim=0)
        h_prg = self.proj(h_prg)
        if return_graph:
            return h_prg, batch_graphs
        return h_prg
