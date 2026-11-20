"""Configuration loading and validation."""
import yaml
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SarcaFuseConfig:
    # PRG channel
    roberta_name: str = "roberta-base"
    nrc_path: str = "lexicons/nrc/NRC-Emotion-Lexicon.txt"
    huliu_path: str = "lexicons/huliu"
    senticnet_path: str = "lexicons/senticnet/senticnet7.txt"
    negation_path: str = "lexicons/lists/negation_markers.txt"
    intensifier_path: str = "lexicons/lists/intensifiers.txt"
    modifier_window: int = 5
    include_contrast: bool = True
    clause_boundary_constraint: bool = False
    gat_heads: int = 4
    gat_hidden: int = 64
    gat_layers: int = 2
    gat_dropout: float = 0.1
    # LLM channel
    llm_model: str = "gpt-4o-2024-05-13"
    llm_temperature: float = 0.1
    llm_top_p: float = 1.0
    llm_max_tokens: int = 512
    sbert_name: str = "all-MiniLM-L12-v2"
    template_path: str = "prompts/templates/cot_incongruity_v3.txt"
    exemplar_path: str = "prompts/exemplars/three_shot.json"
    schema_path: str = "prompts/schemas/incongruity_output.json"
    cache_dir: str = "cached_features/semeval/gpt4o"
    # Fusion
    channel_dim: int = 256
    cls_hidden: int = 128
    cls_dropout: float = 0.3
    contrastive_weight: float = 0.3
    contrastive_temperature: float = 0.07
    # Training
    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 30
    batch_size: int = 64
    warmup_ratio: float = 0.1
    max_length: int = 128
    seed: int = 0

def load_config(path: str) -> SarcaFuseConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    return SarcaFuseConfig(**{k: v for k, v in d.items() if hasattr(SarcaFuseConfig, k)})
