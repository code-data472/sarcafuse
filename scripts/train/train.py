"""Main training loop for SarcaFuse."""
import argparse, os, sys, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.channels.prg import PRGChannel
from src.channels.llm import LLMChannel
from src.fusion.gated_fusion import CrossChannelGatedFusion
from src.fusion.contrastive_loss import SupervisedContrastiveAlignmentLoss
from src.data.dataset import SarcasmDataset, collate_fn
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics

class SarcaFuseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prg_channel = PRGChannel(config)
        self.llm_channel = LLMChannel(config)
        self.fusion = CrossChannelGatedFusion(
            channel_dim=config.channel_dim, cls_hidden=config.cls_hidden,
            cls_dropout=config.cls_dropout,
        )
    def forward(self, input_ids, attention_mask, texts, cached_llm=None):
        h_prg = self.prg_channel(input_ids, attention_mask, texts)
        h_llm = self.llm_channel(texts, cached_features=cached_llm)
        logits, h_fuse, g = self.fusion(h_prg, h_llm)
        return logits, h_prg, h_llm, h_fuse, g

def train_epoch(model, loader, optimizer, ce_loss, con_loss, config, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        logits, h_prg, h_llm, h_fuse, g = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["text"], batch.get("cached_llm"),
        )
        labels = batch["label"].to(device)
        l_ce = ce_loss(logits, labels)
        l_con = con_loss(h_prg, h_llm, labels)
        loss = l_ce + config.contrastive_weight * l_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SarcaFuseModel(config).to(device)
    tokenizer = model.prg_channel.tokenizer

    train_ds = SarcasmDataset(config.data_train, tokenizer, config.cache_dir, config.max_length)
    val_ds = SarcasmDataset(config.data_val, tokenizer, config.cache_dir, config.max_length)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    ce_loss = nn.CrossEntropyLoss()
    con_loss = SupervisedContrastiveAlignmentLoss(config.contrastive_temperature)

    best_f1 = 0
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, ce_loss, con_loss, config, device)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {train_loss:.4f} | Val F1: {metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, f"{config.dataset_name}_seed{config.seed}_best.pt")
            torch.save(model.state_dict(), path)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    import numpy as np
    all_preds, all_labels = [], []
    for batch in loader:
        logits, *_ = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["text"], batch.get("cached_llm"),
        )
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(batch["label"].numpy())
    return compute_metrics(np.array(all_labels), np.array(all_preds))

if __name__ == "__main__":
    main()
