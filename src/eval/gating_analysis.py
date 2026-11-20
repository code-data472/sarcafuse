"""Analysis of gating weights across sarcasm mechanism types."""
import torch
import numpy as np

class GatingAnalyzer:
    def __init__(self):
        pass

    @torch.no_grad()
    def collect_gating_weights(self, model, dataloader, device="cuda"):
        model.eval()
        records = []
        for batch in dataloader:
            _, _, g = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["text"], batch.get("cached_llm"),
            )
            for i in range(g.size(0)):
                records.append({
                    "text": batch["text"][i],
                    "label": batch["label"][i].item(),
                    "mean_gate": g[i].mean().item(),
                })
        return records

    def stratify_by_type(self, records, type_annotations):
        stratified = {}
        for r in records:
            t = type_annotations.get(r["text"], "unknown")
            stratified.setdefault(t, []).append(r["mean_gate"])
        return {k: {"mean": np.mean(v), "std": np.std(v), "n": len(v)} for k, v in stratified.items()}
