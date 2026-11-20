"""Main evaluation loop with multi-seed aggregation."""
import torch
import numpy as np
from ..utils.metrics import compute_metrics, bootstrap_test

class Evaluator:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in dataloader:
            logits, _, _ = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["text"], batch.get("cached_llm"),
            )
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
        return compute_metrics(np.array(all_labels), np.array(all_preds))

    def evaluate_multiseed(self, dataloader, seeds, checkpoint_pattern):
        results = []
        for seed in seeds:
            ckpt = checkpoint_pattern.format(seed=seed)
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            r = self.evaluate(dataloader)
            results.append(r)
        agg = {}
        for key in results[0]:
            vals = [r[key] for r in results]
            if isinstance(vals[0], float):
                agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}
            else:
                agg[key] = vals
        return agg
