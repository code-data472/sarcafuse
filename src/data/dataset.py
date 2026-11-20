"""Dataset class for sarcasm detection benchmarks."""
import torch
from torch.utils.data import Dataset
import json, os, csv

class SarcasmDataset(Dataset):
    def __init__(self, data_path, tokenizer, cached_features_dir=None, max_length=128):
        self.samples = self._load(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cached_dir = cached_features_dir
        self._cache = {}
        if cached_features_dir and os.path.isdir(cached_features_dir):
            self._preload_cache(cached_features_dir)

    def _load(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                samples.append({"text": row["text"], "label": int(row["label"])})
        return samples

    def _preload_cache(self, d):
        for fname in os.listdir(d):
            if fname.endswith(".json"):
                key = fname[:-5]
                with open(os.path.join(d, fname)) as f:
                    self._cache[key] = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(sample["text"], max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": sample["label"],
            "text": sample["text"],
        }
        import hashlib
        h = hashlib.sha256(sample["text"].encode()).hexdigest()[:16]
        if h in self._cache:
            item["cached_llm"] = self._cache[h]
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if k in ("input_ids", "attention_mask"):
            collated[k] = torch.stack([b[k] for b in batch])
        elif k == "label":
            collated[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            collated[k] = [b[k] for b in batch]
    return collated
