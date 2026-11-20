"""Preprocess iSarcasm dataset."""
import csv, os, argparse, json
import numpy as np
from src.data.preprocessing import preprocess_tweet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", default="data/isarcasm/splits")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    samples = []
    with open(args.input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({"text": preprocess_tweet(row["tweet"]), "label": int(row["sarcasm_label"])})

    n = len(samples)
    idx = rng.permutation(n)
    split_80 = int(0.8 * n)
    split_90 = int(0.9 * n) if args.val_ratio == 0.1 else split_80
    train_idx, val_idx, test_idx = idx[:split_80], idx[split_80:split_90], idx[int(0.8*n):]

    for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        path = os.path.join(args.output_dir, f"{name}.tsv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["text", "label"])
            for i in indices:
                writer.writerow([samples[i]["text"], samples[i]["label"]])

if __name__ == "__main__":
    main()
