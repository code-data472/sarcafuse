"""Cross-dataset robustness evaluation."""
import argparse, itertools

DATASETS = ["semeval", "isarcasm", "sarc", "ptacek"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", default="logs/cross_dataset.json")
    args = parser.parse_args()
    for train_ds, test_ds in itertools.product(DATASETS, repeat=2):
        if train_ds == test_ds:
            continue
        print(f"Train: {train_ds} -> Test: {test_ds}")

if __name__ == "__main__":
    main()
