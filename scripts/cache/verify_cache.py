"""Verify completeness and validity of cached LLM features."""
import argparse, os, json, hashlib, csv

REQUIRED_KEYS = ["surface_polarity", "situational_polarity", "hyperbole_score", "incongruity_score", "sarcasm_confidence", "reasoning_chain"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    args = parser.parse_args()
    missing, malformed = 0, 0
    total = 0
    with open(args.data_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            h = hashlib.sha256(row["text"].encode()).hexdigest()[:16]
            path = os.path.join(args.cache_dir, f"{h}.json")
            if not os.path.exists(path):
                missing += 1
                continue
            with open(path) as fj:
                data = json.load(fj)
            if not all(k in data for k in REQUIRED_KEYS):
                malformed += 1
    print(f"Total: {total} | Missing: {missing} | Malformed: {malformed} | Complete: {total - missing - malformed}")

if __name__ == "__main__":
    main()
