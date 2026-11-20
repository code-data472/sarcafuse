"""Preprocess SARC Reddit corpus into balanced subset."""
import csv, os, argparse, json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="data/sarc/splits")
    parser.add_argument("--subset_size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    # Loading and sampling logic placeholder
    print(f"SARC preprocessing: seed={args.seed}, subset={args.subset_size}")

if __name__ == "__main__":
    main()
