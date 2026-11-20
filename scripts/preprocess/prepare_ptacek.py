"""Preprocess Ptacek Twitter sarcasm corpus."""
import csv, os, argparse
import numpy as np
from src.data.preprocessing import preprocess_tweet

SARCASM_HASHTAGS = {"#sarcasm", "#irony", "#sarcastic", "#not"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", default="data/ptacek/splits")
    parser.add_argument("--subset_size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Ptacek preprocessing: seed={args.seed}, subset={args.subset_size}")

if __name__ == "__main__":
    main()
