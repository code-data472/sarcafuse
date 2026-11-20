"""Evaluate trained SarcaFuse model on test set."""
import argparse, os, sys, json
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.config import load_config
from src.utils.metrics import compute_metrics, bootstrap_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Evaluating {args.checkpoint} on {config.dataset_name}")

if __name__ == "__main__":
    main()
