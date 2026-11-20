"""Hyperparameter sensitivity sweep."""
import argparse, yaml, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config", required=True)
    parser.add_argument("--output_dir", default="logs/sensitivity")
    args = parser.parse_args()
    with open(args.sweep_config) as f:
        sweep = yaml.safe_load(f)
    print(f"Sweeping {sweep['sweep_param']}: {sweep['values']}")

if __name__ == "__main__":
    main()
