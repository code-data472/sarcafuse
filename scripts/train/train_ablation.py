"""Run ablation experiments across configurations."""
import argparse, subprocess, json, os

ABLATIONS = {
    "full": {},
    "no_prg": {"disable_prg": True},
    "no_llm": {"disable_llm": True},
    "no_contrast_loss": {"contrastive_weight": 0.0},
    "no_type_edge": {"homogeneous_edges": True},
    "single_lexicon": {"single_lexicon": "nrc"},
    "no_graph_struct": {"mean_pool_only": True},
    "no_contrast_edge": {"include_contrast": False},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_dir", default="logs/ablation")
    args = parser.parse_args()

    for name, overrides in ABLATIONS.items():
        for seed in args.seeds:
            cmd = ["python", "scripts/train/train.py", "--config", args.config, "--seed", str(seed)]
            for k, v in overrides.items():
                cmd.extend([f"--{k}", str(v)])
            print(f"[{name}] seed={seed}: {' '.join(cmd)}")
            subprocess.run(cmd)

if __name__ == "__main__":
    main()
