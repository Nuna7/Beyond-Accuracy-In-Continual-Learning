# /media/20TB/vision/CL/sae_fsgp/analysis/collect_final.py
import json, os, argparse, glob
from pathlib import Path

def load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='./results')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    T=10
    seed=args.seed
    R=args.results_dir

    def result_path(method, dataset):
        return os.path.join(R, f"{method}_{dataset}_t{T}_s{seed}.json")

    def analysis_report(dataset):
        return os.path.join(R, 'analysis', f"{dataset}_t{T}_s{seed}", "analysis_report.json")

    def fate_summary(dataset, method):
        return os.path.join(R, 'analysis', f"{dataset}_t{T}_s{seed}", method, "fate_summary.json")

    def fate_full(dataset, method):
        return os.path.join(R, 'analysis', f"{dataset}_t{T}_s{seed}", method, "feature_fate.json")

    def separability(dataset):
        return os.path.join(R, 'analysis', f"{dataset}_t{T}_s{seed}", "separability", f"interclass_separability_seed{seed}.json")

    final = {
        "seed": seed,
        "num_tasks": T,
        "train": {
            "dense_fsgp_tinyimagenet": load(result_path("dense_fsgp","tinyimagenet")),
        },
        "mechanistic_analysis_tinyimagenet": load(analysis_report("tinyimagenet")),
        "feature_fate": {
            "sae_fsgp_tinyimagenet_summary": load(fate_summary("tinyimagenet","sae_fsgp")),
            "dense_fsgp_tinyimagenet_summary": load(fate_summary("tinyimagenet","dense_fsgp")),
            "sae_fsgp_tinyimagenet_full": load(fate_full("tinyimagenet","sae_fsgp")),
            "dense_fsgp_tinyimagenet_full": load(fate_full("tinyimagenet","dense_fsgp")),
        },
        "interclass_separability": {
            "cifar100": load(separability("cifar100")),
            "tinyimagenet": load(separability("tinyimagenet")),
        }
    }

    out = args.out or os.path.join(R, f"FINAL_AGGREGATED_tiny{T}_s{seed}.json")
    with open(out, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"Saved single file -> {out}")

if __name__ == "__main__":
    main()