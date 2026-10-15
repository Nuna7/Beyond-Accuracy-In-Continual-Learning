"""
print_table.py — aggregate results.csv into a clean comparison table.

Usage:
    python print_table.py --results_dir ./results
    python print_table.py --results_dir ./results --dataset cifar100 --num_tasks 10
"""
import os
import csv
import json
import argparse
import numpy as np
from collections import defaultdict

METHOD_ORDER = ["finetune", "ewc", "lwf", "fecam", "dense_fsgp", "sae_fsgp"]
METHOD_NAMES = {
    "finetune":   "Fine-tuning (LB)",
    "ewc":        "EWC",
    "lwf":        "LwF",
    "fecam":      "FeCAM ★",
    "dense_fsgp": "Dense-FSGP (ablation)",
    "sae_fsgp":   "SAE-FSGP (ours)",
}


def load_results(results_dir: str):
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"No results.csv found in {results_dir}")
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def make_table(rows, dataset: str, num_tasks: int):
    # Group by method, collect avg_acc and forgetting across seeds
    by_method = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['dataset'] != dataset:
            continue
        if int(row['num_tasks']) != num_tasks:
            continue
        m = row['method']
        by_method[m]['avg_acc'].append(float(row['avg_acc']))
        by_method[m]['forgetting'].append(float(row['forgetting']))
        by_method[m]['bwt'].append(float(row['bwt']))

    if not by_method:
        return

    # Header
    title = f"\n  {dataset.upper()}  |  T={num_tasks} tasks"
    sep   = "─" * 72
    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"  {'Method':<30} {'Avg Acc ↑':>12} {'Forgetting ↓':>14} {'BWT ↑':>8}")
    print(sep)

    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        d    = by_method[method]
        n    = len(d['avg_acc'])
        name = METHOD_NAMES.get(method, method)
        acc  = np.mean(d['avg_acc'])
        forg = np.mean(d['forgetting'])
        bwt  = np.mean(d['bwt'])
        std  = np.std(d['avg_acc']) if n > 1 else 0.0

        std_str = f"±{std:.1f}" if n > 1 else f" (n={n})"
        is_ours = "←" if method == "sae_fsgp" else "  "
        print(f"  {name:<30} {acc:>7.2f}{std_str:>5} "
              f"{forg:>14.2f} {bwt:>8.2f}  {is_ours}")

    print(sep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--dataset",     default=None)
    p.add_argument("--num_tasks",   type=int, default=None)
    args = p.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        return

    # Discover all (dataset, num_tasks) combinations present
    combos = sorted({(r['dataset'], int(r['num_tasks'])) for r in rows})
    for ds, T in combos:
        if args.dataset  and ds != args.dataset:
            continue
        if args.num_tasks and T  != args.num_tasks:
            continue
        make_table(rows, ds, T)


if __name__ == "__main__":
    main()
