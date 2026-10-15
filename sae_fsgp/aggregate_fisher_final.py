#!/usr/bin/env python3

import json
import glob
import os
import numpy as np
from collections import defaultdict

RESULTS_DIR = "./results_sweep/analysis"

METHODS = [
    "finetune",
    "ewc",
    "lwf",
    "fecam",
    "sae_fsgp",
    "dense_fsgp",
    "efc",
    "dsal",
]

DATASETS = ["cifar100", "tinyimagenet"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


# (dataset, seed, method) -> layer correlations
records = defaultdict(dict)

for dataset in DATASETS:
    for seed in range(5):
        path = os.path.join(
            RESULTS_DIR,
            f"{dataset}_t10_s{seed}",
            "fisher_sae_alignment_multi.json",
        )

        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue

        data = load_json(path)

        for method in METHODS:
            if method not in data:
                print(f"MISSING METHOD: {path} -> {method}")
                continue

            layer_rhos = []
            for layer, stats in data[method].items():
                r = stats.get("spearman_r")
                if r is not None:
                    layer_rhos.append(float(r))

            if layer_rhos:
                records[(dataset, seed, method)] = {
                    "layer_rhos": layer_rhos,
                    "mean_layer_rho": float(np.mean(layer_rhos)),
                }


print("\n=== PER DATASET / METHOD ===")
print(
    f"{'Method':<14}"
    f"{'Dataset':<16}"
    f"{'Seeds':>6}"
    f"{'Mean rho':>12}"
    f"{'Std':>12}"
)
print("-" * 62)

dataset_method_summary = {}

for dataset in DATASETS:
    for method in METHODS:
        vals = [
            records[(dataset, seed, method)]["mean_layer_rho"]
            for seed in range(5)
            if (dataset, seed, method) in records
        ]

        if not vals:
            continue

        mean = np.mean(vals)
        std = np.std(vals)

        dataset_method_summary[(dataset, method)] = (mean, std, len(vals))

        print(
            f"{method:<14}"
            f"{dataset:<16}"
            f"{len(vals):>6}"
            f"{mean:>12.4f}"
            f"{std:>12.4f}"
        )


print("\n=== OVERALL (10 runs = 5 seeds x 2 datasets) ===")
print(f"{'Method':<14}{'N':>6}{'Mean rho':>12}{'Std':>12}")
print("-" * 48)

for method in METHODS:
    vals = [
        records[(dataset, seed, method)]["mean_layer_rho"]
        for dataset in DATASETS
        for seed in range(5)
        if (dataset, seed, method) in records
    ]

    if not vals:
        continue

    print(
        f"{method:<14}"
        f"{len(vals):>6}"
        f"{np.mean(vals):>12.4f}"
        f"{np.std(vals):>12.4f}"
    )


print("\n=== LAYER-SPECIFIC AGGREGATES ===")

for method in METHODS:
    print(f"\n{method}")

    layer_values = defaultdict(list)

    for dataset in DATASETS:
        for seed in range(5):
            path = os.path.join(
                RESULTS_DIR,
                f"{dataset}_t10_s{seed}",
                "fisher_sae_alignment_multi.json",
            )

            if not os.path.exists(path):
                continue

            data = load_json(path)

            if method not in data:
                continue

            for layer, stats in data[method].items():
                r = stats.get("spearman_r")
                if r is not None:
                    layer_values[(dataset, layer)].append(float(r))

    for (dataset, layer), vals in sorted(layer_values.items()):
        print(
            f"  {dataset:<14} {layer:<8} "
            f"rho={np.mean(vals):.4f} +/- {np.std(vals):.4f} "
            f"(n={len(vals)})"
        )