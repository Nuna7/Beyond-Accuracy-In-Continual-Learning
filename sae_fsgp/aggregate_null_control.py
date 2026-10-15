"""
analysis/aggregate_null_control.py

Aggregate recoverability_null_control.json results across methods / datasets / seeds.

Produces:
  - per (dataset, method) summary: mean cos_real, mean cos_null, mean delta
    (± std across seeds, pooling all tasks & classes)
  - optional per-task breakdown
  - overall real-vs-null comparison (is recovery selective?)
  - CSV + pretty printed tables

Usage:
  python analysis/aggregate_null_control.py \
      --results_dir /media/20TB/vision/CL/sae_fsgp/results_sweep \
      --num_tasks 10
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

def load_all(
    results_dir: str,
    num_tasks: int = 10,
    methods: List[str] | None = None,
    datasets: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Walk analysis/ and collect every recoverability_null_control.json."""
    root = Path(results_dir) / "analysis"
    if not root.exists():
        raise FileNotFoundError(f"No analysis dir: {root}")

    records = []
    for path in sorted(root.rglob("recoverability_null_control.json")):
        # expected layout: analysis/{dataset}_t{T}_s{seed}/{method}/recoverability_null_control.json
        method = path.parent.name
        run_dir = path.parent.parent.name  # e.g. cifar100_t10_s0
        try:
            dataset, rest = run_dir.split("_t", 1)
            t_str, s_str = rest.split("_s")
            t = int(t_str)
            seed = int(s_str)
        except Exception:
            print(f"[warn] could not parse path: {path}")
            continue

        if t != num_tasks:
            continue
        if methods is not None and method not in methods:
            continue
        if datasets is not None and dataset not in datasets:
            continue

        with open(path) as f:
            data = json.load(f)

        # flatten every (task, class) entry
        for task_str, task_blob in data.items():
            task_id = int(task_str)
            s = task_blob.get("s")
            for cls_str, metrics in task_blob.get("per_class", {}).items():
                records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "task_id": task_id,
                        "s": s,
                        "class": int(cls_str),
                        "cos_real": float(metrics["cos_real"]),
                        "cos_null_mean": float(metrics["cos_null_mean"]),
                        "delta": float(metrics["delta"]),
                        "cos_null_vals": metrics.get("cos_null_vals", []),
                        "path": str(path),
                    }
                )
    return records

def _agg(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }

def summarize(
    records: List[Dict[str, Any]],
    group_by: Tuple[str, ...] = ("dataset", "method"),
) -> Dict[Tuple, Dict[str, Any]]:
    """Group records and compute mean/std of cos_real, cos_null_mean, delta."""
    buckets: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in records:
        key = tuple(r[k] for k in group_by)
        buckets[key].append(r)

    out = {}
    for key, rows in sorted(buckets.items()):
        cos_real = [r["cos_real"] for r in rows]
        cos_null = [r["cos_null_mean"] for r in rows]
        delta = [r["delta"] for r in rows]
        # fraction of concepts where real > null (selective recovery)
        selective = sum(1 for d in delta if d > 0) / len(delta) if delta else float("nan")

        out[key] = {
            "n_concepts": len(rows),
            "n_seeds": len({r["seed"] for r in rows}),
            "n_tasks": len({r["task_id"] for r in rows}),
            "cos_real": _agg(cos_real),
            "cos_null": _agg(cos_null),
            "delta": _agg(delta),
            "frac_selective": selective,  # P(delta > 0)
        }
    return out

def print_table(summary: Dict[Tuple, Dict[str, Any]], group_by: Tuple[str, ...]):
    """Pretty fixed-width table."""
    headers = list(group_by) + [
        "n", "seeds", "tasks",
        "real±std", "null±std", "delta±std", "P(δ>0)",
    ]
    rows = []
    for key, s in summary.items():
        rows.append(
            list(key)
            + [
                s["n_concepts"],
                s["n_seeds"],
                s["n_tasks"],
                f"{s['cos_real']['mean']:.3f}±{s['cos_real']['std']:.3f}",
                f"{s['cos_null']['mean']:.3f}±{s['cos_null']['std']:.3f}",
                f"{s['delta']['mean']:+.3f}±{s['delta']['std']:.3f}",
                f"{s['frac_selective']:.2f}",
            ]
        )

    # column widths
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)

    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt.format(*r))

def to_csv(summary: Dict[Tuple, Dict[str, Any]], group_by: Tuple[str, ...], path: str):
    import csv
    fieldnames = list(group_by) + [
        "n_concepts", "n_seeds", "n_tasks",
        "cos_real_mean", "cos_real_std",
        "cos_null_mean", "cos_null_std",
        "delta_mean", "delta_std",
        "frac_selective",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key, s in summary.items():
            row = {k: v for k, v in zip(group_by, key)}
            row.update(
                {
                    "n_concepts": s["n_concepts"],
                    "n_seeds": s["n_seeds"],
                    "n_tasks": s["n_tasks"],
                    "cos_real_mean": s["cos_real"]["mean"],
                    "cos_real_std": s["cos_real"]["std"],
                    "cos_null_mean": s["cos_null"]["mean"],
                    "cos_null_std": s["cos_null"]["std"],
                    "delta_mean": s["delta"]["mean"],
                    "delta_std": s["delta"]["std"],
                    "frac_selective": s["frac_selective"],
                }
            )
            w.writerow(row)
    print(f"Wrote {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results_sweep")
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--group_by", nargs="+", default=["dataset", "method"],
                   help="fields to group by (default: dataset method)")
    p.add_argument("--by_task", action="store_true",
                   help="also print a dataset×method×task breakdown")
    p.add_argument("--out_csv", default=None,
                   help="optional path to write summary CSV")
    args = p.parse_args()

    records = load_all(
        args.results_dir,
        num_tasks=args.num_tasks,
        methods=args.methods,
        datasets=args.datasets,
    )
    print(f"Loaded {len(records)} concept-level records from "
          f"{len({r['path'] for r in records})} json files\n")

    if not records:
        print("Nothing to aggregate.")
        return

    group_by = tuple(args.group_by)
    summary = summarize(records, group_by=group_by)
    print(f"=== Aggregate by {group_by} ===")
    print_table(summary, group_by)

    if args.by_task:
        print("\n=== Aggregate by (dataset, method, task_id) ===")
        summary_t = summarize(records, group_by=("dataset", "method", "task_id"))
        print_table(summary_t, ("dataset", "method", "task_id"))

    # quick global real-vs-null check
    all_delta = [r["delta"] for r in records]
    all_real = [r["cos_real"] for r in records]
    all_null = [r["cos_null_mean"] for r in records]
    print("\n=== Global (all methods / datasets / seeds) ===")
    print(f"  n_concepts     = {len(all_delta)}")
    print(f"  mean cos_real  = {np.mean(all_real):.4f} ± {np.std(all_real, ddof=1):.4f}")
    print(f"  mean cos_null  = {np.mean(all_null):.4f} ± {np.std(all_null, ddof=1):.4f}")
    print(f"  mean delta     = {np.mean(all_delta):+.4f} ± {np.std(all_delta, ddof=1):.4f}")
    print(f"  P(delta > 0)   = {np.mean(np.asarray(all_delta) > 0):.3f}")

    if args.out_csv:
        to_csv(summary, group_by, args.out_csv)

if __name__ == "__main__":
    main()
