#!/usr/bin/env python3
"""
compute_acc1_T.py -- retrospective literature-standard metrics from existing
result JSONs. No rerun needed: reads the R matrix every train.py run already
saves in results_sweep/{method}_{dataset}_t{T}_s{seed}.json.

Computes, per (method, dataset, seed):
  - AccT   : R[T-1, :] mean -- same as your existing 'avg_acc' / SUMMARY.md column.
  - Acc1:T : mean over steps k=0..T-1 of (mean of R[k, 0:k+1]) -- "average
             incremental accuracy" (EFC Eq. 16, DS-AL's A-bar), what published
             tables report. VERIFIED against a synthetic R matrix (see chat).

NCM-based Acc1:T is NOT retrofittable: train.py only saves the FINAL-checkpoint
NCM row (R_ncm), not per-step NCM matrices. That specific combination would
need a rerun with per-step NCM eval added -- not offering a fake shortcut here.
"""
import argparse, json, glob, os
import numpy as np
from collections import defaultdict


def compute_acc1_T(R):
    R = np.array(R, dtype=float)
    T = R.shape[0]
    step_accs = []
    for k in range(T):
        row = R[k, :k + 1]
        row = row[~np.isnan(row)]
        if len(row) == 0:
            continue
        step_accs.append(row.mean())
    return float(np.mean(step_accs)) if step_accs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='./results_sweep')
    args = ap.parse_args()

    rows = defaultdict(list)
    for path in glob.glob(os.path.join(args.results_dir, '*_t*_s*.json')):
        with open(path) as f:
            d = json.load(f)
        if 'R' not in d:
            continue
        method, dataset, seed = d.get('method'), d.get('dataset'), d.get('seed')
        AccT = float(np.mean(d['per_task_acc'])) if d.get('per_task_acc') else float('nan')
        Acc1T = compute_acc1_T(d['R'])
        rows[(method, dataset)].append((seed, AccT, Acc1T))

    print(f"{'Method':<14}{'Dataset':<14}{'n':>3}  {'AccT (mean+/-std)':<22}{'Acc1:T (mean+/-std)':<22}")
    print("-" * 78)
    for (method, dataset), vals in sorted(rows.items()):
        accts = np.array([v[1] for v in vals])
        acc1ts = np.array([v[2] for v in vals])
        print(f"{method:<14}{dataset:<14}{len(vals):>3}  "
              f"{accts.mean():>6.2f}+/-{accts.std():<9.2f}"
              f"{acc1ts.mean():>6.2f}+/-{acc1ts.std():<9.2f}")


if __name__ == '__main__':
    main()