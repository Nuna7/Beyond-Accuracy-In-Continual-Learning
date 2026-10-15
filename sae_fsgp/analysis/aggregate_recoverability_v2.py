"""
analysis/aggregate_recoverability_v2.py

Aggregates the v2 recoverability + null-control outputs
(recoverability_v2.json / recoverability_null_control_v2.json, written by
the corrected analysis/recoverability.py and
analysis/recoverability_null_control.py) across method x dataset x seed
into:

  1. A long-format per-concept CSV (one row per checked class, per task,
     per seed, per method) -- for any further stats you want to run
     yourself.
  2. A per (method, dataset) summary CSV/JSON, averaged across seeds:
     recovered / decodable / lost percentages, mean true vs. null probe
     F1, mean real vs. null-mismatch cosine similarity.
  3. A Markdown table of the same summary, ready to paste into a paper
     appendix.

Does NOT touch training runs, snapshots, SAE banks, or feature_fate.json
-- purely reads the two v2 JSON files this pipeline produces per
(method, dataset, seed) and reduces them. Safe to rerun as many times as
you like; it never writes anywhere except the three output files you
point it at.

Usage:
    python analysis/aggregate_recoverability_v2.py \\
        --results_dir ./results_sweep \\
        --datasets cifar100 tinyimagenet \\
        --seeds 0 1 2 3 4 \\
        --methods finetune ewc lwf fecam sae_fsgp dense_fsgp efc dsal \\
        --num_tasks 10 \\
        --out_csv ./results_sweep/analysis/recoverability_v2_per_concept.csv \\
        --out_summary_csv ./results_sweep/analysis/recoverability_v2_summary.csv \\
        --out_json ./results_sweep/analysis/recoverability_v2_summary.json \\
        --out_md ./results_sweep/analysis/recoverability_v2_summary.md
"""
import os
import sys
import json
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _flatten_recoverability(rec: dict, method: str, dataset: str, seed: int) -> List[dict]:
    rows = []
    for task_key, task_val in rec.items():
        if task_key == '_meta':
            continue
        task_id = int(task_key)
        s = task_val.get('s')
        for class_key, info in task_val.get('per_class', {}).items():
            row = {
                'method': method, 'dataset': dataset, 'seed': seed,
                'task_id': task_id, 's': s, 'class': int(class_key),
                'outcome': info.get('outcome'),
                'cos_after_translation': info.get('cos_after_translation'),
                'probe_f1': info.get('probe_f1'),
                'probe_null_mean': info.get('probe_null_mean'),
                'probe_null_p95': info.get('probe_null_p95'),
                'probe_delta': info.get('probe_delta'),
            }
            rows.append(row)
    return rows


def _flatten_null_control(null: dict, method: str, dataset: str, seed: int) -> List[dict]:
    rows = []
    for task_key, task_val in null.items():
        if task_key == '_meta':
            continue
        task_id = int(task_key)
        s = task_val.get('s')
        for class_key, info in task_val.get('per_class', {}).items():
            rows.append({
                'method': method, 'dataset': dataset, 'seed': seed,
                'task_id': task_id, 's': s, 'class': int(class_key),
                'cos_real': info.get('cos_real'),
                'cos_null_mean': info.get('cos_null_mean'),
                'null_control_delta': info.get('delta'),
            })
    return rows


def load_all(results_dir: str, datasets: List[str], seeds: List[int],
            methods: List[str], num_tasks: int,
            rec_name: str, null_name: str):
    rec_rows, null_rows = [], []
    missing = []
    for dataset in datasets:
        for seed in seeds:
            tag = f'{dataset}_t{num_tasks}_s{seed}'
            for method in methods:
                base = os.path.join(results_dir, 'analysis', tag, method)
                rec = _load_json(os.path.join(base, rec_name))
                null = _load_json(os.path.join(base, null_name))
                if rec is None:
                    missing.append(os.path.join(base, rec_name))
                else:
                    rec_rows.extend(_flatten_recoverability(rec, method, dataset, seed))
                if null is None:
                    missing.append(os.path.join(base, null_name))
                else:
                    null_rows.extend(_flatten_null_control(null, method, dataset, seed))

    if missing:
        print(f"  [WARN] {len(missing)} expected file(s) not found, skipped. "
              f"First few:", file=sys.stderr)
        for m in missing[:10]:
            print(f"    - {m}", file=sys.stderr)

    df_rec = pd.DataFrame(rec_rows)
    df_null = pd.DataFrame(null_rows)
    return df_rec, df_null


def build_per_concept_table(df_rec: pd.DataFrame, df_null: pd.DataFrame) -> pd.DataFrame:
    if df_rec.empty:
        return df_rec
    if df_null.empty:
        return df_rec
    keys = ['method', 'dataset', 'seed', 'task_id', 'class']
    return df_rec.merge(df_null.drop(columns=['s'], errors='ignore'),
                        on=keys, how='left', suffixes=('', '_nullctl'))


def _pct_by_seed(df_rec: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset, seed): counts + percentages of each outcome."""
    def _agg(g):
        n = len(g)
        n_rec = (g['outcome'] == 'recovered').sum()
        n_dec = (g['outcome'] == 'decodable').sum()
        n_lost = (g['outcome'] == 'lost').sum()
        probe_mask = g['outcome'].isin(['decodable', 'lost'])
        return pd.Series({
            'n_checked': n,
            'recovered_pct': 100.0 * n_rec / n if n else np.nan,
            'decodable_pct': 100.0 * n_dec / n if n else np.nan,
            'lost_pct': 100.0 * n_lost / n if n else np.nan,
            'mean_probe_f1_true': g.loc[probe_mask, 'probe_f1'].mean(),
            'mean_probe_null_p95': g.loc[probe_mask, 'probe_null_p95'].mean(),
            'mean_probe_delta': g.loc[probe_mask, 'probe_delta'].mean(),
        })
    return df_rec.groupby(['method', 'dataset', 'seed']).apply(
        _agg, include_groups=False).reset_index()


def _nullctl_by_seed(df_null: pd.DataFrame) -> pd.DataFrame:
    if df_null.empty:
        return df_null
    return df_null.groupby(['method', 'dataset', 'seed']).agg(
        mean_cos_real=('cos_real', 'mean'),
        mean_cos_null=('cos_null_mean', 'mean'),
        mean_nullctl_delta=('null_control_delta', 'mean'),
    ).reset_index()


def build_summary(df_rec: pd.DataFrame, df_null: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset): mean +/- std across seeds of everything in
    _pct_by_seed / _nullctl_by_seed. Averaging per-seed percentages first
    (rather than pooling all concepts across seeds into one big count)
    treats each seed as one independent replicate, which is the right unit
    for a mean +/- std you'd report in a paper."""
    if df_rec.empty:
        return pd.DataFrame()

    per_seed = _pct_by_seed(df_rec)
    null_per_seed = _nullctl_by_seed(df_null)
    if not null_per_seed.empty:
        per_seed = per_seed.merge(null_per_seed, on=['method', 'dataset', 'seed'], how='left')

    value_cols = [c for c in per_seed.columns if c not in ('method', 'dataset', 'seed')]
    grouped = per_seed.groupby(['method', 'dataset'])[value_cols]
    summary = grouped.mean().add_suffix('_mean').join(
        grouped.std().add_suffix('_std')).join(
        grouped.size().rename('n_seeds'))
    summary = summary.reset_index()

    # tidy column order: mean/std pairs together, n_checked/n_seeds last
    ordered_bases = ['recovered_pct', 'decodable_pct', 'lost_pct',
                     'mean_probe_f1_true', 'mean_probe_null_p95', 'mean_probe_delta',
                     'mean_cos_real', 'mean_cos_null', 'mean_nullctl_delta']
    cols = ['method', 'dataset']
    for base in ordered_bases:
        for suf in ('_mean', '_std'):
            c = base + suf
            if c in summary.columns:
                cols.append(c)
    cols += ['n_checked_mean', 'n_seeds']
    cols = [c for c in cols if c in summary.columns]
    return summary[cols]


def write_markdown(summary: pd.DataFrame, out_md: str):
    if summary.empty:
        with open(out_md, 'w') as f:
            f.write("No recoverability_v2 results found.\n")
        return

    cols = ['method', 'dataset', 'n_seeds', 'n_checked_mean',
           'recovered_pct_mean', 'recovered_pct_std',
           'decodable_pct_mean', 'decodable_pct_std',
           'lost_pct_mean', 'lost_pct_std',
           'mean_probe_delta_mean', 'mean_nullctl_delta_mean']
    cols = [c for c in cols if c in summary.columns]
    header = ['Method', 'Dataset', 'Seeds', 'N checked',
             'Recovered %', 'Decodable %', 'Lost %',
             'Probe delta (true-null)', 'Cos delta (real-null)']
    header = header[:len([c for c in cols if c not in
                          ('recovered_pct_std', 'decodable_pct_std', 'lost_pct_std')])]

    lines = []
    lines.append("| Method | Dataset | Seeds | N checked | Recovered % | "
                 "Decodable % | Lost % | Probe delta (true-null) | "
                 "Cos delta (real-null) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        def fmt_pm(mean_c, std_c):
            m = r.get(mean_c, np.nan)
            s = r.get(std_c, np.nan)
            if pd.isna(m):
                return "-"
            if pd.isna(s):
                return f"{m:.1f}"
            return f"{m:.1f} ± {s:.1f}"

        lines.append(
            f"| {r['method']} | {r['dataset']} | {int(r.get('n_seeds', 0))} | "
            f"{r.get('n_checked_mean', float('nan')):.1f} | "
            f"{fmt_pm('recovered_pct_mean', 'recovered_pct_std')} | "
            f"{fmt_pm('decodable_pct_mean', 'decodable_pct_std')} | "
            f"{fmt_pm('lost_pct_mean', 'lost_pct_std')} | "
            f"{r.get('mean_probe_delta_mean', float('nan')):.3f} | "
            f"{r.get('mean_nullctl_delta_mean', float('nan')):.3f} |"
        )
    with open(out_md, 'w') as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results_sweep')
    p.add_argument('--datasets', nargs='+', default=['cifar100', 'tinyimagenet'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--methods', nargs='+',
                   default=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp',
                           'dense_fsgp', 'efc', 'dsal'])
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--rec_name', default='recoverability_v2.json')
    p.add_argument('--null_name', default='recoverability_null_control_v2.json')
    p.add_argument('--out_csv', default='./results_sweep/analysis/recoverability_v2_per_concept.csv')
    p.add_argument('--out_summary_csv', default='./results_sweep/analysis/recoverability_v2_summary.csv')
    p.add_argument('--out_json', default='./results_sweep/analysis/recoverability_v2_summary.json')
    p.add_argument('--out_md', default='./results_sweep/analysis/recoverability_v2_summary.md')
    args = p.parse_args()

    df_rec, df_null = load_all(args.results_dir, args.datasets, args.seeds,
                               args.methods, args.num_tasks,
                               args.rec_name, args.null_name)

    if df_rec.empty:
        print("  [WARN] No recoverability_v2 results found anywhere under "
              f"{args.results_dir} -- nothing to aggregate.", file=sys.stderr)

    per_concept = build_per_concept_table(df_rec, df_null)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    per_concept.to_csv(args.out_csv, index=False)
    print(f"  Per-concept table ({len(per_concept)} rows) -> {args.out_csv}")

    summary = build_summary(df_rec, df_null)
    summary.to_csv(args.out_summary_csv, index=False)
    print(f"  Summary table ({len(summary)} rows) -> {args.out_summary_csv}")

    with open(args.out_json, 'w') as f:
        json.dump(json.loads(summary.to_json(orient='records')), f, indent=2)
    print(f"  Summary JSON -> {args.out_json}")

    write_markdown(summary, args.out_md)
    print(f"  Summary Markdown -> {args.out_md}")


if __name__ == '__main__':
    main()