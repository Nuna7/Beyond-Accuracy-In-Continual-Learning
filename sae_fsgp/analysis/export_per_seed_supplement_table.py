"""
analysis/export_per_seed_supplement_table.py  (patched)

Same as before, but reindexes the per-seed table against the FULL
method x dataset x seed grid before writing output. Seeds where a
method had zero non-surviving concepts (100% survival that seed --
nothing entered the recoverability audit) now appear explicitly as
n_checked=0 rows instead of silently vanishing from a plain groupby.
"""
import argparse
import os
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.aggregate_recoverability_v2 import (
    load_all, _pct_by_seed, _nullctl_by_seed
)


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
    p.add_argument('--out_csv', default='./results_sweep/analysis/recoverability_per_seed.csv')
    p.add_argument('--out_md', default='./results_sweep/analysis/recoverability_per_seed.md')
    args = p.parse_args()

    df_rec, df_null = load_all(args.results_dir, args.datasets, args.seeds,
                                args.methods, args.num_tasks,
                                args.rec_name, args.null_name)

    if df_rec.empty:
        print("No recoverability data found -- nothing to export.")
        return

    per_seed = _pct_by_seed(df_rec)
    null_per_seed = _nullctl_by_seed(df_null)
    if not null_per_seed.empty:
        per_seed = per_seed.merge(null_per_seed,
                                   on=['method', 'dataset', 'seed'], how='left')

    # Reindex against the FULL grid so a (method, dataset, seed) with zero
    # non-surviving concepts appears as an explicit n_checked=0 row instead
    # of silently disappearing from the groupby above.
    full_index = pd.DataFrame(
        list(product(args.methods, args.datasets, args.seeds)),
        columns=['method', 'dataset', 'seed']
    )
    per_seed = full_index.merge(per_seed, on=['method', 'dataset', 'seed'], how='left')
    per_seed['n_checked'] = per_seed['n_checked'].fillna(0).astype(int)

    per_seed = per_seed.sort_values(['dataset', 'method', 'seed'])
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    per_seed.to_csv(args.out_csv, index=False)
    print(f"Per-seed table ({len(per_seed)} rows) -> {args.out_csv}")

    lines = ["| Method | Dataset | Seed | N checked | Recovered % | Decodable % | Lost % |",
             "|---|---|---|---|---|---|---|"]
    for _, r in per_seed.iterrows():
        if r['n_checked'] == 0:
            lines.append(f"| {r['method']} | {r['dataset']} | {int(r['seed'])} | 0 | -- | -- | -- |")
        else:
            lines.append(
                f"| {r['method']} | {r['dataset']} | {int(r['seed'])} | "
                f"{int(r['n_checked'])} | {r['recovered_pct']:.1f} | "
                f"{r['decodable_pct']:.1f} | {r['lost_pct']:.1f} |"
            )
    with open(args.out_md, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown -> {args.out_md}")


if __name__ == '__main__':
    main()