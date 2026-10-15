#!/usr/bin/env python3
"""
report_results.py
==================
Chat-pasteable status report for the SAE-FSGP continual-learning experiment
tree (train.py output + analysis/*.py output).

Point it at one or more results-root directories (e.g. `results_v2`, `results`)
and it prints a compact, structured summary you can paste straight into chat,
or save with --out / --json_out and upload instead.

Sections (in order):
  1. Accuracy & interference        per method x dataset, seed-by-seed + mean/std
  2. Pipeline completeness          snapshots / posthoc SAE / feature_fate /
                                     drift / separability / report, per run
  3. Feature fate                   survived/drifted/dead % + cos_sim/rank_corr
                                     by layer, seed-averaged
  4. Within-class drift             layer4 (or all layers with --full) cos_sim
                                     / rank_corr per method, from
                                     analysis_report.json / drift_by_layer.json
  5. Inter-class separability       mean_offdiag_sim / worst_nearest_confuser /
                                     effective_rank per method -- the metric
                                     meant to explain accuracy gaps that
                                     within-class drift alone can't
  6. sae_fsgp vs dense_fsgp         head-to-head on every metric above, with
                                     an auto-check of the "high within-class
                                     cos_sim, low accuracy" pattern
  7. EWC/FT <-> SAE alignment       Spearman rho per layer, if computed
  8. [--full] per-task accuracy curves, forgetting-trajectory and
     visual-confirmation qualitative numbers
  9. TODO / gaps                    auto-generated list of what's missing,
                                     partial, or worth re-running

Only needs the standard library. sae_bank.pt files are never opened by
default (they're large torch checkpoints); pass --peek_sae_bank to have the
script torch.load() each one just to report how many (task, layer) SAEs it
holds -- silently skipped if torch isn't importable or the load fails.

Usage
-----
  cd sae_fsgp    # same directory that contains results/ and results_v2/
  python report_results.py                       # scans results_v2 then results
  python report_results.py --results_dir results_v2
  python report_results.py --full --out summary.md --json_out summary.json

Known path quirks in the current analysis/ scripts, so you don't mistake
these for report bugs:
  - forgetting_trajectory.py, visual_confirmation.py and temporal_fate_plot.py
    hardcode "_t10_" in the paths they read/write regardless of --num_tasks.
    If you ever run with num_tasks != 10 those artifacts will show as
    "missing" here even though the step ran fine.
  - visual_confirmation.py saves to
    analysis/visual_confirmation/{dataset}_s{seed}_stats_fixed.json with NO
    method in the filename, so it's overwritten by whichever --method ran
    last for that dataset+seed. The report flags this explicitly.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from datetime import datetime

DEFAULT_METHODS  = ['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp', 'dense_fsgp']
DEFAULT_DATASETS = ['cifar100', 'tinyimagenet']
DEFAULT_SEEDS    = [0, 1, 2]
LAYERS           = ['layer1', 'layer2', 'layer3', 'layer4']


# ───────────────────────────── small utilities ──────────────────────────────

def load_json(path):
    """Returns (data, error_str). Never raises."""
    if not path.exists():
        return None, 'missing'
    try:
        with open(path) as f:
            return json.load(f), None
    except Exception as e:
        return None, f'parse error: {e}'


def fmt(x, nd=4):
    if x is None:
        return 'n/a'
    if isinstance(x, float):
        return f'{x:.{nd}f}'
    return str(x)


def mtime_str(path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    except Exception:
        return 'n/a'


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), pstdev(vals)


class Table:
    """Minimal fixed-width text table -- zero external deps."""
    def __init__(self, headers):
        self.headers = list(headers)
        self.rows = []

    def add(self, *row):
        assert len(row) == len(self.headers), (row, self.headers)
        self.rows.append([str(c) for c in row])

    def render(self):
        widths = [len(h) for h in self.headers]
        for row in self.rows:
            for i, c in enumerate(row):
                widths[i] = max(widths[i], len(c))
        lines = ['  '.join(h.ljust(widths[i]) for i, h in enumerate(self.headers))]
        lines.append('-' * len(lines[0]))
        for row in self.rows:
            lines.append('  '.join(c.ljust(widths[i]) for i, c in enumerate(row)))
        return '\n'.join(lines)


def compute_interference_from_R(R):
    """Mirror analysis/method_compare.py's build_interference_matrix, collapsed
    to a single average positive-forgetting number. R: (T,T) list of lists,
    R[after_task][task_id], None where unmeasured."""
    try:
        T = len(R)
        drops = []
        for i in range(T):
            for j in range(i + 1, T):
                before = R[j - 1][i] if j > 0 and R[j - 1][i] is not None else None
                after  = R[j][i] if R[j][i] is not None else None
                if before is not None and after is not None:
                    drops.append(max(0.0, before - after))
        return mean(drops) if drops else None
    except Exception:
        return None


def peek_sae_bank(path):
    """Best-effort: how many (task, layer) SAE entries a sae_bank.pt holds.
    Returns None on any failure (torch missing, load error, etc.) -- never raises."""
    try:
        import torch
        bank = torch.load(str(path), map_location='cpu', weights_only=False)
        return len(bank)
    except Exception:
        return None


# ───────────────────────────── path helpers ──────────────────────────────

def top_level_run_json(results_dir, method, dataset, T, seed):
    return results_dir / f"{method}_{dataset}_t{T}_s{seed}.json"


def snapshot_dir(results_dir, method, dataset, T, seed):
    return results_dir / 'snapshots' / f"{method}_{dataset}_t{T}_s{seed}"


def analysis_run_dir(results_dir, dataset, T, seed):
    return results_dir / 'analysis' / f"{dataset}_t{T}_s{seed}"


# ───────────────────────────── per-run gathering ──────────────────────────────

def gather_run(results_dir, method, dataset, T, seed, peek_bank=False):
    rec = {'method': method, 'dataset': dataset, 'seed': seed, 'num_tasks': T}

    # 1. top-level accuracy / interference json
    data, _ = load_json(top_level_run_json(results_dir, method, dataset, T, seed))
    rec['top_level_exists'] = data is not None
    if data is not None:
        rec['avg_acc']      = data.get('avg_acc')
        rec['avg_acc_ncm']  = data.get('avg_acc_ncm')
        rec['per_task_acc'] = data.get('per_task_acc')
        if rec['avg_acc'] is None and rec['per_task_acc']:
            rec['avg_acc'] = mean(rec['per_task_acc'])
        rec['interference_from_R'] = (
            compute_interference_from_R(data['R']) if data.get('R') else None)
    else:
        rec['avg_acc'] = rec['avg_acc_ncm'] = rec['per_task_acc'] = None
        rec['interference_from_R'] = None

    # 2. snapshots
    sdir = snapshot_dir(results_dir, method, dataset, T, seed)
    task_files = sorted(sdir.glob('task*.pt')) if sdir.is_dir() else []
    rec['snapshot_dir_exists']  = sdir.is_dir()
    rec['n_snapshots']          = len(task_files)
    rec['last_snapshot_mtime']  = mtime_str(task_files[-1]) if task_files else None

    # 3. analysis/<tag>/<method>/...
    adir = analysis_run_dir(results_dir, dataset, T, seed) / method
    fate_summary, _   = load_json(adir / 'fate_summary.json')
    drift_by_layer, _ = load_json(adir / 'drift_by_layer.json')
    bank_path          = adir / 'posthoc_sae' / 'sae_bank.pt'

    rec['feature_fate']      = fate_summary
    rec['drift_by_layer']    = drift_by_layer
    rec['sae_bank_exists']   = bank_path.exists()
    rec['sae_bank_mtime']    = mtime_str(bank_path) if bank_path.exists() else None
    rec['sae_bank_size_mb']  = (round(bank_path.stat().st_size / 1e6, 1)
                                if bank_path.exists() else None)
    rec['sae_bank_n_entries'] = (peek_sae_bank(bank_path)
                                  if (peek_bank and bank_path.exists()) else None)

    # 4. interclass separability -- one file per dataset/seed, keyed by method
    sep_all, _ = load_json(
        analysis_run_dir(results_dir, dataset, T, seed) / 'separability' /
        f'interclass_separability_seed{seed}.json')
    rec['separability'] = (sep_all or {}).get(method)

    # 5. analysis_report.json -- one file per dataset/seed, has per-method fields
    report, _ = load_json(analysis_run_dir(results_dir, dataset, T, seed) / 'analysis_report.json')
    rec['report_exists']        = report is not None
    rec['report_drift']         = (report or {}).get('drift_comparison', {}).get(method)
    rec['report_interference']  = (report or {}).get('methods', {}).get(method, {}).get('avg_interference')
    rec['interference'] = (rec['report_interference']
                            if rec['report_interference'] is not None
                            else rec['interference_from_R'])

    return rec


def gather_all(results_dir, methods, datasets, seeds, T, peek_bank=False):
    records = {}
    for dataset in datasets:
        records[dataset] = {}
        for method in methods:
            records[dataset][method] = {}
            for seed in seeds:
                records[dataset][method][seed] = gather_run(
                    results_dir, method, dataset, T, seed, peek_bank=peek_bank)
    return records


# ───────────────────────────── report sections ──────────────────────────────

def section_accuracy(records, seeds, T):
    out = []
    for dataset, methods_dict in records.items():
        out.append(f"\n### 1. Accuracy & interference — {dataset} (t{T})\n")
        headers = ['method'] + [f'acc_s{s}' for s in seeds] + \
                  ['acc(mean±std)', 'ncm(mean±std)', 'interf%(mean±std)', 'snapshots']
        tbl = Table(headers)
        for method, seed_dict in methods_dict.items():
            accs   = [seed_dict[s]['avg_acc'] for s in seeds]
            ncms   = [seed_dict[s]['avg_acc_ncm'] for s in seeds]
            interf = [seed_dict[s]['interference'] for s in seeds]
            snaps  = [f"{seed_dict[s]['n_snapshots']}/{T}" for s in seeds]
            acc_m, acc_s = mean_std(accs)
            ncm_m, ncm_s = mean_std(ncms)
            int_m, int_s = mean_std(interf)
            row = [method]
            row += [fmt(a, 2) if a is not None else 'MISSING' for a in accs]
            row += [f"{fmt(acc_m,2)}±{fmt(acc_s,2)}" if acc_m is not None else 'n/a']
            row += [f"{fmt(ncm_m,2)}±{fmt(ncm_s,2)}" if ncm_m is not None else 'n/a']
            row += [f"{fmt(int_m,2)}±{fmt(int_s,2)}" if int_m is not None else 'n/a']
            row += ['/'.join(snaps)]
            tbl.add(*row)
        out.append(tbl.render())
    return '\n'.join(out)


def section_completeness(records, T):
    out = []
    for dataset, methods_dict in records.items():
        out.append(f"\n### 2. Pipeline completeness — {dataset}\n")
        tbl = Table(['method', 'seed', 'snapshots', 'sae_bank', 'feature_fate',
                     'drift', 'separability', 'report'])
        for method, seed_dict in methods_dict.items():
            for seed, rec in seed_dict.items():
                tbl.add(
                    method, seed,
                    f"{rec['n_snapshots']}/{T}",
                    'OK' if rec['sae_bank_exists'] else '--',
                    'OK' if rec['feature_fate'] else '--',
                    'OK' if (rec['report_drift'] or rec['drift_by_layer']) else '--',
                    'OK' if rec['separability'] else '--',
                    'OK' if rec['report_exists'] else '--',
                )
        out.append(tbl.render())
    return '\n'.join(out)


def section_feature_fate(records, seeds):
    out = ["\n### 3. Feature fate (survived / drifted / dead, by layer)\n"]
    any_data = False
    for dataset, methods_dict in records.items():
        for method, seed_dict in methods_dict.items():
            fates = [seed_dict[s]['feature_fate'] for s in seeds if seed_dict[s]['feature_fate']]
            if not fates:
                continue
            any_data = True
            out.append(f"-- {method} / {dataset} (n_seeds={len(fates)}) --")
            tbl = Table(['layer', 'survived%', 'drifted%', 'dead%', 'cos_sim', 'rank_corr'])
            for layer in LAYERS:
                surv = [f[layer]['survived_pct'] for f in fates if layer in f]
                if not surv:
                    continue
                drft = [f[layer]['drifted_pct']  for f in fates if layer in f]
                dead = [f[layer]['dead_pct']     for f in fates if layer in f]
                cos  = [f[layer]['mean_cos_sim'] for f in fates
                        if layer in f and f[layer].get('mean_cos_sim') is not None]
                rank = [f[layer]['mean_rank_corr'] for f in fates
                        if layer in f and f[layer].get('mean_rank_corr') is not None]
                sm, ss = mean_std(surv); dm, ds = mean_std(drft); dem, des = mean_std(dead)
                cm, cs = mean_std(cos);  rm, rs = mean_std(rank)
                tbl.add(layer,
                        f"{fmt(sm,1)}±{fmt(ss,1)}",
                        f"{fmt(dm,1)}±{fmt(ds,1)}",
                        f"{fmt(dem,1)}±{fmt(des,1)}",
                        f"{fmt(cm,3)}±{fmt(cs,3)}" if cm is not None else 'n/a',
                        f"{fmt(rm,3)}±{fmt(rs,3)}" if rm is not None else 'n/a')
            out.append(tbl.render())
    if not any_data:
        out.append('  (no fate_summary.json found yet)')
    return '\n'.join(out)


def section_drift(records, seeds, full=False):
    layers_to_show = LAYERS if full else ['layer4']
    headers = ['method'] + [f'{l}_cos/rank' for l in layers_to_show]
    out = ["\n### 4. Within-class drift, backbone vs final snapshot (cos_sim/rank_corr)\n"]
    for dataset, methods_dict in records.items():
        out.append(f"-- {dataset} --")
        tbl = Table(headers)
        any_data = False
        for method, seed_dict in methods_dict.items():
            drifts = [seed_dict[s]['report_drift'] or seed_dict[s]['drift_by_layer']
                      for s in seeds
                      if (seed_dict[s]['report_drift'] or seed_dict[s]['drift_by_layer'])]
            if not drifts:
                continue
            any_data = True
            row = [method]
            for layer in layers_to_show:
                cos_vals  = [d[layer]['mean_cos_sim']   for d in drifts
                             if layer in d and d[layer].get('mean_cos_sim') is not None]
                rank_vals = [d[layer]['mean_rank_corr'] for d in drifts
                             if layer in d and d[layer].get('mean_rank_corr') is not None]
                cm, _ = mean_std(cos_vals)
                rm, _ = mean_std(rank_vals)
                row.append(f"{fmt(cm,3)}/{fmt(rm,3)}")
            tbl.add(*row)
        out.append(tbl.render() if any_data else '  (no drift data found yet)')
    return '\n'.join(out)


def section_separability(records, seeds):
    out = ["\n### 5. Inter-class separability (final-snapshot NCM prototypes)\n"]
    for dataset, methods_dict in records.items():
        out.append(f"-- {dataset} --")
        tbl = Table(['method', 'mean_offdiag_sim', 'worst_confuser', 'eff_rank', 'nominal_dim', 'n_seeds'])
        any_data = False
        for method, seed_dict in methods_dict.items():
            seps = [seed_dict[s]['separability'] for s in seeds if seed_dict[s]['separability']]
            if not seps:
                continue
            any_data = True
            mo, _ = mean_std([x['mean_offdiag_sim'] for x in seps])
            wc, _ = mean_std([x['worst_nearest_confuser'] for x in seps])
            er, _ = mean_std([x['effective_rank'] for x in seps])
            nd = seps[0].get('nominal_dim')
            tbl.add(method, fmt(mo, 4), fmt(wc, 4), fmt(er, 1), fmt(nd), len(seps))
        out.append(tbl.render() if any_data else
                    '  (no interclass_separability_seed*.json found for this dataset yet)')
    return '\n'.join(out)


def section_head_to_head(records, seeds):
    out = []
    for dataset, methods_dict in records.items():
        if 'sae_fsgp' not in methods_dict or 'dense_fsgp' not in methods_dict:
            continue
        out.append(f"\n### 6. sae_fsgp vs dense_fsgp — {dataset}\n")
        tbl = Table(['metric', 'sae_fsgp', 'dense_fsgp', 'delta(sae-dense)'])

        def collect(method, key_fn):
            vals = [key_fn(methods_dict[method][s]) for s in seeds]
            return mean_std(vals)

        rows = [
            ('avg_acc (linear head)', lambda r: r['avg_acc']),
            ('avg_acc (NCM)',         lambda r: r['avg_acc_ncm']),
            ('L4 within-class cos_sim',  lambda r: ((r['report_drift'] or r['drift_by_layer'] or {})
                                                     .get('layer4', {}).get('mean_cos_sim'))),
            ('L4 within-class rank_corr', lambda r: ((r['report_drift'] or r['drift_by_layer'] or {})
                                                      .get('layer4', {}).get('mean_rank_corr'))),
            ('interclass mean_offdiag_sim (lower=better)',
             lambda r: (r['separability'] or {}).get('mean_offdiag_sim')),
            ('interclass worst_confuser (lower=better)',
             lambda r: (r['separability'] or {}).get('worst_nearest_confuser')),
            ('interclass effective_rank (higher=better)',
             lambda r: (r['separability'] or {}).get('effective_rank')),
        ]
        vals_by_row = {}
        for label, fn in rows:
            sae_m, _   = collect('sae_fsgp', fn)
            dense_m, _ = collect('dense_fsgp', fn)
            delta = (sae_m - dense_m) if (sae_m is not None and dense_m is not None) else None
            vals_by_row[label] = (sae_m, dense_m)
            tbl.add(label, fmt(sae_m, 4), fmt(dense_m, 4), fmt(delta, 4))
        out.append(tbl.render())

        acc_sae   = vals_by_row['avg_acc (NCM)'][0] if vals_by_row['avg_acc (NCM)'][0] is not None else vals_by_row['avg_acc (linear head)'][0]
        acc_dense = vals_by_row['avg_acc (NCM)'][1] if vals_by_row['avg_acc (NCM)'][1] is not None else vals_by_row['avg_acc (linear head)'][1]
        cos_sae   = vals_by_row['L4 within-class cos_sim'][0]
        cos_dense = vals_by_row['L4 within-class cos_sim'][1]
        if None not in (acc_sae, acc_dense, cos_sae, cos_dense):
            if cos_dense > cos_sae and acc_dense < acc_sae:
                out.append(
                    "  >>> Pattern confirmed here: dense_fsgp has HIGHER L4 within-class cos_sim\n"
                    "      but LOWER accuracy than sae_fsgp. Check the separability rows above --\n"
                    "      if dense_fsgp's mean_offdiag_sim/worst_confuser are higher and/or its\n"
                    "      effective_rank is lower, that's the between-class explanation.")
            else:
                out.append("  (dense_fsgp does not show higher L4 cos_sim than sae_fsgp on this "
                            "dataset -- pattern not replicated here.)")
    return '\n'.join(out)


def section_alignment(results_dir, datasets, seeds, T):
    out = []
    for dataset in datasets:
        ewc_rows, ft_rows = [], []
        for seed in seeds:
            report, _ = load_json(analysis_run_dir(results_dir, dataset, T, seed) / 'analysis_report.json')
            if not report:
                continue
            if report.get('ewc_sae_alignment'):
                ewc_rows.append(report['ewc_sae_alignment'])
            if report.get('ft_sae_alignment'):
                ft_rows.append(report['ft_sae_alignment'])
        if not (ewc_rows or ft_rows):
            continue
        out.append(f"\n### 7. EWC/FT <-> SAE-FSGP discriminativity alignment (Spearman rho) — {dataset}\n")
        tbl = Table(['layer', 'ewc_rho(mean)', 'ft_rho(mean)'])
        for layer in LAYERS:
            ewc_vals = [r[layer]['spearman_r'] for r in ewc_rows if layer in r]
            ft_vals  = [r[layer]['spearman_r'] for r in ft_rows  if layer in r]
            if not ewc_vals and not ft_vals:
                continue
            em, _ = mean_std(ewc_vals)
            fm, _ = mean_std(ft_vals)
            tbl.add(layer, fmt(em, 3), fmt(fm, 3))
        out.append(tbl.render())
    return '\n'.join(out)


def section_per_task(records, seeds):
    out = ["\n### 8a. Per-task test-accuracy trajectory (first seed with data)\n"]
    any_data = False
    for dataset, methods_dict in records.items():
        out.append(f"-- {dataset} --")
        for method, seed_dict in methods_dict.items():
            rec = next((seed_dict[s] for s in seeds if seed_dict[s]['per_task_acc']), None)
            if rec is None:
                continue
            any_data = True
            accs = rec['per_task_acc']
            out.append(f"  {method:<12} (seed {rec['seed']}): " + ' '.join(f"{a:5.1f}" for a in accs))
    if not any_data:
        out.append('  (no per_task_acc found in top-level run jsons)')
    return '\n'.join(out)


def section_qualitative(results_dir, datasets, seeds, T, methods):
    out = []
    header_added = False
    for dataset in datasets:
        for method in methods:
            rows = []
            for seed in seeds:
                ft_path = (results_dir / 'analysis' / 'forgetting_trajectory' /
                           f"{method}_{dataset}_t{T}_s{seed}" / f"{method}_{dataset}_s{seed}_fixed.json")
                data, _ = load_json(ft_path)
                if data is None:
                    continue
                tf, ts = data.get('traj_forgotten', {}), data.get('traj_stable', {})
                tf_keys = sorted(int(k) for k in tf.keys())
                ts_keys = sorted(int(k) for k in ts.keys())
                cos_f0 = tf.get(str(tf_keys[0]), {}).get('__cos') if tf_keys else None
                cos_f9 = tf.get(str(tf_keys[-1]), {}).get('__cos') if tf_keys else None
                cos_s0 = ts.get(str(ts_keys[0]), {}).get('__cos') if ts_keys else None
                cos_s9 = ts.get(str(ts_keys[-1]), {}).get('__cos') if ts_keys else None
                rows.append((seed, data.get('forgotten_class'), cos_f0, cos_f9,
                             data.get('stable_class'), cos_s0, cos_s9))
            if not rows:
                continue
            if not header_added:
                out.append("\n### 8b. Forgetting-trajectory qualitative check (per-latent, task-SAE)\n")
                header_added = True
            out.append(f"-- {method} / {dataset} --")
            tbl = Table(['seed', 'forgot_cls', 'cos@t0', 'cos@tN', 'stable_cls', 'cos@t0', 'cos@tN'])
            for seed, fc, cf0, cf9, sc, cs0, cs9 in rows:
                tbl.add(seed, fmt(fc), fmt(cf0, 3), fmt(cf9, 3), fmt(sc), fmt(cs0, 3), fmt(cs9, 3))
            out.append(tbl.render())
    return '\n'.join(out)


def section_visual_confirmation(results_dir, datasets, seeds):
    out = []
    for dataset in datasets:
        rows = []
        for seed in seeds:
            vc_path = results_dir / 'analysis' / 'visual_confirmation' / f"{dataset}_s{seed}_stats_fixed.json"
            data, _ = load_json(vc_path)
            if data is None:
                continue
            f, s = data.get('forgotten', {}), data.get('stable', {})
            rows.append((seed, fmt(f.get('class_id')), fmt(f.get('mean_act_t0'), 3),
                         fmt(f.get('mean_act_t9'), 3), fmt(f.get('act_retained'), 3),
                         fmt(s.get('class_id')), fmt(s.get('mean_act_t0'), 3),
                         fmt(s.get('mean_act_t9'), 3), fmt(s.get('act_retained'), 3)))
        if not rows:
            continue
        out.append(f"\n### 8c. Visual confirmation stats — {dataset}\n"
                    "  NOTE: visual_confirmation.py saves to '<dataset>_s<seed>_stats_fixed.json'\n"
                    "  with no method in the filename, so it's silently overwritten by whichever\n"
                    "  --method ran last for this dataset+seed.\n")
        tbl = Table(['seed', 'forgot_cls', 'f_act@t0', 'f_act@tN', 'f_retained',
                     'stable_cls', 's_act@t0', 's_act@tN', 's_retained'])
        for row in rows:
            tbl.add(*row)
        out.append(tbl.render())
    return '\n'.join(out)


def section_todo(records, T):
    warns = []
    for dataset, methods_dict in records.items():
        for method, seed_dict in methods_dict.items():
            for seed, rec in seed_dict.items():
                tag = f"{method}/{dataset}/seed{seed}"
                if not rec['top_level_exists']:
                    warns.append(f"[MISSING RUN]       {tag}: no top-level result json")
                    continue
                if rec['n_snapshots'] < T:
                    warns.append(f"[PARTIAL SNAPSHOTS] {tag}: {rec['n_snapshots']}/{T} "
                                 f"task checkpoints on disk (last: {rec['last_snapshot_mtime']})")
                if not rec['sae_bank_exists']:
                    warns.append(f"[NO POSTHOC SAE]    {tag}: posthoc_sae/sae_bank.pt not found")
                if not rec['feature_fate']:
                    warns.append(f"[NO FEATURE FATE]   {tag}: fate_summary.json not found")
                if not rec['separability']:
                    warns.append(f"[NO SEPARABILITY]   {tag}: interclass_separability entry not found")
    return warns


# ───────────────────────────── main ──────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_dir', nargs='+', default=['results_v2', 'results'],
                    help='One or more results roots to scan (default: results_v2 results)')
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--methods',  nargs='+', default=DEFAULT_METHODS)
    p.add_argument('--seeds',    nargs='+', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--full', action='store_true',
                   help='show all 4 layers in drift table + per-task curves + '
                        'forgetting-trajectory + visual-confirmation sections')
    p.add_argument('--peek_sae_bank', action='store_true',
                   help='torch.load() each sae_bank.pt just to count entries (slower, needs torch)')
    p.add_argument('--out', default=None, help='also write the report as plain text/markdown here')
    p.add_argument('--json_out', default=None, help='also dump the raw gathered records as JSON here')
    return p.parse_args()


def main():
    args = parse_args()
    all_output = []
    all_records = {}

    for rd in args.results_dir:
        results_dir = Path(rd)
        if not results_dir.is_dir():
            print(f"!! results_dir not found, skipping: {results_dir}", file=sys.stderr)
            continue

        all_output.append(f"\n{'='*78}\nRESULTS ROOT: {results_dir.resolve()}\n{'='*78}")
        records = gather_all(results_dir, args.methods, args.datasets, args.seeds,
                              args.num_tasks, peek_bank=args.peek_sae_bank)
        all_records[str(results_dir)] = records

        all_output.append(section_accuracy(records, args.seeds, args.num_tasks))
        all_output.append(section_completeness(records, args.num_tasks))
        all_output.append(section_feature_fate(records, args.seeds))
        all_output.append(section_drift(records, args.seeds, full=args.full))
        all_output.append(section_separability(records, args.seeds))
        all_output.append(section_head_to_head(records, args.seeds))
        align = section_alignment(results_dir, args.datasets, args.seeds, args.num_tasks)
        if align:
            all_output.append(align)

        if args.full:
            all_output.append(section_per_task(records, args.seeds))
            qual = section_qualitative(results_dir, args.datasets, args.seeds, args.num_tasks, args.methods)
            if qual:
                all_output.append(qual)
            vis = section_visual_confirmation(results_dir, args.datasets, args.seeds)
            if vis:
                all_output.append(vis)

        warns = section_todo(records, args.num_tasks)
        all_output.append(f"\n### 9. TODO / gaps — {results_dir}\n")
        if warns:
            all_output.extend(f"  {w}" for w in warns)
        else:
            all_output.append("  none detected for the requested method/dataset/seed grid.")

    report = '\n'.join(all_output)
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding='utf-8')
        print(f"\n[saved report -> {args.out}]", file=sys.stderr)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(all_records, indent=2, default=str), encoding='utf-8')
        print(f"[saved raw records -> {args.json_out}]", file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        # e.g. `python report_results.py | head` -- exit quietly instead of a traceback
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)