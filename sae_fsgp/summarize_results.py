#!/usr/bin/env python3
"""
summarize_results.py

Walks a results directory produced by run_full_pipeline.sh / train.py /
analysis/*.py and produces one consolidated, paste-back-to-Claude report.

It is schema-aware for everything I have full visibility into (train.py's
result JSON, analysis/run_analysis.py's analysis_report.json, feature_fate's
fate_summary.json, interclass_separability's output) and schema-agnostic
(auto-discovers extra numeric fields) for anything produced by config.py /
utils/metrics.py, which weren't part of what I reviewed.

Usage:
    python summarize_results.py --results_dir ./results_sweep
    python summarize_results.py --results_dir ./results_sweep --save SUMMARY.md
"""
import re
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

RESULT_RE = re.compile(
    r'^(?P<method>.+)_(?P<dataset>cifar100|tinyimagenet)_t(?P<num_tasks>\d+)_s(?P<seed>\d+)\.json$'
)

KNOWN_METHODS_ORDER = ['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp', 'dense_fsgp', 'ada_gauss', 'dsal', 'efc']
SAE_METHODS = ['sae_fsgp', 'dense_fsgp']

KNOWN_TOP_LEVEL_KEYS = {
    'method', 'dataset', 'num_tasks', 'seed', 'per_task_acc', 'per_task_acc_ncm',
    'avg_acc_ncm', 'R', 'R_ncm', 'avg_acc',
}


def method_sort_key(m):
    return KNOWN_METHODS_ORDER.index(m) if m in KNOWN_METHODS_ORDER else len(KNOWN_METHODS_ORDER)


def discover_training_results(results_dir: Path):
    """Parse every top-level {method}_{dataset}_t{T}_s{seed}.json file."""
    records = []
    for f in sorted(results_dir.glob('*.json')):
        m = RESULT_RE.match(f.name)
        if not m:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  [WARN] could not parse {f.name}: {e}")
            continue
        rec = dict(m.groupdict())
        rec['num_tasks'] = int(rec['num_tasks'])
        rec['seed'] = int(rec['seed'])
        rec['_path'] = str(f)
        rec['_data'] = data
        records.append(rec)
    return records


def load_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [WARN] could not parse {path}: {e}")
        return None


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt(m, s=None, nd=2):
    if m is None:
        return "-"
    if s is None or s == 0.0:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}+/-{s:.{nd}f}"


def fmt_with_n(values, n_expected, nd=2):
    """Like fmt(), but flags when fewer seeds actually contributed a numeric
    value than were found on disk -- otherwise a genuine std=0 (all seeds
    agree) and an n=1 fallback (only one seed had a usable value, the rest
    were None) render identically, which is exactly the kind of thing that
    should not be silently indistinguishable in a results table."""
    numeric = [v for v in values if isinstance(v, (int, float))]
    m, s = mean_std(numeric)
    text = fmt(m, s, nd=nd)
    if len(numeric) < n_expected:
        text += f" (n={len(numeric)}/{n_expected})"
    return text


def build_report(results_dir: Path, expect_methods, expect_datasets, expect_seeds):
    lines = []

    def out(s=""):
        lines.append(s)

    out(f"# Experiment summary -- `{results_dir}`\n")

    records = discover_training_results(results_dir)
    out(f"Found **{len(records)}** completed training runs (top-level `*_t*_s*.json` files).\n")

    # ---- completeness check --------------------------------------------------
    found_keys = {(r['method'], r['dataset'], r['seed']) for r in records}
    expected_keys = {(m, d, s) for m in expect_methods for d in expect_datasets for s in expect_seeds}
    missing = sorted(expected_keys - found_keys)
    if missing:
        out(f"## Missing training runs ({len(missing)}/{len(expected_keys)} expected combos not found)")
        for m, d, s in missing:
            out(f"- {m} / {d} / seed {s}")
        out()
    elif expected_keys:
        out("All expected (method, dataset, seed) training combinations are present.\n")

    if not records:
        report_text = "\n".join(lines)
        return report_text, records

    # ---- group by (method, dataset) ------------------------------------------
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['method'], r['dataset'])].append(r)

    datasets_present = sorted({r['dataset'] for r in records})

    # discover any extra scalar keys (from compute_metrics()/count_stored_params(),
    # whose exact schema I did not have visibility into) ONCE, globally, so every
    # per-dataset section reports on a consistent set of fields.
    extra_keys = set()
    for r in records:
        for k, v in r['_data'].items():
            if k in KNOWN_TOP_LEVEL_KEYS:
                continue
            if v is None or isinstance(v, (int, float)):
                extra_keys.add(k)

    for dataset in datasets_present:
        out(f"## {dataset}\n")
        out("| Method | avg_acc | avg_acc_ncm | seeds found |")
        out("|---|---|---|---|")
        methods_here = sorted({m for (m, d) in grouped if d == dataset}, key=method_sort_key)
        for method in methods_here:
            recs = grouped[(method, dataset)]
            data_list = [r['_data'] for r in recs]
            accs = []
            for d in data_list:
                if isinstance(d.get('avg_acc'), (int, float)):
                    accs.append(d['avg_acc'])
                elif d.get('per_task_acc'):
                    accs.append(sum(d['per_task_acc']) / len(d['per_task_acc']))
            ncm_accs = [d.get('avg_acc_ncm') for d in data_list
                       if isinstance(d.get('avg_acc_ncm'), (int, float))]
            seeds_here = sorted(r['seed'] for r in recs)
            n_expected = len(recs)
            acc_text = fmt_with_n(accs, n_expected)
            ncm_text = fmt_with_n(ncm_accs, n_expected)
            out(f"| {method} | {acc_text} | {ncm_text} | {seeds_here} |")
        out()

        if extra_keys:
            out(f"### Other numeric fields found in the result JSONs ({dataset})")
            out("_(schema discovered at runtime -- these come from your `compute_metrics()` / "
                "`count_stored_params()`, which I did not have the source for)_\n")
            for method in methods_here:
                recs = grouped[(method, dataset)]
                data_list = [r['_data'] for r in recs]
                parts = []
                for k in sorted(extra_keys):
                    vals = [d.get(k) for d in data_list if isinstance(d.get(k), (int, float))]
                    m, s = mean_std(vals)
                    if m is not None:
                        parts.append(f"{k}={fmt(m, s, nd=3)}")
                if parts:
                    out(f"- **{method}**: " + ", ".join(parts))
            out()

    # ---- analysis_report.json (drift, interference, alignment) --------------
    out("## Mechanistic analysis (`analysis/run_analysis.py`)\n")
    any_report = False
    combos = sorted({(r['dataset'], r['seed'], r['num_tasks']) for r in records})
    for dataset, seed, num_tasks in combos:
        report_path = (results_dir / 'analysis' / f'{dataset}_t{num_tasks}_s{seed}'
                      / 'analysis_report.json')
        report = load_json_safe(report_path)
        if report is None:
            continue
        any_report = True
        out(f"### {dataset}, seed {seed}")
        drift = report.get('drift_comparison', {})
        methods_report = report.get('methods', {})
        if drift or methods_report:
            out("| Method | L4 cos_sim | L4 rank_corr | avg_interference |")
            out("|---|---|---|---|")
            all_methods_seen = sorted(set(drift) | set(methods_report), key=method_sort_key)
            for method in all_methods_seen:
                l4 = drift.get(method, {}).get('layer4', {})
                interference = methods_report.get(method, {}).get('avg_interference')
                out(f"| {method} | {fmt(l4.get('mean_cos_sim'), nd=4)} "
                    f"| {fmt(l4.get('mean_rank_corr'), nd=4)} "
                    f"| {fmt(interference, nd=2)} |")
        ewc_align = report.get('ewc_sae_alignment')
        if ewc_align:
            rhos = [v['spearman_r'] for v in ewc_align.values() if 'spearman_r' in v]
            layer_strs = ", ".join(f"{k}={v['spearman_r']:.3f}" for k, v in ewc_align.items()
                                   if 'spearman_r' in v)
            if rhos:
                mean_rho = sum(rhos) / len(rhos)
                out(f"  EWC-Fisher vs SAE-discriminativity mean rho = {fmt(mean_rho, nd=3)} "
                    f"(by layer: {layer_strs})")
        out()
    if not any_report:
        out("_No `analysis_report.json` files found yet -- run `analysis/run_analysis.py` first._\n")

    # ---- feature fate summary (all 6 methods, now that the analysis scripts'
    # --method CLI restriction has been widened beyond sae_fsgp/dense_fsgp) ---
    out("## Feature fate at layer4 (`analysis/feature_fate_runner.py`)\n")
    any_fate = False
    for dataset, seed, num_tasks in combos:
        row_shown = False
        for method in KNOWN_METHODS_ORDER:
            fate_path = (results_dir / 'analysis' / f'{dataset}_t{num_tasks}_s{seed}'
                        / method / 'fate_summary.json')
            fs = load_json_safe(fate_path)
            if fs is None or 'layer4' not in fs:
                continue
            if not row_shown:
                out(f"### {dataset}, seed {seed}")
                out("| Method | survived% | drifted% | dead% | mean_cos_sim |")
                out("|---|---|---|---|---|")
                row_shown = True
                any_fate = True
            l4 = fs['layer4']
            out(f"| {method} | {fmt(l4.get('survived_pct'), nd=1)} "
                f"| {fmt(l4.get('drifted_pct'), nd=1)} | {fmt(l4.get('dead_pct'), nd=1)} "
                f"| {fmt(l4.get('mean_cos_sim'), nd=3)} |")
        if row_shown:
            out()
    if not any_fate:
        out("_No `fate_summary.json` files found yet -- run `analysis/feature_fate_runner.py`._\n")

    # ---- interclass separability ---------------------------------------------
    out("## Inter-class separability (`analysis/interclass_separability.py`)\n")
    any_sep = False
    for dataset, seed, num_tasks in combos:
        sep_path = (results_dir / 'analysis' / f'{dataset}_t{num_tasks}_s{seed}'
                   / 'separability' / f'interclass_separability_seed{seed}.json')
        sep = load_json_safe(sep_path)
        if not sep:
            continue
        any_sep = True
        out(f"### {dataset}, seed {seed}")
        out("| Method | mean_offdiag_sim | worst_confuser | effective_rank |")
        out("|---|---|---|---|")
        for method in sorted(sep.keys(), key=method_sort_key):
            s = sep[method]
            out(f"| {method} | {fmt(s.get('mean_offdiag_sim'), nd=4)} "
                f"| {fmt(s.get('worst_nearest_confuser'), nd=4)} "
                f"| {fmt(s.get('effective_rank'), nd=1)} |")
        out()
    if not any_sep:
        out("_No separability files found yet -- run `analysis/interclass_separability.py`._\n")

    # ---- sanity checks --------------------------------------------------------
    out("## Sanity checks\n")

    # (a) suspiciously identical results across different methods, same dataset/seed
    # -- a concrete signature of the snapshot cross-contamination risk in
    # run_analysis.py's find_snapshot_dir() legacy fallback, or of results.csv /
    # json-path collisions.
    signatures = defaultdict(set)
    for r in records:
        d = r['_data']
        acc = d.get('avg_acc')
        if not isinstance(acc, (int, float)):
            continue
        ncm = d.get('avg_acc_ncm')
        sig = (r['dataset'], r['seed'], round(acc, 4),
              round(ncm, 4) if isinstance(ncm, (int, float)) else None)
        signatures[sig].add(r['method'])

    dupes = {sig: methods for sig, methods in signatures.items() if len(methods) > 1}
    if dupes:
        out("**Possible cross-run contamination** -- these methods report IDENTICAL "
            "avg_acc/avg_acc_ncm for the same dataset+seed, which would be an "
            "extraordinary coincidence if genuine:")
        for (dataset, seed, acc, ncm), methods in dupes.items():
            out(f"- {dataset} seed {seed}: {sorted(methods)} all report avg_acc={acc}")
        out()
    else:
        out("- No duplicate/colliding result signatures detected across methods.")

    # (b) training done but analysis stage missing, for sae_fsgp/dense_fsgp specifically
    gaps = []
    for r in records:
        if r['method'] not in SAE_METHODS:
            continue
        fate_path = (results_dir / 'analysis' / f"{r['dataset']}_t{r['num_tasks']}_s{r['seed']}"
                    / r['method'] / 'fate_summary.json')
        if not fate_path.exists():
            gaps.append(f"{r['method']}/{r['dataset']}/seed{r['seed']}")
    if gaps:
        out(f"- Training done but feature-fate analysis missing for: {', '.join(gaps)} "
            f"(run `analysis/run_analysis.py` then `analysis/feature_fate_runner.py`).")
    else:
        out("- All sae_fsgp/dense_fsgp runs found have matching feature-fate analysis.")

    out()
    report_text = "\n".join(lines)
    return report_text, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='./results_sweep')
    ap.add_argument('--expect_methods', nargs='+', default=KNOWN_METHODS_ORDER)
    ap.add_argument('--expect_datasets', nargs='+', default=['cifar100', 'tinyimagenet'])
    ap.add_argument('--expect_seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--save', default=None, help='Optional path to also save the report as markdown')
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        return

    report_text, records = build_report(
        results_dir, args.expect_methods, args.expect_datasets, args.expect_seeds)
    print(report_text)

    if args.save:
        Path(args.save).write_text(report_text, encoding='utf-8')
        print(f"\n[Saved report -> {args.save}]")


if __name__ == '__main__':
    main()