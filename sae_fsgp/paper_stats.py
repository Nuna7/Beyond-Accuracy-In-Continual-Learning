#!/usr/bin/env python3
"""
paper_stats.py

Paired significance tests for the head-to-head comparisons that matter for
the paper: sae_fsgp vs dense_fsgp (the ablation) and sae_fsgp vs fecam (the
strongest baseline), on avg_acc and avg_acc_ncm, per dataset.

Why paired rather than an independent two-sample test: train.py calls
set_seed(cfg.seed) -- which seeds random/numpy/torch -- identically for every
method before training starts, and ClassIncrementalScenario's class order is
controlled by a SEPARATE, fixed class_order_seed, not cfg.seed. So for a given
seed, every method sees the same task/class ordering and the same initial
weights/data-shuffling draws. Seed-matched runs across methods are genuinely
paired observations (they share a source of variation), not independent
samples -- a paired test is the statistically correct choice here, and it's
also more powerful than an unpaired test at these small sample sizes.

If the two methods being compared don't have the same seeds on disk (e.g.
sae_fsgp has 5 seeds, fecam has 3), this uses only the INTERSECTION and says
so explicitly, rather than silently comparing a subset of one against a
different subset of the other.

Usage:
    python paper_stats.py --results_dir ./results_sweep
    python paper_stats.py --results_dir ./results_sweep \\
        --pairs sae_fsgp:dense_fsgp sae_fsgp:fecam \\
        --datasets cifar100 tinyimagenet --metrics avg_acc avg_acc_ncm
"""
import re
import json
import argparse
import statistics
from pathlib import Path

try:
    from scipy import stats as _stats
except ImportError:
    _stats = None

RESULT_RE = re.compile(
    r'^(?P<method>.+)_(?P<dataset>cifar100|tinyimagenet)_t(?P<num_tasks>\d+)_s(?P<seed>\d+)\.json$'
)


def load_values(results_dir: Path, method: str, dataset: str, metric: str):
    """Returns {seed: value} for one method/dataset/metric, only where the
    metric is present and numeric."""
    out = {}
    for f in results_dir.glob(f"{method}_{dataset}_t*_s*.json"):
        m = RESULT_RE.match(f.name)
        if not m or m.group('method') != method or m.group('dataset') != dataset:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  [WARN] could not parse {f.name}: {e}")
            continue
        val = data.get(metric)
        if isinstance(val, (int, float)):
            out[int(m.group('seed'))] = val
    return out


def paired_report(method_a, method_b, dataset, metric, results_dir: Path):
    vals_a = load_values(results_dir, method_a, dataset, metric)
    vals_b = load_values(results_dir, method_b, dataset, metric)

    seeds_a = set(vals_a)
    seeds_b = set(vals_b)
    common = sorted(seeds_a & seeds_b)

    print(f"\n### {method_a} vs {method_b} -- {dataset} -- {metric}")
    if not common:
        print(f"  No overlapping seeds ({method_a}: {sorted(seeds_a)}, "
              f"{method_b}: {sorted(seeds_b)}). Skipping.")
        return

    if seeds_a != seeds_b:
        extra_a = sorted(seeds_a - seeds_b)
        extra_b = sorted(seeds_b - seeds_a)
        note = []
        if extra_a:
            note.append(f"{method_a} has extra seeds {extra_a} not used here")
        if extra_b:
            note.append(f"{method_b} has extra seeds {extra_b} not used here")
        print(f"  NOTE: seed sets differ -- {'; '.join(note)}. "
              f"Using only the {len(common)} overlapping seeds: {common}.")

    a = [vals_a[s] for s in common]
    b = [vals_b[s] for s in common]
    diffs = [x - y for x, y in zip(a, b)]
    n = len(common)

    print(f"  n = {n} (seeds {common})")
    print(f"  {method_a}: {[round(v, 2) for v in a]}  mean={statistics.mean(a):.2f}")
    print(f"  {method_b}: {[round(v, 2) for v in b]}  mean={statistics.mean(b):.2f}")
    mean_diff = statistics.mean(diffs)
    print(f"  mean difference ({method_a} - {method_b}) = {mean_diff:+.2f}")

    if n < 2:
        print("  Only 1 overlapping seed -- no variance, cannot test.")
        return

    if _stats is None:
        print("  scipy not available in this environment -- install scipy to get "
              "t-test / Wilcoxon p-values (numeric summary above still valid).")
        return

    # Paired t-test with an analytic 95% CI on the mean difference
    t_stat, p_val = _stats.ttest_rel(a, b)
    sd_diff = statistics.stdev(diffs)
    se_diff = sd_diff / (n ** 0.5)
    if n > 1:
        t_crit = _stats.t.ppf(0.975, df=n - 1)
        ci_lo, ci_hi = mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff
        print(f"  paired t-test: t={t_stat:.3f}, p={p_val:.4f}, "
              f"95% CI on mean diff = [{ci_lo:+.2f}, {ci_hi:+.2f}]")

    # Wilcoxon signed-rank as a distribution-free cross-check
    if n >= 4 and any(d != 0 for d in diffs):
        try:
            w_stat, w_p = _stats.wilcoxon(a, b)
            print(f"  Wilcoxon signed-rank: p={w_p:.4f}")
        except ValueError as e:
            print(f"  Wilcoxon signed-rank: not computable ({e})")
    else:
        print(f"  Wilcoxon signed-rank: skipped (n={n} -- with n<4 the smallest "
              f"possible two-sided p-value already exceeds 0.05, so the test "
              f"cannot be informative regardless of the data)")

    if n < 5:
        print(f"  CAUTION: n={n} is small. Treat this p-value as suggestive, not "
              f"confirmatory -- df={n-1} paired t-tests have limited power to "
              f"detect anything but large, consistent effects.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='./results_sweep')
    ap.add_argument('--pairs', nargs='+', default=['sae_fsgp:dense_fsgp', 'sae_fsgp:fecam'],
                    help="method_a:method_b pairs, e.g. sae_fsgp:dense_fsgp")
    ap.add_argument('--datasets', nargs='+', default=['cifar100', 'tinyimagenet'])
    ap.add_argument('--metrics', nargs='+', default=['avg_acc', 'avg_acc_ncm'])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        return

    if _stats is None:
        print("NOTE: scipy is not importable in this environment -- t-test/Wilcoxon "
              "p-values will be skipped, only means and raw values will print.\n")

    for pair in args.pairs:
        try:
            method_a, method_b = pair.split(':')
        except ValueError:
            print(f"Skipping malformed --pairs entry '{pair}' (expected a:b)")
            continue
        for dataset in args.datasets:
            for metric in args.metrics:
                paired_report(method_a, method_b, dataset, metric, results_dir)


if __name__ == '__main__':
    main()