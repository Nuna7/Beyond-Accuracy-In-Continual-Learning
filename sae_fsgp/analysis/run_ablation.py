"""
Ablation: dense_fsgp vs sae_fsgp.

Tests whether the SAE component in SAE-FSGP is actually necessary by
comparing against a version that distills raw layer activations instead
of SAE decoder projections. Both methods have identical architecture,
training schedule, and hyperparameters; the only difference is
fsgp_use_sae: true/false.

Expected result:
  dense_fsgp:  high cos_sim (distillation preserves feature directions)
               BUT low rank_corr (no structure on which features matter)
  sae_fsgp:    similar cos_sim, HIGHER rank_corr (SAE projections
               explicitly constrain which features are preserved)

If this expectation holds, it directly proves that:
  (a) The SAE is not just a computational wrapper — it changes WHAT
      gets preserved
  (b) rank_corr improvement is caused by the SAE projection loss,
      not just by having any distillation at all

Usage:
  python analysis/run_ablation.py \
      --results_dir ./results \
      --dataset cifar100 \
      --seed 0 \
      --data_root /media/20TB/vision/CL/data_cache

Runs 1 training run (~40 min) then full analysis pipeline.

Equivalently, run training directly:
  python train.py --method dense_fsgp --dataset cifar100 --num_tasks 10 \
      --epochs 50 --no_fsgp_norm_proj --seed 0
  python analysis/run_analysis.py --results_dir ./results \
      --dataset cifar100 --num_tasks 10 --seed 0 \
      --data_root /media/20TB/vision/CL/data_cache

Expected table (add to paper as ablation):
  Method          Native  NCM   L4_cos  L4_rank  Interference
  sae_fsgp        23.9    37.4  0.847   0.478    11.47
  dense_fsgp      ~23     ~35   ~0.84   ~0.38    ~13-14   ← prediction
  (pure distil, no SAE structure)

If dense_fsgp rank_corr < sae_fsgp rank_corr:
  → SAE component is causally responsible for rank_corr improvement ✓

Note: dense_fsgp should already exist as a method variant in train.py
controlled by --method dense_fsgp (fsgp_use_sae=False). If not, set:
  fsgp_use_sae: false  in the config, or pass --no_fsgp_use_sae.
"""

import os, sys, subprocess, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))



ABLATION_METHODS = [
    # (method_name, extra_flags, description)
    ('sae_fsgp',   ['--no_fsgp_norm_proj'],
     'Full SAE-FSGP (projection onto SAE decoder directions)'),
    ('dense_fsgp', ['--no_fsgp_norm_proj'],
     'Dense distillation only (no SAE structure)'),
]

def run_ablation(results_dir, dataset, seed, data_root,
                 num_tasks=10, epochs=50, device='cuda'):
    """
    Run both ablation conditions and compare their drift metrics.
    Returns comparison dict.
    """
    results = {}

    for method, extra_flags, desc in ABLATION_METHODS:
        print(f"\n{'='*60}")
        print(f"Running: {method}  [{desc}]")
        print(f"{'='*60}")

        cmd = [
            'python', 'train.py',
            '--method',    method,
            '--dataset',   dataset,
            '--num_tasks', str(num_tasks),
            '--epochs',    str(epochs),
            '--seed',      str(seed),
        ] + extra_flags

        proc = subprocess.run(cmd, capture_output=False, text=True)
        if proc.returncode != 0:
            print(f"  ERROR: training failed for {method}")
            continue

        # Run analysis
        analysis_cmd = [
            'python', 'analysis/run_analysis.py',
            '--results_dir', results_dir,
            '--dataset',     dataset,
            '--num_tasks',   str(num_tasks),
            '--seed',        str(seed),
            '--data_root',   data_root,
        ]
        subprocess.run(analysis_cmd, capture_output=False, text=True)

        # Load results
        result_path = os.path.join(
            results_dir, f'{method}_{dataset}_t{num_tasks}_s{seed}.json')
        if os.path.exists(result_path):
            with open(result_path) as f:
                results[method] = json.load(f)

    # Compare
    _print_comparison(results, dataset, seed)
    return results


def _print_comparison(results, dataset, seed):
    print(f"\n{'='*65}")
    print(f"ABLATION COMPARISON — {dataset.upper()} seed {seed}")
    print(f"{'='*65}")
    print(f"{'Method':<16} {'Native%':>8} {'NCM%':>7}  "
          f"{'L4 cos':>8} {'L4 rank':>9} {'Interf':>8}")
    print("-"*65)
    for method, data in results.items():
        print(f"{method:<16} "
              f"{data.get('avg_acc', 0):>8.2f} "
              f"{data.get('avg_acc_ncm', 0):>7.2f}  "
              f"(drift metrics in analysis_report.json)")
    print()
    print("Key question: does sae_fsgp have HIGHER rank_corr at L4 than dense_fsgp?")
    print("If yes → SAE projection loss is causally responsible for rank preservation.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--dataset',     default='cifar100')
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--data_root',   default='/media/20TB/vision/CL/data_cache')
    p.add_argument('--num_tasks',   type=int, default=10)
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--device',      default='cuda')
    args = p.parse_args()
    run_ablation(args.results_dir, args.dataset, args.seed,
                 args.data_root, args.num_tasks, args.epochs, args.device)