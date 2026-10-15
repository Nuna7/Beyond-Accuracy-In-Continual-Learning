# Beyond Accuracy: Auditing Representation Forgetting in Class-Incremental Learning

This repository trains seven exemplar-free class-incremental learners plus a `Dense-FSGP` ablation, then runs a post-hoc SAE-based audit (feature survival, linear recoverability, concept-level
decodability) on the saved backbones. It provides the code needed to reproduce Table 1 and Table 2 of the main
paper and Tables S1-S3 of the supplement.

No data and no trained checkpoints are included. Everything here is code only.

## Repository layout

```
config.py            experiment configuration (dataclasses)
train.py             single-run training entry point
run_full_pipeline.sh training + analysis sweep across methods/datasets/seeds
data/                dataset loading and class-incremental task splitting
models/              ResNet-18 backbone with activation hooks
methods/             one trainer per compared method
sae/                 TopK sparse autoencoder + offline trainer used by SAE-FSGP
                     and by the post-hoc audit
analysis/            post-hoc audit: feature fate, recoverability, decodability,
                     translation-specificity control, drift/separability diagnostics
utils/                shared metrics and the SAE-FSGP interpretability helper
```

## Environment

Requires Python 3.10+ and a CUDA-capable GPU (all reported experiments used a
single GPU per run).

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

Not included. Place the standard downloads under `--data_root` (default
`./data_cache`) as:

```
data_cache/
  cifar-100-python/{train,test,meta}
  tiny-imagenet-200/{train,val,test,wnids.txt,words.txt}
```

CIFAR-100 (Python pickle version) and Tiny-ImageNet-200 are both publicly
available from their standard hosts.

## Reproducing the main results

Single run:

```bash
python train.py --method sae_fsgp --dataset cifar100 --num_tasks 10 \
    --seed 0 --epochs 50 --data_root ./data_cache --save_dir ./results \
    --save_snapshots
```

`--method` accepts `finetune`, `ewc`, `lwf`, `fecam`, `sae_fsgp`, `dense_fsgp`,
`dsal`, `efc`. `dense_fsgp` reuses the `sae_fsgp` trainer with
`fsgp_use_sae=False` (see `config.py` / `train.py:build_config`); it is not a
separate trainer file.

Full sweep (all methods, both datasets, 5 seeds, training + core post-hoc audit):

```bash
RESULTS_DIR=./results_sweep DATA_ROOT=./data_cache SEEDS="0 1 2 3 4" \
    ./run_full_pipeline.sh
```

The pipeline also generates supporting diagnostics and qualitative figures.

Set `STAGE=train`, `STAGE=analysis`, or `STAGE=all` to control which part runs;
`QUICK=1` runs a 2-epoch, seed-0, CIFAR-100-only smoke test. The script is
idempotent per (method, dataset, seed) and skips work it has already done
unless `FORCE=1`.

## What each analysis script produces

- `analysis/run_analysis.py` — trains the post-hoc TopK SAEs on saved
snapshots and computes per-layer backbone drift reported in the analysis.
- `analysis/feature_fate_runner.py` — the feature-survival test
  (survived / drifted / dead) underlying the "Survived" columns of Table 2.
- `analysis/recoverability.py` — the linear-translation recoverability test
  and the label-permutation-null decodability probe, i.e. the Rec./Dec./Lost
  split in Table 2. Run with `--out_name recoverability.json`.
- `analysis/recoverability_null_control.py` — the mismatched-concept
  translation-specificity control (Sec. 5.3, Table S3).
- `analysis/interclass_separability.py`, `forgetting_trajectory.py`,
  `visual_confirmation.py`, `temporal_fate_plot.py` — supporting diagnostics
  and qualitative figures, not required to reproduce Tables 1-2.

Per-seed outcome counts are written under
`results/analysis/<dataset>_t10_s<seed>/<method>/`; the seed-level percentages
in Table 2 and in the supplement's per-seed tables are unweighted means of
these files across seeds 0-4.

## Randomness

Five seeds (0-4) are used throughout; the class-to-task permutation uses a
fixed, separate seed (42) so that class order is identical across the five
training seeds. The recoverability test (ridge-regularized linear map) is
deterministic given its inputs. The decodability probe is a small
logistic-regression-style linear classifier trained by SGD from a random
initialization that is not separately fixed across analysis re-runs; as a
result, re-running the audit on the same snapshots can shift a small number
of borderline concepts between the "decodable" and "lost" categories, while
leaving the deterministic recoverability outcome unchanged. This is the
source of the small per-run differences documented in Supplement S2.