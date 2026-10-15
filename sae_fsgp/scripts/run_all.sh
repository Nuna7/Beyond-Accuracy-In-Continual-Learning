#!/usr/bin/env bash
# scripts/run_all.sh  — master script: train → analyze → figures
#
# Usage:
#   bash scripts/run_all.sh          # full paper run
#   bash scripts/run_all.sh --quick  # sanity check (3 tasks, 5 epochs)
#
# Environment overrides:
#   METHODS="sae_fsgp ewc"  bash scripts/run_all.sh
#   SEEDS="0"               bash scripts/run_all.sh
#   DATASET=tinyimagenet    bash scripts/run_all.sh

set -e
cd "$(dirname "$0")/.."

export PYTHON=${PYTHON:-python3}
export DATA_ROOT=${DATA_ROOT:-./data_cache}
export SAVE_ROOT=${SAVE_ROOT:-./results}
export WORKERS=${WORKERS:-4}
export DATASET=${DATASET:-cifar100}
export NUM_TASKS=${NUM_TASKS:-10}
export EPOCHS=${EPOCHS:-100}
export SEEDS=${SEEDS:-"0 1 2"}
export METHODS=${METHODS:-"finetune ewc lwf fecam sae_fsgp"}
export SAE_EXPANSION=${SAE_EXPANSION:-8}
export SAE_EPOCHS=${SAE_EPOCHS:-20}

# Pass --quick through if given
QUICK_FLAG=""
for arg in "$@"; do
  case $arg in --quick) QUICK_FLAG="--quick" ;; esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          SAE-FSGP: Full Experiment Pipeline                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  dataset   : $DATASET"
echo "║  num_tasks : $NUM_TASKS"
echo "║  epochs    : $EPOCHS"
echo "║  seeds     : $SEEDS"
echo "║  methods   : $METHODS"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ── Phase 1: Train ─────────────────────────────────────────────────────────────
echo "━━━ Phase 1 / 2: Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/01_train_all.sh $QUICK_FLAG

# ── Phase 2: Analyse seed 0 only (representative) ──────────────────────────────
echo
echo "━━━ Phase 2 / 2: Analysis (seed=0) ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export SEED=0
bash scripts/02_analyze.sh $QUICK_FLAG

echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pipeline complete                                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Accuracy table  → results/results.csv                      ║"
echo "║  Interp data     → results/interp/                          ║"
echo "║  Analysis        → results/analysis/                        ║"
echo "║  Figures         → results/figures/                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
