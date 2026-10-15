#!/usr/bin/env bash
# scripts/02_analyze.sh
# Run all mechanistic analyses on saved results + generate figures.
# Must be run AFTER 01_train_all.sh.
#
# Usage:
#   bash scripts/02_analyze.sh
#   bash scripts/02_analyze.sh --quick   (fewer SAE epochs)

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:-./data_cache}
SAVE_ROOT=${SAVE_ROOT:-./results}
WORKERS=${WORKERS:-4}
DATASET=${DATASET:-cifar100}
NUM_TASKS=${NUM_TASKS:-10}
SEED=${SEED:-0}
SAE_EXPANSION=${SAE_EXPANSION:-8}
SAE_EPOCHS=${SAE_EPOCHS:-20}

QUICK=false
for arg in "$@"; do
  case $arg in --quick) QUICK=true ;; esac
done
if $QUICK; then SAE_EPOCHS=5; fi

echo "================================================================"
echo "  SAE-FSGP Mechanistic Analysis"
echo "  dataset=$DATASET  T=$NUM_TASKS  seed=$SEED"
echo "================================================================"

# ── Step 1: Feature fate (requires SAE-FSGP trainer state) ────────────────────
# This is run as part of training (interp/ is already written).
echo
echo "[A] Checking existing interp results …"
INTERP_DIR="$SAVE_ROOT/interp/${DATASET}_t${NUM_TASKS}_seed${SEED}"
if [[ -d "$INTERP_DIR" ]]; then
  echo "    found: $INTERP_DIR"
  ls "$INTERP_DIR"
else
  echo "    ⚠  Not found: $INTERP_DIR"
  echo "       Run 01_train_all.sh with --method sae_fsgp first."
fi

# ── Step 2: Post-hoc SAE + cross-method analysis ──────────────────────────────
echo
echo "[B] Running full mechanistic analysis …"
$PYTHON analysis/run_analysis.py \
  --results_dir  "$SAVE_ROOT" \
  --dataset      "$DATASET" \
  --num_tasks    "$NUM_TASKS" \
  --seed         "$SEED" \
  --data_root    "$DATA_ROOT" \
  --num_workers  "$WORKERS" \
  --sae_expansion "$SAE_EXPANSION" \
  --sae_epochs    "$SAE_EPOCHS" \
  2>&1 | tee "$SAVE_ROOT/analysis_run.log"

# ── Step 3: Figures ────────────────────────────────────────────────────────────
echo
echo "[C] Generating figures …"
$PYTHON viz/plot_analysis.py \
  --analysis_dir "$SAVE_ROOT/analysis" \
  --out_dir      "$SAVE_ROOT/figures"

echo
echo "================================================================"
echo "  Analysis complete."
echo "  Figures → $SAVE_ROOT/figures/"
echo "  Report  → $SAVE_ROOT/analysis/analysis_report.json"
echo "================================================================"
