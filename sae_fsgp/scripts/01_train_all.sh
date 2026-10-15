#!/usr/bin/env bash
# scripts/01_train_all.sh
# Train all baselines + SAE-FSGP variants.
# Each run saves per-task backbone snapshots for post-hoc SAE analysis.
#
# Usage:
#   bash scripts/01_train_all.sh                    # full paper (3 seeds)
#   bash scripts/01_train_all.sh --quick            # 3 tasks / 5 epochs
#   METHODS="sae_fsgp ewc" bash scripts/01_train_all.sh

set -e
cd "$(dirname "$0")/.."

# ── defaults ──────────────────────────────────────────────────────────────────
PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:-./data_cache}
SAVE_ROOT=${SAVE_ROOT:-./results}
WORKERS=${WORKERS:-4}
DATASET=${DATASET:-cifar100}
NUM_TASKS=${NUM_TASKS:-10}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"finetune ewc lwf fecam sae_fsgp"}

QUICK=false
for arg in "$@"; do
  case $arg in --quick) QUICK=true ;; esac
done

if $QUICK; then
  NUM_TASKS=3; EPOCHS=5; SEEDS="0"
  echo "⚡  Quick mode: T=${NUM_TASKS} epochs=${EPOCHS}"
fi

mkdir -p "$SAVE_ROOT"

# ── helper ────────────────────────────────────────────────────────────────────
run_one() {
  local method=$1 seed=$2
  local tag="${method}_${DATASET}_t${NUM_TASKS}_s${seed}"
  local done_file="$SAVE_ROOT/${tag}.json"
  [[ -f "$done_file" ]] && { echo "  [skip] $tag"; return; }

  echo; echo "  ► $tag"
  $PYTHON train.py \
    --method "$method" --dataset "$DATASET" \
    --num_tasks "$NUM_TASKS" --epochs "$EPOCHS" \
    --seed "$seed" --data_root "$DATA_ROOT" \
    --save_dir "$SAVE_ROOT" --num_workers "$WORKERS" \
    --save_snapshots \
    2>&1 | tee "$SAVE_ROOT/${tag}.log"
}

# ── SAE-FSGP variants (ablation) ──────────────────────────────────────────────
run_saefsgp_variants() {
  local seed=$1
  # activity mask  (default, best accuracy)
  run_one sae_fsgp "$seed"

  # dense-only baseline (no SAE projection)
  local tag="dense_fsgp_${DATASET}_t${NUM_TASKS}_s${seed}"
  [[ -f "$SAVE_ROOT/${tag}.json" ]] || {
    echo; echo "  ► $tag (dense ablation)"
    $PYTHON train.py --method dense_fsgp \
      --dataset "$DATASET" --num_tasks "$NUM_TASKS" --epochs "$EPOCHS" \
      --seed "$seed" --data_root "$DATA_ROOT" --save_dir "$SAVE_ROOT" \
      --num_workers "$WORKERS" --save_snapshots \
      2>&1 | tee "$SAVE_ROOT/${tag}.log"
  }
}

# ── main loop ─────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  SAE-FSGP Training  |  methods: $METHODS"
echo "  dataset=$DATASET  T=$NUM_TASKS  epochs=$EPOCHS  seeds: $SEEDS"
echo "================================================================"

for seed in $SEEDS; do
  for method in $METHODS; do
    if [[ "$method" == "sae_fsgp" ]]; then
      run_saefsgp_variants "$seed"
    else
      run_one "$method" "$seed"
    fi
  done
done

echo
echo "================================================================"
echo "  All training complete. Results in: $SAVE_ROOT"
echo "================================================================"
$PYTHON print_table.py --results_dir "$SAVE_ROOT"
