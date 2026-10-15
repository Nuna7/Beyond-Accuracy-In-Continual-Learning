set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results_sweep}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_cache}"
CONDA_ENV="${CONDA_ENV:-}"
STAGE="${STAGE:-all}"                
FORCE="${FORCE:-0}"
QUICK="${QUICK:-0}"

METHODS="${METHODS:-finetune ewc lwf fecam sae_fsgp dense_fsgp efc dsal}"
DATASETS="${DATASETS:-cifar100 tinyimagenet}"
SEEDS="${SEEDS:-0 1 2 3 4}"
NUM_TASKS="${NUM_TASKS:-10}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAE_EXPANSION="${SAE_EXPANSION:-8}"
SAE_EPOCHS="${SAE_EPOCHS:-20}"
TRAJ_SEED="${TRAJ_SEED:-0}"            
VISUAL_SEED="${VISUAL_SEED:-1}"        
FSGP_LAMBDA="${FSGP_LAMBDA:-}"          
EFC_OPTIMIZER="${EFC_OPTIMIZER:-paper}"

if [ "$QUICK" = "1" ]; then
    echo "[QUICK MODE] epochs=2, seed=0 only, cifar100 only -- pipeline smoke test, not real results"
    EPOCHS=2
    SEEDS="0"
    DATASETS="cifar100"
    TRAJ_SEED=0
    VISUAL_SEED=0
fi

read -ra METHODS_ARR  <<< "$METHODS"
read -ra DATASETS_ARR <<< "$DATASETS"
read -ra SEEDS_ARR    <<< "$SEEDS"

LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"

if [ -n "$CONDA_ENV" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

cd "$PROJECT_ROOT" || { echo "ERROR: cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }

FAILED_STEPS=()
N_OK=0
N_SKIP=0
N_FAIL=0

log()  { echo "[$(date '+%H:%M:%S')] $*"; }


run_logged() {
    local logfile="$1"; shift
    log "  -> $* (log: ${logfile#"$RESULTS_DIR"/})"
    if "$@" > "$logfile" 2>&1; then
        N_OK=$((N_OK+1))
        return 0
    else
        local rc=$?
        N_FAIL=$((N_FAIL+1))
        FAILED_STEPS+=("$logfile :: $*")
        log "     FAILED (exit $rc) -- see $logfile"
        return 1
    fi
}

skip_if_done() {
    local target="$1"
    if [ "$FORCE" != "1" ] && [ -f "$target" ]; then
        N_SKIP=$((N_SKIP+1))
        return 0
    fi
    return 1
}

preflight() {
    log "Preflight checks"
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
            while read -r line; do log "  GPU: $line"; done
    else
        log "  WARNING: nvidia-smi not found -- is a GPU visible in this shell?"
    fi

    if [ ! -d "$DATA_ROOT" ]; then
        log "  WARNING: DATA_ROOT does not exist: $DATA_ROOT"
    fi

    N_TRAIN_RUNS=$(( ${#METHODS_ARR[@]} * ${#DATASETS_ARR[@]} * ${#SEEDS_ARR[@]} ))
    N_ANALYSIS_COMBOS=$(( ${#DATASETS_ARR[@]} * ${#SEEDS_ARR[@]} ))
    EST_GB=$(( (N_TRAIN_RUNS * 450 + N_ANALYSIS_COMBOS * 200) / 1024 + 1 ))
    if command -v df > /dev/null 2>&1; then
        mkdir -p "$RESULTS_DIR"
        AVAIL_KB=$(df -Pk "$RESULTS_DIR" 2>/dev/null | tail -1 | awk '{print $4}')
        if [ -n "$AVAIL_KB" ]; then
            AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
            log "  Disk free at RESULTS_DIR: ${AVAIL_GB}GB   Estimated sweep footprint: ~${EST_GB}GB"
            if [ "$AVAIL_GB" -lt "$EST_GB" ]; then
                log "  WARNING: estimated footprint (~${EST_GB}GB) exceeds free space (${AVAIL_GB}GB)."
                log "           torch.save() failures mid-sweep look like random crashes but are this."
            fi
        fi
    fi

    TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    TORCH_WARN=$(python -c "
import torch
v = torch.__version__.split('+')[0].split('.')
major, minor = int(v[0]), int(v[1])
print('WARN' if (major, minor) >= (2, 6) else 'OK')
" 2>/dev/null)
    log "  torch: ${TORCH_VER:-could not import torch in this shell/env}"
    if [ "$TORCH_WARN" = "WARN" ]; then
        log "  NOTE: torch>=2.6 defaults torch.load(weights_only=True)."
        log "        analysis/posthoc_sae.py's load_snapshot() calls torch.load() WITHOUT"
        log "        weights_only=False (every other snapshot-loading call in analysis/ does)."
        log "        This can raise an UnpicklingError inside run_analysis.py and"
        log "        interclass_separability.py, which both import that function."
        log "        Patch that one call before trusting STAGE=analysis results."
    fi

    mkdir -p "$RESULTS_DIR"
    log "Results dir: $RESULTS_DIR"
    log "Data root:   $DATA_ROOT"
    log "Methods:     ${METHODS_ARR[*]}"
    log "Datasets:    ${DATASETS_ARR[*]}"
    log "Seeds:       ${SEEDS_ARR[*]}"
    log "Epochs/task: $EPOCHS   Num tasks: $NUM_TASKS"
    if [ -n "$FSGP_LAMBDA" ]; then
        log "FSGP_LAMBDA: $FSGP_LAMBDA  (passed to sae_fsgp/dense_fsgp training)"
    else
        log "FSGP_LAMBDA: (not set -- train.py will use config.py's default)"
    fi
}

# ---- stage 1: training --------------------------------------------------------
stage_train() {
    log "=== STAGE: train ==="
    local dataset method seed target logfile
    local fsgp_lambda_args=()
    if [ -n "$FSGP_LAMBDA" ]; then
        read -ra fsgp_lambda_args <<< "$FSGP_LAMBDA"
        fsgp_lambda_args=(--fsgp_lambda "${fsgp_lambda_args[@]}")
    fi
    for dataset in "${DATASETS_ARR[@]}"; do
        for method in "${METHODS_ARR[@]}"; do
            for seed in "${SEEDS_ARR[@]}"; do
                target="${RESULTS_DIR}/${method}_${dataset}_t${NUM_TASKS}_s${seed}.json"
                if skip_if_done "$target"; then
                    log "  [skip] $method/$dataset/seed$seed (found $(basename "$target"))"
                    continue
                fi
                logfile="${LOG_DIR}/train_${method}_${dataset}_s${seed}.log"
                run_logged "$logfile" python train.py \
                    --method "$method" --dataset "$dataset" \
                    --num_tasks "$NUM_TASKS" --seed "$seed" \
                    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
                    --num_workers "$NUM_WORKERS" \
                    "${fsgp_lambda_args[@]}" \
                    --data_root "$DATA_ROOT" --save_dir "$RESULTS_DIR" \
                    --save_snapshots \
                    --efc_optimizer "$EFC_OPTIMIZER"
            done
        done
    done
}

# ---- stages 2-7: analysis -------------------------------------------------------
stage_analysis() {
    log "=== STAGE: analysis ==="
    local dataset seed m target logfile
    for dataset in "${DATASETS_ARR[@]}"; do
        for seed in "${SEEDS_ARR[@]}"; do
            # 2. run_analysis.py -- all methods for this (dataset,seed) in one call
            target="${RESULTS_DIR}/analysis/${dataset}_t${NUM_TASKS}_s${seed}/analysis_report.json"
            if skip_if_done "$target"; then
                log "  [skip] run_analysis $dataset/seed$seed"
            else
                logfile="${LOG_DIR}/analysis_${dataset}_s${seed}.log"
                run_logged "$logfile" python analysis/run_analysis.py \
                    --results_dir "$RESULTS_DIR" --dataset "$dataset" \
                    --num_tasks "$NUM_TASKS" --seed "$seed" --data_root "$DATA_ROOT" \
                    --num_workers "$NUM_WORKERS" --sae_expansion "$SAE_EXPANSION" \
                    --sae_epochs "$SAE_EPOCHS" --methods "${METHODS_ARR[@]}"
            fi

            # 3. feature_fate_runner.py -- ALL methods in METHODS_ARR (default:
            for m in "${METHODS_ARR[@]}"; do
                target="${RESULTS_DIR}/analysis/${dataset}_t${NUM_TASKS}_s${seed}/${m}/fate_summary.json"
                if skip_if_done "$target"; then
                    log "  [skip] feature_fate $m/$dataset/seed$seed"
                else
                    logfile="${LOG_DIR}/fate_${m}_${dataset}_s${seed}.log"
                    run_logged "$logfile" python analysis/feature_fate_runner.py \
                        --results_dir "$RESULTS_DIR" --dataset "$dataset" --seed "$seed" \
                        --data_root "$DATA_ROOT" --num_tasks "$NUM_TASKS" --method "$m"
                fi
            done

            # 4. interclass_separability.py -- scoped to METHODS_ARR like
            target="${RESULTS_DIR}/analysis/${dataset}_t${NUM_TASKS}_s${seed}/separability/interclass_separability_seed${seed}.json"
            if skip_if_done "$target"; then
                log "  [skip] separability $dataset/seed$seed"
            else
                logfile="${LOG_DIR}/separability_${dataset}_s${seed}.log"
                run_logged "$logfile" python analysis/interclass_separability.py \
                    --results_dir "$RESULTS_DIR" --dataset "$dataset" --num_tasks "$NUM_TASKS" \
                    --seed "$seed" --data_root "$DATA_ROOT" --num_workers "$NUM_WORKERS" \
                    --methods "${METHODS_ARR[@]}"
            fi

            for METHOD in "${METHODS_ARR[@]}"; do
                LOG="$RESULTS_DIR/logs/recoverability_${METHOD}_${dataset}_s${seed}.log"

                python analysis/recoverability.py \
                    --results_dir "$RESULTS_DIR" \
                    --dataset "$dataset" \
                    --seed "$seed" \
                    --data_root "$DATA_ROOT" \
                    --num_tasks "$NUM_TASKS" \
                    --method "$METHOD" \
                    --out_name recoverability.json \
                    2>&1 | tee "$LOG"

                if [ ${PIPESTATUS[0]} -ne 0 ]; then
                    echo "WARNING: FAILED $METHOD $dataset s$seed"
                fi
            done
            
        done

        # 5 & 6: illustrative qualitative figures -- ONE seed each (these scripts
        for m in "${METHODS_ARR[@]}"; do
            target="${RESULTS_DIR}/figures/fig_trajectory_${m}_${dataset}_s${TRAJ_SEED}.png"
            if skip_if_done "$target"; then
                log "  [skip] trajectory $m/$dataset"
            else
                logfile="${LOG_DIR}/trajectory_${m}_${dataset}.log"
                run_logged "$logfile" python analysis/forgetting_trajectory.py \
                    --results_dir "$RESULTS_DIR" --dataset "$dataset" --seed "$TRAJ_SEED" \
                    --num_tasks "$NUM_TASKS" --data_root "$DATA_ROOT" --method "$m"
            fi

            target="${RESULTS_DIR}/analysis/visual_confirmation/${m}_${dataset}_s${VISUAL_SEED}_visual.png"
            if skip_if_done "$target"; then
                log "  [skip] visual_confirmation $m/$dataset"
            else
                logfile="${LOG_DIR}/visual_${m}_${dataset}.log"
                run_logged "$logfile" python analysis/visual_confirmation.py \
                    --results_dir "$RESULTS_DIR" --dataset "$dataset" --seed "$VISUAL_SEED" \
                    --num_tasks "$NUM_TASKS" --data_root "$DATA_ROOT" --method "$m"
            fi
        done
    done

    # 7. temporal_fate_plot.py -- needs BOTH datasets x seeds [0,1,2], sae_fsgp only
    logfile="${LOG_DIR}/temporal_fate_plot.log"
    run_logged "$logfile" python analysis/temporal_fate_plot.py --results_dir "$RESULTS_DIR"

    
}



# ---- main --------------------------------------------------------------------
preflight

case "$STAGE" in
    train)     stage_train ;;
    analysis)  stage_analysis ;;
    all)       stage_train; stage_analysis ;;
    *) echo "Unknown STAGE='$STAGE' (expected train|analysis|all)"; exit 1 ;;
esac

log "=== DONE === ok=$N_OK skipped=$N_SKIP failed=$N_FAIL"
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    log "Failed steps:"
    for s in "${FAILED_STEPS[@]}"; do log "  - $s"; done
fi
