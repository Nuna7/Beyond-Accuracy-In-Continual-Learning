#!/usr/bin/env python3
"""
Consolidate all results relevant to:

python train.py --method dense_fsgp --dataset tinyimagenet --num_tasks 10 --epochs 50 --fsgp_lambda 1.0 1.0 2.0 4.0 --seed 0
+ full analysis chain

Output: ./results/consolidated/dense_fsgp_tinyimagenet_t10_s0_ALL.json
"""

import os, json, glob
from pathlib import Path

RESULTS_DIR = "./results"
DATASET = "tinyimagenet"
NUM_TASKS = 10
SEED = 0
TAG = f"{DATASET}_t{NUM_TASKS}_s{SEED}"
METHOD_FOCUS = "dense_fsgp"

def load_json(p):
    if not os.path.exists(p):
        return {"_missing": str(p)}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": str(p)}

def main():
    results_dir = Path(RESULTS_DIR)
    tag = TAG
    
    consolidated = {
        "experiment": {
            "command": "python train.py --method dense_fsgp --dataset tinyimagenet --num_tasks 10 --epochs 50 --fsgp_lambda 1.0 1.0 2.0 4.0 --seed 0",
            "method": METHOD_FOCUS,
            "dataset": DATASET,
            "num_tasks": NUM_TASKS,
            "seed": SEED,
            "tag": f"{METHOD_FOCUS}_{DATASET}_t{NUM_TASKS}_s{SEED}"
        },
        "train_results": {},
        "analysis_report": {},
        "drift_comparison": {},
        "interference": {},
        "feature_fate": {},
        "interclass_separability": {},
    }

    # 1. Main training result: results/dense_fsgp_tinyimagenet_t10_s0.json
    train_json = results_dir / f"{METHOD_FOCUS}_{DATASET}_t{NUM_TASKS}_s{SEED}.json"
    consolidated["train_results"]["dense_fsgp_tinyimagenet"] = load_json(train_json)
    
    # Also include csv line if exists
    csv_path = results_dir / "results.csv"
    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('method') == 'dense_fsgp' and DATASET in row.get('dataset','') and row.get('seed')=='0' or str(SEED) in row.get('seed',''):
                    # loose match: method + dataset + seed
                    if f"{METHOD_FOCUS}_{DATASET}" in f"{row.get('method')}_{row.get('dataset')}" and row.get('num_tasks')==str(NUM_TASKS):
                        consolidated["train_results"]["csv_row"] = row
                        break

    # 2. Analysis report: results/analysis/tinyimagenet_t10_s0/analysis_report.json
    analysis_base = results_dir / "analysis" / tag
    report_path = analysis_base / "analysis_report.json"
    consolidated["analysis_report"] = load_json(report_path)

    # Per-method drift and interference inside analysis/tinyimagenet_t10_s0/{method}/
    for method in ['finetune','ewc','lwf','fecam','sae_fsgp','dense_fsgp']:
        m_dir = analysis_base / method
        if m_dir.exists():
            drift = load_json(m_dir / "drift_by_layer.json")
            # also check posthoc location
            if "_missing" in drift:
                drift = load_json(m_dir / "posthoc_sae" / "drift_by_layer.json" ) if False else drift
                # try alternative: analysis report already has drift_comparison
            consolidated["drift_comparison"][method] = drift
            
            interference = {}
            # interference saved by save_comparison_results
            for fn in ["interference.json", "comparison.json", "interference_matrix.json"]:
                p = m_dir / fn
                if p.exists():
                    interference = load_json(p)
                    break
            if interference:
                consolidated["interference"][method] = interference

    # If analysis_report has drift_comparison, merge
    if isinstance(consolidated["analysis_report"], dict) and "drift_comparison" in consolidated["analysis_report"]:
        consolidated["drift_comparison"]["_from_report"] = consolidated["analysis_report"]["drift_comparison"]

    # 3. Feature fate runner outputs
    # results/analysis/{tag}/sae_fsgp/feature_fate.json + fate_summary.json
    # results/analysis/{tag}/dense_fsgp/feature_fate.json + fate_summary.json
    for method in ['sae_fsgp', 'dense_fsgp']:
        f_dir = analysis_base / method
        fate = load_json(f_dir / "feature_fate.json")
        summary = load_json(f_dir / "fate_summary.json")
        consolidated["feature_fate"][method] = {
            "feature_fate": fate,
            "fate_summary": summary
        }

    # 4. Interclass separability
    # results/analysis/{dataset}_t10_s{seed}/separability/interclass_separability_seed{seed}.json
    for ds in ['cifar100', 'tinyimagenet']:
        sep_tag = f"{ds}_t{NUM_TASKS}_s{SEED}"
        sep_path = results_dir / "analysis" / sep_tag / "separability" / f"interclass_separability_seed{SEED}.json"
        consolidated["interclass_separability"][ds] = load_json(sep_path)
        # fallback search
        if "_missing" in consolidated["interclass_separability"][ds]:
            alt = list((results_dir / "analysis").rglob(f"interclass_separability_seed{SEED}.json"))
            for a in alt:
                if ds in str(a):
                    consolidated["interclass_separability"][ds] = load_json(a)

    # 5. Save consolidated
    out_dir = results_dir / "consolidated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{METHOD_FOCUS}_{DATASET}_t{NUM_TASKS}_s{SEED}_ALL.json"
    with open(out_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    # Also save a human-readable summary txt
    txt_path = out_dir / f"{METHOD_FOCUS}_{DATASET}_t{NUM_TASKS}_s{SEED}_SUMMARY.txt"
    with open(txt_path, 'w') as f:
        f.write(f"CONSOLIDATED RESULTS - {METHOD_FOCUS} {DATASET} T={NUM_TASKS} S={SEED}\n")
        f.write("="*70+"\n\n")
        # train metrics
        tr = consolidated["train_results"].get("dense_fsgp_tinyimagenet", {})
        f.write(f"[TRAIN] {train_json}\n")
        for k in ["final_avg_acc", "avg_forgetting", "bwt", "avg_acc", "avg_acc_ncm", "per_task_acc", "per_task_acc_ncm"]:
            if k in tr:
                f.write(f"  {k}: {tr[k]}\n")
        f.write("\n[ANALYSIS REPORT]\n")
        f.write(json.dumps(consolidated["analysis_report"], indent=2)[:5000]+"\n\n")
        f.write("[DRIFT COMPARISON]\n")
        for m, d in consolidated["drift_comparison"].items():
            if isinstance(d, dict) and "layer4" in d:
                f.write(f"  {m}: {d['layer4']}\n")
        f.write("\n[FEATURE FATE]\n")
        for m in ['sae_fsgp','dense_fsgp']:
            if m in consolidated["feature_fate"]:
                fs = consolidated["feature_fate"][m].get("fate_summary", {})
                f.write(f"  {m}: {fs}\n")
        f.write("\n[SEPARABILITY]\n")
        for ds, sep in consolidated["interclass_separability"].items():
            if isinstance(sep, dict):
                f.write(f"  {ds}:\n")
                for meth, stats in sep.items():
                    if isinstance(stats, dict) and "mean_offdiag_sim" in stats:
                        f.write(f"    {meth}: offdiag={stats['mean_offdiag_sim']:.4f} worst={stats['worst_nearest_confuser']:.4f} eff_rank={stats['effective_rank']:.1f}\n")

    print(f"✓ Consolidated JSON saved -> {out_path}")
    print(f"✓ Summary TXT saved -> {txt_path}")
    return str(out_path)

if __name__ == "__main__":
    main()
