# save as: collect_llm_report.py
# usage: python collect_llm_report.py --results_dir./results_sweep --num_tasks 10
import os, json, glob
from pathlib import Path
from collections import defaultdict, Counter
import argparse

def load_json(p):
    try:
        with open(p) as f: return json.load(f)
    except: return None

p = argparse.ArgumentParser()
p.add_argument('--results_dir', default='./results_sweep')
p.add_argument('--num_tasks', type=int, default=10)
p.add_argument('--out_txt', default='./results_sweep/LLM_REPORT.txt')
p.add_argument('--out_json', default='./results_sweep/LLM_REPORT.json')
args = p.parse_args()

analysis_root = Path(args.results_dir) / 'analysis'
tags = sorted([d.name for d in analysis_root.iterdir() if d.is_dir()]) # e.g. cifar100_t10_s0

# ---- 1. RECOVERABILITY ----
# structure: analysis/{tag}/{method}/recoverability.json
# inside: {task_id: {s, per_class: {class_id: {outcome, cos_after_translation, probe_f1}}}}
rec_rows = [] # flat rows for csv/json
rec_agg = defaultdict(Counter) # key = (dataset, method) -> Counter

for tag in tags:
    # parse dataset from tag: cifar100_t10_s0 -> dataset=cifar100, seed=0
    try:
        dataset_part, seed_part = tag.rsplit('_s',1)
        dataset = dataset_part.rsplit(f'_t{args.num_tasks}_')[0]
        seed = int(seed_part)
    except:
        continue
    tag_dir = analysis_root / tag
    for method_dir in tag_dir.iterdir():
        if not method_dir.is_dir(): continue
        method = method_dir.name
        # skip files like fisher_sae_alignment_multi.json which live in tag_dir, not method_dir
        if method in ['posthoc_sae']: continue
        rec_path = method_dir / 'recoverability.json'
        if not rec_path.exists(): continue
        data = load_json(rec_path)
        if not data: continue
        for task_id, task_data in data.items():
            for cls, res in task_data.get('per_class', {}).items():
                outcome = res.get('outcome')
                rec_rows.append({
                    'dataset': dataset, 'seed': seed, 'method': method,
                    'task_id': int(task_id), 's': task_data.get('s'),
                    'class': int(cls), 'outcome': outcome,
                    'cos_after': res.get('cos_after_translation'),
                    'f1': res.get('probe_f1')
                })
                rec_agg[(dataset, method)][outcome] += 1
                rec_agg[(dataset, method)]['total'] += 1
                rec_agg[('ALL', method)][outcome] += 1
                rec_agg[('ALL', method)]['total'] += 1

# ---- 2. FISHER ALIGNMENT MULTI ----
# structure: analysis/{tag}/fisher_sae_alignment_multi.json
# {method: {layer: {spearman_r, p_value, n_units}}}
fish_rows = []
fish_agg = defaultdict(list) # (dataset, method) -> [mean_rho]

for tag in tags:
    try:
        dataset_part, seed_part = tag.rsplit('_s',1)
        dataset = dataset_part.rsplit(f'_t{args.num_tasks}_')[0]
        seed = int(seed_part)
    except: continue
    f_path = analysis_root / tag / 'fisher_sae_alignment_multi.json'
    if not f_path.exists(): continue
    data = load_json(f_path)
    if not data: continue
    for method, layers in data.items():
        rhos = []
        for layer, stats in layers.items():
            r = stats.get('spearman_r')
            if r is None: continue
            rhos.append(r)
            fish_rows.append({
                'dataset': dataset, 'seed': seed, 'method': method,
                'layer': layer, 'rho': r, 'p': stats.get('p_value'), 'n': stats.get('n_units')
            })
        if rhos:
            mean_r = sum(rhos)/len(rhos)
            fish_agg[(dataset, method)].append(mean_r)
            fish_agg[('ALL', method)].append(mean_r)

# ---- BUILD LLM REPORT ----
lines = []
lines.append(f"# CL Analysis Summary (num_tasks={args.num_tasks})")
lines.append(f"Tags found: {len(tags)} -> {', '.join(tags[:10])}{'...' if len(tags)>10 else ''}")
lines.append("")

lines.append("## 1. RECOVERABILITY (Filus Eq.3 adaptation)")
lines.append("Question: Of features flagged dead/drifted by feature_fate.py, how many are recovered by linear translation, decodable by probe, or lost?")
lines.append("")
if rec_agg:
    lines.append("| dataset | method | total_checked | recovered% | decodable% | lost% |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for (dataset, method), c in sorted(rec_agg.items()):
        total = c['total']
        if total==0: continue
        rc = c['recovered']/total*100
        dc = c['decodable']/total*100
        lc = c['lost']/total*100
        lines.append(f"| {dataset} | {method:<12} | {total} | {rc:.1f}% | {dc:.1f}% | {lc:.1f}% |")
    lines.append("")
    # per-dataset per-seed breakdown for LLM to see variance
    lines.append("### Per (dataset, seed, method) counts")
    from collections import defaultdict
    per_seed = defaultdict(Counter)
    for r in rec_rows:
        per_seed[(r['dataset'], r['seed'], r['method'])][r['outcome']] += 1
        per_seed[(r['dataset'], r['seed'], r['method'])]['total'] += 1
    lines.append("| dataset | seed | method | total | rec | dec | lost |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for k in sorted(per_seed):
        ds, seed, meth = k
        c = per_seed[k]
        lines.append(f"| {ds} | {seed} | {meth:<12} | {c['total']} | {c['recovered']} | {c['decodable']} | {c['lost']} |")
else:
    lines.append("No recoverability.json found yet.")
lines.append("")

lines.append("## 2. FISHER ALIGNMENT MULTI")
lines.append("Question: Does Fisher diagonal (freshly computed on final backbone, same MC procedure as EWC) correlate with SAE discriminativity? Spearman rho.")
lines.append("")
if fish_agg:
    lines.append("| dataset | method | mean_rho (mean over seeds & layers) | n_seeds |")
    lines.append("|---|---|---:|---:|")
    for (dataset, method), rhos in sorted(fish_agg.items()):
        if not rhos: continue
        mean_rho = sum(rhos)/len(rhos)
        lines.append(f"| {dataset} | {method:<12} | {mean_rho:.3f} | {len(rhos)} |")
    lines.append("")
    lines.append("### Layer breakdown")
    lines.append("| dataset | seed | method | layer | rho | n_units |")
    lines.append("|---|---|---|---|---:|---:|")
    for r in sorted(fish_rows, key=lambda x:(x['dataset'], x['seed'], x['method'])):
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['method']:<12} | {r['layer']} | {r['rho']:.3f} | {r['n']} |")
else:
    lines.append("No fisher_sae_alignment_multi.json found yet.")
lines.append("")

lines.append("## 3. Raw file list")
lines.append("```")
for pat in ["analysis/*/fisher_sae_alignment_multi*.json", "analysis/*/*/recoverability.json"]:
    for f in sorted(glob.glob(os.path.join(args.results_dir, pat))):
        lines.append(f)
lines.append("```")

txt = "\n".join(lines)
print(txt)

Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_txt, 'w') as f: f.write(txt)

summary_json = {
    'rec_agg': {f"{k[0]}__{k[1]}": dict(v) for k,v in rec_agg.items()},
    'fish_agg_mean_rho': {f"{k[0]}__{k[1]}": sum(v)/len(v) if v else None for k,v in fish_agg.items()},
    'rec_rows': rec_rows,
    'fish_rows': fish_rows,
    'tags': tags
}
with open(args.out_json, 'w') as f: json.dump(summary_json, f, indent=2)

print(f"\nSaved -> {args.out_txt}")
print(f"Saved -> {args.out_json}")
