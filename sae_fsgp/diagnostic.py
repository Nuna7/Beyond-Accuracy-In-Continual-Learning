"""
Quick diagnostic on ALREADY-SAVED data (LLM_REPORT.json) -- no rerun needed.

Question: for DS-AL/FeCAM/EFC's "decodable" (i.e. NOT recovered-by-translation)
concepts, are the cos_after_translation values clustering just BELOW the 0.85
threshold (near-miss -> consistent with a translation-fit artifact, since
these backbones are frozen or near-frozen and the true mapping should be
close to identity) or well below it (genuine mismatch -> "translation doesn't
fix frozen-backbone forgetting" is likely a real finding, not an artifact)?
"""
import json
from collections import defaultdict

with open('results_sweep/LLM_REPORT.json') as f:
    data = json.load(f)

rows = data['rec_rows']
FROZEN_METHODS = {'dsal', 'fecam', 'efc'}

by_method = defaultdict(list)
for r in rows:
    if r['method'] in FROZEN_METHODS and r['outcome'] == 'decodable' and r['cos_after'] is not None:
        by_method[r['method']].append(r['cos_after'])

print(f"{'method':<10}{'n':>5}{'mean':>8}{'median':>8}{'min':>8}{'max':>8}{'frac>0.7':>10}")
for method, vals in sorted(by_method.items()):
    vals_sorted = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    median = vals_sorted[n // 2]
    frac_high = sum(1 for v in vals if v > 0.7) / n
    print(f"{method:<10}{n:>5}{mean:>8.3f}{median:>8.3f}{min(vals):>8.3f}{max(vals):>8.3f}{frac_high:>10.1%}")

print()
print("mean/median near 0.85, frac>0.7 high  -> near-misses, likely a ridge-fit artifact")
print("mean/median well below 0.5            -> genuine mismatch, real finding")
