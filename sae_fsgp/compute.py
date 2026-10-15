"""
Decisive test: for the SAME (task_id, class) concepts, compare cos_sim
BEFORE translation (feature_fate.json, already computed by
feature_fate_runner.py) against cos_after_translation (recoverability.json).

For DS-AL specifically this is close to a clean experiment: since the
backbone is frozen after task 0, the TRUE relationship between task-t and
task-(t+s) features is the identity map. A correctly-fit translation should
push cos_sim UP, toward ~1.0. If instead cos_after < cos_before for DS-AL's
concepts, that's direct evidence the ridge fit is distorting an
otherwise-good signal, not revealing genuine non-linear reorganization.

No rerun needed -- reads files already on disk.
"""
import json
import glob
import os
from collections import defaultdict

RESULTS_DIR = 'results_sweep'
TARGET_METHODS = ['dsal', 'fecam', 'efc']

stats = defaultdict(list)  # method -> list of (cos_before, cos_after)

for fate_path in glob.glob(os.path.join(RESULTS_DIR, 'analysis', '*', '*', 'feature_fate.json')):
    method = os.path.basename(os.path.dirname(fate_path))
    if method not in TARGET_METHODS:
        continue
    rec_path = os.path.join(os.path.dirname(fate_path), 'recoverability.json')
    if not os.path.exists(rec_path):
        continue

    with open(fate_path) as f:
        fate = json.load(f)
    with open(rec_path) as f:
        rec = json.load(f)

    for task_id, task_rec in rec.items():
        fate_layer4 = fate.get(task_id, {}).get('layer4', {}).get('per_class', {})
        for cls, rec_entry in task_rec.get('per_class', {}).items():
            cos_after = rec_entry.get('cos_after_translation')
            cos_before = fate_layer4.get(cls, {}).get('cos_sim')
            if cos_after is not None and cos_before is not None:
                stats[method].append((cos_before, cos_after))

print(f"{'method':<10}{'n':>5}{'mean_before':>13}{'mean_after':>12}{'mean_delta':>12}{'n_worse':>9}{'n_better':>9}")
for method, pairs in sorted(stats.items()):
    n = len(pairs)
    if n == 0:
        print(f"{method:<10}   0  (no matched concepts found)")
        continue
    mean_before = sum(p[0] for p in pairs) / n
    mean_after = sum(p[1] for p in pairs) / n
    mean_delta = mean_after - mean_before
    n_worse = sum(1 for b, a in pairs if a < b)
    n_better = sum(1 for b, a in pairs if a > b)
    print(f"{method:<10}{n:>5}{mean_before:>13.3f}{mean_after:>12.3f}{mean_delta:>+12.3f}{n_worse:>9}{n_better:>9}")

print()
print("If mean_after > mean_before (positive delta), translation is helping --")
print("  the low absolute values are then a real 'not enough' finding, not an artifact.")
print("If mean_after < mean_before or n_worse >> n_better, especially for dsal,")
print("  the ridge fit is likely hurting an otherwise-near-identity mapping --")
print("  refit with a smaller RIDGE_LAMBDA before trusting the recovered/decodable split.")