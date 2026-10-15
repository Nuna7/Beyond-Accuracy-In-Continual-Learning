import pandas as pd
from pathlib import Path

RESULTS_DIR = "./results_sweep"
DATASETS = ["cifar100", "tinyimagenet"]
METHODS = [
    "finetune", "ewc", "lwf", "fecam",
    "sae_fsgp", "dense_fsgp", "efc", "dsal"
]
SEEDS = [0, 1, 2, 3, 4]
NUM_TASKS = 10
FNAME = "recoverability_v3.json"

rows = []

for dataset in DATASETS:
    for seed in SEEDS:
        for method in METHODS:
            path = (
                Path(RESULTS_DIR)
                / "analysis"
                / f"{dataset}_t{NUM_TASKS}_s{seed}"
                / method
                / FNAME
            )

            if not path.exists():
                continue

            data = pd.read_json(path)

            recovered = 0
            decodable = 0
            lost = 0

            for task_id, task_data in data.items():
                if task_id == "_meta":
                    continue

                for result in task_data["per_class"].values():
                    outcome = result["outcome"]

                    if outcome == "recovered":
                        recovered += 1
                    elif outcome == "decodable":
                        decodable += 1
                    elif outcome == "lost":
                        lost += 1

            total = recovered + decodable + lost

            if total == 0:
                continue

            rows.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "checked": total,
                "recovered": recovered / total,
                "decodable": decodable / total,
                "lost": lost / total,
            })

df = pd.DataFrame(rows)

summary = (
    df.groupby(["dataset", "method"])
      .agg(
          n_seeds=("seed", "nunique"),
          total_checked=("checked", "sum"),
          mean_checked_per_seed=("checked", "mean"),
          recovered_mean=("recovered", "mean"),
          recovered_std=("recovered", "std"),
          decodable_mean=("decodable", "mean"),
          decodable_std=("decodable", "std"),
          lost_mean=("lost", "mean"),
          lost_std=("lost", "std"),
      )
      .reset_index()
)

for col in [
    "recovered_mean", "recovered_std",
    "decodable_mean", "decodable_std",
    "lost_mean", "lost_std"
]:
    summary[col] *= 100

print(summary.to_string(index=False))

out = Path(RESULTS_DIR) / "analysis" / "aggregated_v3"
out.mkdir(parents=True, exist_ok=True)

summary.to_csv(out / "summary_seed_level.csv", index=False)

print("\n=== LATEX ===")
for dataset in DATASETS:
    print(f"\n{dataset}")
    sub = summary[summary.dataset == dataset]

    for _, r in sub.iterrows():
        print(
            f"{r.method} & "
            f"{r.recovered_mean:.1f} $\\pm$ {r.recovered_std:.1f} & "
            f"{r.decodable_mean:.1f} $\\pm$ {r.decodable_std:.1f} & "
            f"{r.lost_mean:.1f} $\\pm$ {r.lost_std:.1f} \\\\"
        )
