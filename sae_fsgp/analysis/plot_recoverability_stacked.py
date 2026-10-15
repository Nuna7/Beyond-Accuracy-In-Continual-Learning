#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("./results_sweep")
INPUT = RESULTS_DIR / "analysis" / "aggregated_v3" / "all_class_outcomes.csv"
OUT_DIR = RESULTS_DIR / "analysis" / "aggregated_v3"

METHODS = [
    "finetune",
    "ewc",
    "lwf",
    "fecam",
    "dsal",
    "efc",
    "sae_fsgp",
    "dense_fsgp",
]

METHOD_LABELS = {
    "finetune": "Fine-tuning",
    "ewc": "EWC",
    "lwf": "LwF",
    "fecam": "FeCAM",
    "dsal": "DS-AL",
    "efc": "EFC",
    "sae_fsgp": "SAE-FSGP",
    "dense_fsgp": "Dense-FSGP",
}

DATASETS = [
    ("cifar100", "CIFAR-100"),
    ("tinyimagenet", "Tiny-ImageNet"),
]

SEEDS = [0, 1, 2, 3, 4]

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Aggregate checked concepts within each seed
# ---------------------------------------------------------------------

def seed_level_summary(df):
    """
    Construct one row for every expected
    (dataset, method, seed) combination.

    Important:
    - A seed with zero checked concepts remains in the table.
    - Its outcome fractions are NaN rather than 0.
    - This means zero-coverage seeds do not enter the conditional
      outcome percentage calculation.
    """

    expected = pd.MultiIndex.from_product(
        [
            [d[0] for d in DATASETS],
            METHODS,
            SEEDS,
        ],
        names=["dataset", "method", "seed"],
    ).to_frame(index=False)

    observed = (
        df.groupby(
            ["dataset", "method", "seed", "outcome"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["recovered", "decodable", "lost"]:
        if col not in observed.columns:
            observed[col] = 0

    seed = expected.merge(
        observed,
        on=["dataset", "method", "seed"],
        how="left",
    )

    for col in ["recovered", "decodable", "lost"]:
        seed[col] = seed[col].fillna(0)

    seed["total"] = (
        seed["recovered"]
        + seed["decodable"]
        + seed["lost"]
    )

    # Conditional fractions:
    # only defined when at least one concept was checked.
    seed["recovered_frac"] = np.where(
        seed["total"] > 0,
        seed["recovered"] / seed["total"],
        np.nan,
    )

    seed["decodable_frac"] = np.where(
        seed["total"] > 0,
        seed["decodable"] / seed["total"],
        np.nan,
    )

    seed["lost_frac"] = np.where(
        seed["total"] > 0,
        seed["lost"] / seed["total"],
        np.nan,
    )

    return seed


# ---------------------------------------------------------------------
# Build one dataset panel
# ---------------------------------------------------------------------

def add_stacked_bars(ax, summary, methods):
    x = np.arange(len(methods))

    recovered = summary["recovered_mean"].to_numpy() * 100
    decodable = summary["decodable_mean"].to_numpy() * 100
    lost = summary["lost_mean"].to_numpy() * 100

    # Recovered
    ax.bar(
        x,
        recovered,
        label="Recovered",
        edgecolor="black",
        linewidth=0.55,
    )

    # Decodable
    ax.bar(
        x,
        decodable,
        bottom=recovered,
        label="Decodable",
        edgecolor="black",
        linewidth=0.55,
        hatch="//",
    )

    # Lost
    ax.bar(
        x,
        lost,
        bottom=recovered + decodable,
        label="Lost",
        edgecolor="black",
        linewidth=0.55,
        hatch="xx",
    )

    return x


def annotate_counts(ax, summary, methods):
    """
    N = total checked concept instances across all five seeds.

    A trailing '+' indicates that fewer than all five seeds contributed
    at least one checked concept.
    """

    for i, method in enumerate(methods):

        total_checked = int(summary.loc[method, "total_checked"])
        nonempty_seeds = int(summary.loc[method, "nonempty_seeds"])

        suffix = "(*)" if nonempty_seeds < len(SEEDS) else ""

        label = rf"$N={total_checked}{suffix}$"

        ax.text(
            i,
            101.5,
            label,
            ha="center",
            va="bottom",
            fontsize=5.5,
        )


def make_dataset_summary(seed_df, dataset):
    sub = seed_df[
        (seed_df["dataset"] == dataset)
        & (seed_df["method"].isin(METHODS))
    ].copy()

    methods = [m for m in METHODS if m in sub["method"].unique()]

    summary = (
        sub.groupby("method")
        .agg(
            recovered_mean=("recovered_frac", "mean"),
            decodable_mean=("decodable_frac", "mean"),
            lost_mean=("lost_frac", "mean"),
            total_checked=("total", "sum"),
            n_seed_rows=("seed", "count"),
            nonempty_seeds=("total", lambda x: int((x > 0).sum())),
        )
        .reindex(methods)
    )

    return methods, summary


# ---------------------------------------------------------------------
# Single-panel figures
# ---------------------------------------------------------------------

def make_dataset_figure(seed_df, dataset, dataset_label):

    methods, summary = make_dataset_summary(seed_df, dataset)

    fig, ax = plt.subplots(figsize=(7.1, 3.7))

    add_stacked_bars(ax, summary, methods)
    annotate_counts(ax, summary, methods)

    ax.set_title(
        dataset_label,
        fontsize=11,
        pad=5,
    )

    ax.set_ylabel(
        "Outcome among checked concepts (%)",
        fontsize=9,
    )

    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in methods],
        rotation=40,
        ha="right",
        fontsize=8,
    )

    ax.tick_params(axis="y", labelsize=8)

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.55,
        alpha=0.5,
    )

    ax.set_axisbelow(True)

    # Legend close to the actual plot.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.7,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.28,
        top=0.88,
    )

    out_pdf = OUT_DIR / f"recoverability_stacked_{dataset}.pdf"
    out_png = OUT_DIR / f"recoverability_stacked_{dataset}.png"

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------
# Combined two-panel figure for the paper
# ---------------------------------------------------------------------

def make_combined_figure(seed_df):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.55),
        sharey=True,
    )

    for ax, (dataset, dataset_label) in zip(axes, DATASETS):

        methods, summary = make_dataset_summary(seed_df, dataset)

        add_stacked_bars(ax, summary, methods)
        annotate_counts(ax, summary, methods)

        ax.set_title(
            dataset_label,
            fontsize=10,
            pad=4,
        )

        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(
            [METHOD_LABELS[m] for m in methods],
            rotation=43,
            ha="right",
            fontsize=6.7,
        )

        ax.tick_params(
            axis="y",
            labelsize=7.5,
        )

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.5,
            alpha=0.5,
        )

        ax.set_axisbelow(True)

    axes[0].set_ylabel(
        "Outcome among checked concepts (%)",
        fontsize=8.5,
    )

    # One shared legend.
    handles, labels = axes[1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.7,
        columnspacing=1.3,
        borderaxespad=0.0,
    )

    # Very little whitespace between legend / figure.
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.28,
        top=0.84,
        wspace=0.08,
    )

    out_pdf = OUT_DIR / "recoverability_stacked_both.pdf"
    out_png = OUT_DIR / "recoverability_stacked_both.png"

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}\n"
            "Run analysis/aggregate_recoverability.py first."
        )

    df = pd.read_csv(INPUT)

    required = {
        "dataset",
        "method",
        "seed",
        "outcome",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Keep the eight reported configurations.
    df = df[df["method"].isin(METHODS)].copy()

    seed_df = seed_level_summary(df)

    print("\nSeed-level summary used for plotting:")
    print(
        seed_df[
            [
                "dataset",
                "method",
                "seed",
                "total",
                "recovered_frac",
                "decodable_frac",
                "lost_frac",
            ]
        ].to_string(index=False)
    )

    # Individual dataset figures
    for dataset, dataset_label in DATASETS:
        make_dataset_figure(
            seed_df,
            dataset,
            dataset_label,
        )

    # Combined figure for the paper
    make_combined_figure(seed_df)


if __name__ == "__main__":
    main()