import os, sys, json, argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

METHODS      = ['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp']
METHOD_COLOR = {
    'finetune': '#E24B4A',
    'ewc':      '#F5A623',
    'lwf':      '#7ED321',
    'fecam':    '#4A90D9',
    'sae_fsgp': '#534AB7',
}
METHOD_LABEL = {
    'finetune': 'Fine-tuning',
    'ewc':      'EWC',
    'lwf':      'LwF',
    'fecam':    'FeCAM',
    'sae_fsgp': 'SAE-FSGP (ours)',
}

METHOD_COLOR.update({
    'dense_fsgp': '#8B5A2B',
    'efc':        '#D6336C',
    'dsal':       '#0CA678'
})
METHOD_LABEL.update({
    'dense_fsgp': 'Dense-FSGP (ablation)',
    'efc':        'EFC',
    'dsal':       'DS-AL'
})

LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']
LAYER_LABEL = {'layer1': 'Layer 1\n(64-d)', 'layer2': 'Layer 2\n(128-d)',
               'layer3': 'Layer 3\n(256-d)', 'layer4': 'Layer 4\n(512-d)'}

def plot_fate_stacked_bar(analysis_dir: str, out_dir: str):
    fate_path = os.path.join(analysis_dir, 'sae_fsgp', 'fate_summary.json')
    if not os.path.exists(fate_path):
        print(f"  [fig1] fate_summary.json not found, skipping")
        return

    with open(fate_path) as f:
        summary = json.load(f)

    layers     = [l for l in LAYERS if l in summary]
    survived   = [summary[l]['survived_pct'] for l in layers]
    drifted    = [summary[l]['drifted_pct']  for l in layers]
    dead       = [summary[l]['dead_pct']     for l in layers]
    x          = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x, survived, color='#4CAF50', label='Survived')
    b2 = ax.bar(x, drifted,  bottom=survived, color='#FF9800', label='Drifted')
    b3 = ax.bar(x, dead,
                bottom=[s + d for s, d in zip(survived, drifted)],
                color='#F44336', label='Dead')

    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABEL[l] for l in layers], fontsize=9)
    ax.set_ylabel('% of class-active latents', fontsize=10)
    ax.set_title('SAE Feature Fate After 10 Tasks (CIFAR-100)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 105)
    ax.spines[['top', 'right']].set_visible(False)

    path = os.path.join(out_dir, 'fig1_fate_stacked_bar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig1] saved → {path}")

def plot_interference_heatmap(analysis_dir: str, out_dir: str,
                               method: str = 'sae_fsgp'):
    npy_path = os.path.join(analysis_dir, method, 'interference_matrix.npy')
    if not os.path.exists(npy_path):
        print(f"  [fig2] interference_matrix.npy not found for {method}, skipping")
        return

    I   = np.load(npy_path)
    T   = I.shape[0]
    I_masked = np.where(np.isnan(I), 0, I)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(I_masked, cmap='Reds', aspect='auto',
                   vmin=0, vmax=np.nanpercentile(I, 95))
    plt.colorbar(im, ax=ax, label='Accuracy drop (%)')

    ax.set_xlabel('Causative task j', fontsize=10)
    ax.set_ylabel('Forgotten task i', fontsize=10)
    ax.set_title(f'Task Interference Matrix — {METHOD_LABEL.get(method, method)}',
                 fontsize=11)
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticklabels([f'T{i}' for i in range(T)], fontsize=7)
    ax.set_yticklabels([f'T{i}' for i in range(T)], fontsize=7)

    path = os.path.join(out_dir, f'fig2_interference_{method}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig2] saved → {path}")

def plot_drift_curves(analysis_dir: str, out_dir: str):
    """
    For each method, plot mean layer4 cos_sim per old task.
    Uses drift_by_task.json (or drift_by_layer.json as fallback).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False

    for method in METHODS:
        path = os.path.join(analysis_dir, method, 'drift_by_task.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        tasks   = sorted(int(k) for k in data.keys())
        cos_sim = [data[str(t)]['mean_cos_sim'] for t in tasks]
        ax.plot(tasks, cos_sim, marker='o', markersize=4, linewidth=1.8,
                color=METHOD_COLOR.get(method, '#888'),
                label=METHOD_LABEL.get(method, method))
        plotted = True

    if not plotted:
        print("  [fig3] no drift_by_task.json found, skipping")
        return

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5,
               label='Perfect preservation')
    ax.set_xlabel('Old task index', fontsize=10)
    ax.set_ylabel('Mean layer4 cosine similarity ↑', fontsize=10)
    ax.set_title('Backbone Feature Drift by Method (layer4)', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines[['top', 'right']].set_visible(False)

    path = os.path.join(out_dir, 'fig3_drift_curves_layer4.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig3] saved → {path}")

def plot_ewc_sae_scatter(analysis_dir: str, out_dir: str):
    align_path = os.path.join(analysis_dir, 'ewc', 'ewc_sae_alignment.json')
    if not os.path.exists(align_path):
        print("  [fig4] ewc_sae_alignment.json not found, skipping")
        return

    with open(align_path) as f:
        alignment = json.load(f)

    fig, axes = plt.subplots(1, len(alignment), figsize=(4 * len(alignment), 3.5),
                              sharey=False)
    if len(alignment) == 1:
        axes = [axes]

    for ax, (layer, stats) in zip(axes, alignment.items()):
        r   = stats['spearman_r']
        p   = stats['p_value']
        lbl = f"ρ={r:.3f}\np={p:.1e}"
        ax.text(0.05, 0.93, lbl, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(LAYER_LABEL.get(layer, layer), fontsize=10)
        ax.set_xlabel('EWC Fisher importance', fontsize=9)
        ax.set_ylabel('SAE discriminativity', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('EWC Weight Importance vs SAE Feature Discriminativity',
                 fontsize=11, y=1.02)
    path = os.path.join(out_dir, 'fig4_ewc_sae_alignment.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig4] saved → {path}")

def plot_rank_vs_cos(analysis_dir: str, out_dir: str):
    """
    Scatter: x = mean cos_sim, y = mean rank_corr per layer per method.
    Interpretation: rank_corr measures whether WHICH features matter is preserved;
    cos_sim measures whether their VALUES are preserved.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    markers = {'layer1': 'o', 'layer2': 's', 'layer3': '^', 'layer4': 'D'}

    for method in METHODS:
        path = os.path.join(analysis_dir, method, 'drift_by_layer.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for layer in LAYERS:
            if layer not in data:
                continue
            cs = data[layer].get('mean_cos_sim', np.nan)
            rc = data[layer].get('mean_rank_corr', np.nan)
            if np.isnan(cs) or np.isnan(rc):
                continue
            ax.scatter(cs, rc,
                       color=METHOD_COLOR.get(method, '#888'),
                       marker=markers[layer], s=60, zorder=3,
                       label=f"{METHOD_LABEL.get(method, method)} {layer}"
                             if layer == 'layer4' else '_nolegend_')

    ax.set_xlabel('Mean cosine similarity (feature magnitude preserved)', fontsize=10)
    ax.set_ylabel('Mean rank correlation (feature ordering preserved)',    fontsize=10)
    ax.set_title('Direction vs Ordering Preservation by Method & Layer',   fontsize=11)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0.4, 1.05); ax.set_ylim(0.6, 1.05)
    ax.plot([0.4, 1.05], [0.4, 1.05], 'k--', alpha=0.3, linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)

    path = os.path.join(out_dir, 'fig5_rank_vs_cos_scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig5] saved → {path}")

def plot_fisher_sae_scatter_multi(analysis_dir: str, out_dir: str,
                                  layer: str = 'layer4'):
    """
    Real scatter (unlike the old plot_ewc_sae_scatter): each point is one
    output unit at `layer`, x = fresh Fisher importance on that method's
    final backbone, y = SAE discriminativity for that same unit. One panel
    per method, log-scaled since Fisher magnitudes span orders of magnitude.
    """
    raw_path = os.path.join(analysis_dir, 'fisher_sae_alignment_multi_raw.json')
    summary_path = os.path.join(analysis_dir, 'fisher_sae_alignment_multi.json')
    if not os.path.exists(raw_path):
        print("  [fig4b] fisher_sae_alignment_multi_raw.json not found "
              "(run analysis/fisher_alignment_multi.py first), skipping")
        return

    with open(raw_path) as f:
        raw = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    methods = [m for m in raw if layer in raw[m]]
    if not methods:
        print(f"  [fig4b] no methods have {layer} data, skipping")
        return

    ncols = min(len(methods), 5)
    nrows = (len(methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows),
                              squeeze=False)

    for idx, method in enumerate(methods):
        ax = axes[idx // ncols][idx % ncols]
        pts = raw[method][layer]
        fisher = np.array(pts['fisher'])
        sae_disc = np.array(pts['sae_disc'])
        r = summary.get(method, {}).get(layer, {}).get('spearman_r', float('nan'))

        ax.scatter(fisher, sae_disc, s=14, alpha=0.6,
                  color=METHOD_COLOR.get(method, '#888'))
        ax.set_xscale('symlog', linthresh=1e-8)
        ax.set_title(f"{METHOD_LABEL.get(method, method)}\n\u03c1={r:.3f}",
                    fontsize=9)
        ax.tick_params(labelsize=7)
        if idx // ncols == nrows - 1:
            ax.set_xlabel('Fisher importance', fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel('SAE discriminativity', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    for idx in range(len(methods), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    fig.suptitle(f'Fisher Importance vs SAE Discriminativity — {layer}', fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f'fig4b_fisher_sae_scatter_multi_{layer}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig4b] saved -> {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis_dir', default='./results/analysis')
    p.add_argument('--out_dir',      default='./results/figures')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nGenerating figures → {args.out_dir}")

    plot_fate_stacked_bar(args.analysis_dir, args.out_dir)
    for method in ['sae_fsgp', 'ewc', 'lwf']:
        plot_interference_heatmap(args.analysis_dir, args.out_dir, method)
    plot_drift_curves(args.analysis_dir, args.out_dir)
    plot_ewc_sae_scatter(args.analysis_dir, args.out_dir)
    plot_rank_vs_cos(args.analysis_dir, args.out_dir)
    plot_fisher_sae_scatter_multi(args.analysis_dir, args.out_dir, 'layer4')

    print(f"\n✓ All figures saved to {args.out_dir}")

if __name__ == '__main__':
    main()
