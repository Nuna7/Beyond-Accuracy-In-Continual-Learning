import os, sys, json, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']
SEEDS  = [0, 1, 2]

COLOURS = {
    'survived': '#2ca02c',
    'drifted':  '#ff7f0e',
    'dead':     '#d62728',
}

def load_fate_data(results_dir, dataset, seed, num_tasks=10):
    """Return per-task layer4 fate counts and full fate_summary."""
    fate_path = os.path.join(
        results_dir, 'analysis',
        f'{dataset}_t10_s{seed}', 'sae_fsgp', 'feature_fate.json')
    summ_path = os.path.join(
        results_dir, 'analysis',
        f'{dataset}_t10_s{seed}', 'sae_fsgp', 'fate_summary.json')

    with open(fate_path) as f:
        fate = json.load(f)
    with open(summ_path) as f:
        summary = json.load(f)

    per_task_l4 = {}
    for t in range(num_tasks):
        d = fate[str(t)]['layer4']
        tot = max(d['total'], 1)
        per_task_l4[t] = {
            'survived': d['survived'] / tot * 100,
            'drifted':  d['drifted']  / tot * 100,
            'dead':     d['dead']     / tot * 100,
        }

    return per_task_l4, summary

def main(results_dir, fig_dir=None):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if fig_dir is None:
        fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    layer_summary = {}
    temporal      = {}

    for ds, n_tasks in [('cifar100', 10), ('tinyimagenet', 10)]:
        layer_summary[ds] = {l: {'survived':[], 'drifted':[], 'dead':[]}
                             for l in LAYERS}
        temporal[ds]      = {t: {'survived':[], 'drifted':[], 'dead':[]}
                             for t in range(n_tasks)}

        for seed in SEEDS:
            try:
                per_task, summary = load_fate_data(results_dir, ds, seed, n_tasks)
            except FileNotFoundError as e:
                print(f"  MISSING: {e} — skipping")
                continue

            for layer in LAYERS:
                for fate in ('survived', 'drifted', 'dead'):
                    layer_summary[ds][layer][fate].append(
                        summary[layer][f'{fate}_pct'])

            for t in range(n_tasks):
                for fate in ('survived', 'drifted', 'dead'):
                    temporal[ds][t][fate].append(per_task[t][fate])

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            width_ratios=[1.3, 1, 1],
                            hspace=0.42, wspace=0.32)

    ax_bar_c = fig.add_subplot(gs[:, 0])
    ax_tmp_c = fig.add_subplot(gs[0, 1])
    ax_tmp_t = fig.add_subplot(gs[1, 1])
    ax_vis   = fig.add_subplot(gs[:, 2])

    layer_labels = ['L1', 'L2', 'L3', 'L4']
    x = np.arange(len(LAYERS))
    w = 0.38

    ax_bar_c.set_title('(a) Feature fate by layer\n(SAE-FSGP, 3 seeds each)',
                        fontsize=10)

    for di, (ds, offset, hatch) in enumerate([
            ('cifar100',    -w/2, ''), ('tinyimagenet', +w/2, '///')]):

        ds_label = 'CIFAR-100' if ds == 'cifar100' else 'Tiny-ImageNet'
        summ     = layer_summary[ds]
        bottom   = np.zeros(len(LAYERS))

        for fate in ('survived', 'drifted', 'dead'):
            vals = [np.mean(summ[l][fate]) if summ[l][fate] else 0
                    for l in LAYERS]
            errs = [np.std(summ[l][fate]) if len(summ[l][fate]) > 1 else 0
                    for l in LAYERS]
            ax_bar_c.bar(x + offset, vals, w, bottom=bottom,
                         color=COLOURS[fate], alpha=0.82, hatch=hatch,
                         label=f'{fate.capitalize()} ({ds_label})'
                               if di == 0 else None,
                         yerr=errs if fate == 'dead' else None,
                         error_kw=dict(capsize=3, elinewidth=1))
            bottom += np.array(vals)

    ax_bar_c.set_xticks(x)
    ax_bar_c.set_xticklabels(layer_labels, fontsize=10)
    ax_bar_c.set_ylabel('% of class-level SAE features', fontsize=9)
    ax_bar_c.set_ylim(0, 105)
    ax_bar_c.axhline(50, color='k', ls=':', lw=0.6, alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOURS['survived'], label='Survived'),
        Patch(facecolor=COLOURS['drifted'],  label='Drifted'),
        Patch(facecolor=COLOURS['dead'],     label='Dead'),
        Patch(facecolor='grey', alpha=0.5, label='CIFAR-100'),
        Patch(facecolor='grey', alpha=0.5, hatch='///', label='Tiny-ImageNet'),
    ]
    ax_bar_c.legend(handles=legend_elements, fontsize=7.5,
                    loc='upper right', framealpha=0.85)

    ax_bar_c.annotate('L3 on\nTiny-IN:\n60% drifted',
                      xy=(2 + w/2, 50), xytext=(2.6, 65),
                      fontsize=7, color='#ff7f0e',
                      arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.2))

    for ax, ds, title in [
        (ax_tmp_c, 'cifar100',    '(b) CIFAR-100: Layer-4 fate by task order'),
        (ax_tmp_t, 'tinyimagenet','(c) Tiny-ImageNet: Layer-4 fate by task order'),
    ]:
        n_tasks  = 10
        tasks    = np.arange(n_tasks)
        tmp      = temporal[ds]

        s_mean = np.array([np.mean(tmp[t]['survived']) for t in tasks])
        d_mean = np.array([np.mean(tmp[t]['drifted'])  for t in tasks])
        de_mean= np.array([np.mean(tmp[t]['dead'])     for t in tasks])

        s_std  = np.array([np.std(tmp[t]['survived']) for t in tasks])
        d_std  = np.array([np.std(tmp[t]['drifted'])  for t in tasks])
        de_std = np.array([np.std(tmp[t]['dead'])     for t in tasks])

        ax.stackplot(tasks, de_mean, d_mean, s_mean,
                     labels=['Dead', 'Drifted', 'Survived'],
                     colors=[COLOURS['dead'], COLOURS['drifted'],
                             COLOURS['survived']],
                     alpha=0.80)

        ax.fill_between(tasks,
                        np.clip(de_mean - de_std, 0, 100),
                        np.clip(de_mean + de_std, 0, 100),
                        alpha=0.25, color=COLOURS['dead'])

        ax.set_xticks(tasks)
        ax.set_xticklabels([f'T{t}' for t in tasks], fontsize=8)
        ax.set_ylabel('% features', fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=9.5)

        if ds == 'tinyimagenet':
            ax.set_xlabel('Task (T0 = oldest, T9 = most recent)', fontsize=8)

        ax.legend(fontsize=7.5, loc='lower right', framealpha=0.85)

        dead_peak_t = int(np.argmax(de_mean[:5]))
        ax.annotate(f'{de_mean[dead_peak_t]:.0f}%\ndead',
                    xy=(dead_peak_t, de_mean[dead_peak_t]/2),
                    fontsize=7.5, ha='center', color='white', fontweight='bold')
        ax.annotate('0%\ndead', xy=(9, 5), fontsize=7.5,
                    ha='center', color='white', fontweight='bold')

    ax_vis.set_axis_off()
    ax_vis.text(0.5, 0.55,
                "Visual confirmation\n(Fig. 1d)\n\n"
                "See:\n"
                "results/analysis/visual_confirmation/\n"
                "cifar100_s1_visual_fixed.png\n\n"
                "Forgotten latent (palm tree):\n"
                "  act 0.489 → 0.000  (0% retained)\n\n"
                "Stable latent (sea/ocean):\n"
                "  act 1.829 → 1.737  (95% retained)",
                transform=ax_vis.transAxes,
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax_vis.set_title('(d) Latent visual fate (CIFAR-100 seed 1)', fontsize=9.5)
    ax_vis.set_xlim(0, 1); ax_vis.set_ylim(0, 1)

    fig.suptitle(
        'Fig. 1 — Feature Fate in SAE-FSGP: where does catastrophic forgetting happen?',
        fontsize=12, fontweight='bold', y=0.99)

    out_path = os.path.join(fig_dir, 'fig1_feature_fate_complete.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close()

    print("\nLAYER-WISE FATE SUMMARY (3-seed mean ± std)")
    print(f"{'Dataset':<15} {'Layer':<8} {'Survived':>12} {'Drifted':>12} {'Dead':>10}")
    print("-" * 60)
    for ds in ('cifar100', 'tinyimagenet'):
        ds_label = 'CIFAR-100' if ds == 'cifar100' else 'TinyImageNet'
        for layer in LAYERS:
            summ = layer_summary[ds][layer]
            def ms(k):
                v = summ[k]
                if not v: return ' N/A'
                return f"{np.mean(v):5.1f}±{np.std(v):.1f}%"
            print(f"{ds_label:<15} {layer.replace('layer','L'):<8} "
                  f"{ms('survived'):>12} {ms('drifted'):>12} {ms('dead'):>10}")
        print()

    print("\nTEMPORAL LAYER-4 FATE (3-seed mean)")
    for ds in ('cifar100', 'tinyimagenet'):
        print(f"\n  {ds.upper()}:")
        print(f"  {'Task':<7} {'Survived':>10} {'Drifted':>10} {'Dead':>8}")
        for t in range(10):
            tmp = temporal[ds][t]
            if not tmp['survived']:
                continue
            s  = np.mean(tmp['survived']); d  = np.mean(tmp['drifted'])
            de = np.mean(tmp['dead'])
            print(f"  T{t:<6} {s:>9.1f}%  {d:>9.1f}%  {de:>7.1f}%")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--fig_dir',     default=None)
    args = p.parse_args()
    main(args.results_dir, args.fig_dir)
