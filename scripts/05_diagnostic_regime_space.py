#!/usr/bin/env python3
"""
Deep diagnostic of the simulated regime space.

Questions this script answers:
1. Do the 7 regimes form separable clouds, or do some collapse?
2. Which descriptor dimensions carry information vs which are degenerate?
3. Are parameter sweeps creating continuous manifolds or just noise?
4. Is the PCA structure interpretable or arbitrary?
5. Which regime pairs are confusable?

Runs 10 realizations per parameter setting for speed (~15 min).
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulators import SIMULATORS, SWEEP_PARAMS, simulate_regime
from src.fingerprinting import compute_fingerprint
from src.descriptors import (extract_descriptors, descriptors_to_matrix,
                              descriptors_to_dicts)

DESC_NAMES = ['peak_height', 'threshold_scale', 'width',
              'small_scale_slope', 'large_scale_slope', 'integrated_excess']

RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

N_REALIZATIONS = 10
N_NULL = 20  # Reduced null for speed; sufficient for VMR ratio stability


def run_simulations():
    """Run all 7 regimes × parameter sweeps × 10 realizations."""
    rng = np.random.default_rng(42)
    all_descriptors = []
    all_vmr_curves = []  # Keep raw curves for additional diagnostics
    total_start = time.time()

    for regime, sweep in sorted(SWEEP_PARAMS.items()):
        param_name = sweep['param']
        param_values = sweep['values']
        regime_start = time.time()
        print(f'\n{regime} ({param_name}): {len(param_values)} settings × {N_REALIZATIONS} realizations')

        for pval in param_values:
            print(f'  {param_name}={pval}', end='', flush=True)
            patterns = simulate_regime(regime, pval,
                                       n_realizations=N_REALIZATIONS,
                                       rng=rng.integers(0, 2**31))
            for i, pattern in enumerate(patterns):
                seed = rng.integers(0, 2**31)
                fp = compute_fingerprint(pattern, roi_size=10000,
                                         n_null=N_NULL, rng=seed)
                desc = extract_descriptors(
                    vmr_ratio=fp.vmr_ratio,
                    scales=fp.scales,
                    label=f'{regime}_{param_name}={pval}_r{i}',
                    source='simulation',
                    regime=regime,
                    positions=pattern,
                    parameters={param_name: float(pval), 'realization': i},
                )
                all_descriptors.append(desc)
                all_vmr_curves.append({
                    'regime': regime,
                    'param': float(pval),
                    'vmr_ratio': fp.vmr_ratio.tolist(),
                    'scales': fp.scales.tolist(),
                })
            print(f' ✓')

        elapsed = time.time() - regime_start
        print(f'  → {elapsed:.0f}s')

    total_elapsed = time.time() - total_start
    print(f'\nTotal: {len(all_descriptors)} descriptors in {total_elapsed:.0f}s')
    return all_descriptors, all_vmr_curves


def diagnose_descriptors(all_descriptors):
    """Examine each descriptor dimension for regime discrimination."""
    X = descriptors_to_matrix(all_descriptors)
    labels = np.array([d.regime for d in all_descriptors])
    regimes = sorted(set(labels))

    print('\n' + '=' * 70)
    print('DESCRIPTOR DIAGNOSTICS')
    print('=' * 70)

    # Per-descriptor: range, variance, discriminative power (ANOVA F-stat)
    from scipy.stats import f_oneway
    print(f'\n{"Descriptor":<22s} {"Range":>12s} {"Std":>8s} {"F-stat":>10s} {"p-value":>12s}  Verdict')
    print('-' * 80)
    f_stats = []
    for i, name in enumerate(DESC_NAMES):
        groups = [X[labels == r, i] for r in regimes]
        f_stat, p_val = f_oneway(*groups)
        overall_range = X[:, i].max() - X[:, i].min()
        overall_std = X[:, i].std()
        verdict = '✓ discriminative' if p_val < 1e-10 and f_stat > 20 else '⚠ weak' if p_val < 0.01 else '✗ useless'
        print(f'{name:<22s} [{X[:,i].min():>6.2f}, {X[:,i].max():>6.2f}] {overall_std:>8.3f} {f_stat:>10.1f} {p_val:>12.2e}  {verdict}')
        f_stats.append(f_stat)

    # Per-regime: centroid in descriptor space
    print(f'\n{"Regime":<22s}', end='')
    for name in DESC_NAMES:
        print(f' {name[:8]:>10s}', end='')
    print()
    print('-' * 85)
    for regime in regimes:
        mask = labels == regime
        centroid = X[mask].mean(axis=0)
        print(f'{regime:<22s}', end='')
        for v in centroid:
            print(f' {v:>10.3f}', end='')
        print()

    return f_stats


def diagnose_separability(all_descriptors):
    """Pairwise regime separability analysis."""
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import StandardScaler

    X = descriptors_to_matrix(all_descriptors)
    labels = np.array([d.regime for d in all_descriptors])
    regimes = sorted(set(labels))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print('\n' + '=' * 70)
    print('SEPARABILITY DIAGNOSTICS')
    print('=' * 70)

    # Overall silhouette
    overall_sil = silhouette_score(X_s, labels)
    print(f'\nOverall silhouette (7 regimes): {overall_sil:.3f}')

    # Per-regime silhouette
    sample_sils = silhouette_samples(X_s, labels)
    print(f'\nPer-regime silhouette:')
    for regime in regimes:
        mask = labels == regime
        regime_sil = sample_sils[mask].mean()
        print(f'  {regime:<25s}: {regime_sil:.3f}')

    # Pairwise silhouette (the confusion matrix)
    print(f'\nPairwise silhouette (higher = more separable):')
    header = f'{"":>22s}'
    for r in regimes:
        header += f' {r[:8]:>8s}'
    print(header)
    pairwise_sils = np.zeros((len(regimes), len(regimes)))
    for i, r1 in enumerate(regimes):
        row = f'{r1:<22s}'
        for j, r2 in enumerate(regimes):
            if i == j:
                row += f' {"---":>8s}'
                continue
            mask = (labels == r1) | (labels == r2)
            X_pair = X_s[mask]
            l_pair = labels[mask]
            sil = silhouette_score(X_pair, l_pair)
            pairwise_sils[i, j] = sil
            row += f' {sil:>8.3f}'
        print(row)

    # Identify most confusable pairs
    print(f'\nMost confusable regime pairs (lowest pairwise silhouette):')
    pairs = []
    for i in range(len(regimes)):
        for j in range(i+1, len(regimes)):
            pairs.append((pairwise_sils[i, j], regimes[i], regimes[j]))
    pairs.sort()
    for sil, r1, r2 in pairs[:5]:
        print(f'  {r1} ↔ {r2}: {sil:.3f}')

    return overall_sil, pairwise_sils, regimes


def diagnose_pca(all_descriptors):
    """PCA structure: variance explained, loadings, interpretability."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = descriptors_to_matrix(all_descriptors)
    labels = np.array([d.regime for d in all_descriptors])

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pca = PCA(n_components=6)  # Fit all 6 to see the full spectrum
    X_pca = pca.fit_transform(X_s)

    print('\n' + '=' * 70)
    print('PCA DIAGNOSTICS')
    print('=' * 70)

    ev = pca.explained_variance_ratio_
    cumev = np.cumsum(ev)
    print(f'\nExplained variance:')
    for i in range(6):
        bar = '█' * int(ev[i] * 50)
        print(f'  PC{i+1}: {ev[i]*100:5.1f}%  (cumulative: {cumev[i]*100:5.1f}%)  {bar}')

    print(f'\nPCA loadings (how each PC relates to descriptors):')
    header = f'{"":>8s}'
    for name in DESC_NAMES:
        header += f' {name[:10]:>12s}'
    print(header)
    for i in range(min(4, 6)):
        row = f'PC{i+1:>5d}  '
        for j in range(6):
            val = pca.components_[i, j]
            marker = '***' if abs(val) > 0.5 else '  *' if abs(val) > 0.3 else '   '
            row += f' {val:>8.3f}{marker}'
        print(row)

    # Test: does the PCA have a natural "elbow"?
    print(f'\nVariance gap analysis:')
    for i in range(5):
        gap = ev[i] - ev[i+1]
        print(f'  PC{i+1}→PC{i+2} gap: {gap*100:.1f}%')

    return pca, X_pca, ev


def diagnose_sweep_monotonicity(all_descriptors):
    """Do parameter sweeps create ordered manifolds or just scatter?"""
    print('\n' + '=' * 70)
    print('PARAMETER SWEEP MONOTONICITY')
    print('=' * 70)
    print('(Spearman correlation between sweep parameter and each descriptor)')
    print('High |ρ| = ordered manifold. Low |ρ| = no systematic trend.\n')

    from scipy.stats import spearmanr
    labels = np.array([d.regime for d in all_descriptors])
    regimes = sorted(set(labels))

    header = f'{"Regime":<22s} {"param":>12s}'
    for name in DESC_NAMES:
        header += f' {name[:8]:>8s}'
    print(header)
    print('-' * 100)

    for regime in regimes:
        mask = labels == regime
        descs = [d for d, m in zip(all_descriptors, mask) if m]
        param_name = SWEEP_PARAMS[regime]['param']
        params = np.array([d.parameters[param_name] for d in descs])
        X = descriptors_to_matrix(descs)

        row = f'{regime:<22s} {param_name:>12s}'
        for i in range(6):
            rho, p = spearmanr(params, X[:, i])
            marker = '**' if abs(rho) > 0.5 else ' *' if abs(rho) > 0.3 else '  '
            row += f' {rho:>6.2f}{marker}'
        print(row)


def diagnose_vmr_curves(all_vmr_curves):
    """Examine raw VMR curve shapes for interpretability."""
    print('\n' + '=' * 70)
    print('VMR CURVE SHAPE DIAGNOSTICS')
    print('=' * 70)

    regimes = sorted(set(c['regime'] for c in all_vmr_curves))
    for regime in regimes:
        curves = [c for c in all_vmr_curves if c['regime'] == regime]
        ratios = np.array([c['vmr_ratio'] for c in curves])
        log_ratios = np.log2(np.maximum(ratios, 1e-10))

        mean_lr = log_ratios.mean(axis=0)
        std_lr = log_ratios.std(axis=0)
        max_mean = mean_lr.max()
        min_mean = mean_lr.min()
        dynamic_range = max_mean - min_mean

        # SNR: mean signal / noise (across-realization std)
        mean_std = std_lr.mean()
        snr = dynamic_range / max(mean_std, 1e-10)

        print(f'  {regime:<22s}: log2 range [{min_mean:>6.2f}, {max_mean:>6.2f}], '
              f'dynamic range={dynamic_range:.2f}, '
              f'mean σ={mean_std:.2f}, SNR={snr:.1f}')


def generate_diagnostic_figures(all_descriptors, pca, X_pca, ev, all_vmr_curves):
    """Generate diagnostic visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    X = descriptors_to_matrix(all_descriptors)
    labels = np.array([d.regime for d in all_descriptors])
    regimes = sorted(set(labels))
    colors = {
        'branch_constrained': '#e67e22', 'clustered': '#e74c3c',
        'compartmental': '#9b59b6', 'exclusion': '#3498db',
        'hierarchical': '#1abc9c', 'layered': '#2ecc71',
        'random': '#888888',
    }

    # --- Fig 1: PCA scatter (PC1 vs PC2, PC1 vs PC3) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    pc_pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (pci, pcj) in zip(axes, pc_pairs):
        for regime in regimes:
            mask = labels == regime
            ax.scatter(X_pca[mask, pci], X_pca[mask, pcj], s=10, alpha=0.4,
                       c=colors.get(regime, 'k'), label=regime.replace('_', ' '),
                       rasterized=True)
        ax.set_xlabel(f'PC{pci+1} ({ev[pci]*100:.1f}%)')
        ax.set_ylabel(f'PC{pcj+1} ({ev[pcj]*100:.1f}%)')
        ax.legend(fontsize=7, loc='best')
    fig.suptitle('Regime Space: PCA Projections', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'diag_pca_scatter.pdf', dpi=150)
    plt.close()

    # --- Fig 2: Mean VMR curves per regime (all param values, colored by param) ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flat
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        curves = [c for c in all_vmr_curves if c['regime'] == regime]
        param_name = SWEEP_PARAMS[regime]['param']
        params = sorted(set(c['param'] for c in curves))
        cmap = plt.cm.viridis
        for pi, pval in enumerate(params):
            pval_curves = [c for c in curves if c['param'] == pval]
            ratios = np.array([c['vmr_ratio'] for c in pval_curves])
            scales = np.array(pval_curves[0]['scales'])
            mean_lr = np.log2(np.maximum(ratios.mean(axis=0), 1e-10))
            color = cmap(pi / max(1, len(params) - 1))
            ax.plot(scales, mean_lr, color=color, alpha=0.7, lw=1.5)
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_title(f'{regime}\n({param_name})', fontsize=10)
        ax.set_xscale('log')
        ax.set_ylabel('log₂(VMR ratio)')
    # Hide extra subplot
    if len(regimes) < len(list(axes)):
        axes[-1].set_visible(False)
    fig.suptitle('VMR Fingerprint Sweeps (color = parameter value)', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'diag_vmr_sweeps.pdf', dpi=150)
    plt.close()

    # --- Fig 3: Descriptor distributions violin-style ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, (ax, name) in enumerate(zip(axes.flat, DESC_NAMES)):
        data_per_regime = []
        for regime in regimes:
            mask = labels == regime
            data_per_regime.append(X[mask, idx])
        bp = ax.boxplot(data_per_regime, labels=[r[:8] for r in regimes],
                        patch_artist=True)
        for patch, regime in zip(bp['boxes'], regimes):
            patch.set_facecolor(colors.get(regime, '#cccccc'))
            patch.set_alpha(0.6)
        ax.set_title(name, fontsize=11)
        ax.tick_params(axis='x', rotation=45)
    fig.suptitle('Descriptor Distributions by Regime', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'diag_descriptor_boxplots.pdf', dpi=150)
    plt.close()

    # --- Fig 4: Sweep monotonicity — each regime, descriptor vs parameter ---
    fig, axes = plt.subplots(len(regimes), 6, figsize=(24, 3 * len(regimes)))
    for ri, regime in enumerate(regimes):
        mask = labels == regime
        descs = [d for d, m in zip(all_descriptors, mask) if m]
        param_name = SWEEP_PARAMS[regime]['param']
        params = np.array([d.parameters[param_name] for d in descs])
        X_regime = descriptors_to_matrix(descs)
        for di in range(6):
            ax = axes[ri, di]
            ax.scatter(params, X_regime[:, di], s=8, alpha=0.4,
                       c=colors.get(regime, 'k'))
            if ri == 0:
                ax.set_title(DESC_NAMES[di][:10], fontsize=9)
            if di == 0:
                ax.set_ylabel(regime[:10], fontsize=9)
            if ri == len(regimes) - 1:
                ax.set_xlabel('param')
    fig.suptitle('Descriptor vs Sweep Parameter (each dot = 1 realization)', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'diag_sweep_monotonicity.pdf', dpi=150)
    plt.close()

    print(f'\nDiagnostic figures saved to {FIGURES_DIR}/')


def main():
    # Step 1: Run simulations
    all_descriptors, all_vmr_curves = run_simulations()

    # Save for reuse
    output = {
        'n_descriptors': len(all_descriptors),
        'descriptors': descriptors_to_dicts(all_descriptors),
    }
    with open(RESULTS_DIR / 'diagnostic_descriptors.json', 'w') as f:
        json.dump(output, f)
    with open(RESULTS_DIR / 'diagnostic_vmr_curves.json', 'w') as f:
        json.dump(all_vmr_curves, f)

    # Step 2: Diagnose
    f_stats = diagnose_descriptors(all_descriptors)
    overall_sil, pairwise_sils, regime_names = diagnose_separability(all_descriptors)
    pca, X_pca, ev = diagnose_pca(all_descriptors)
    diagnose_sweep_monotonicity(all_descriptors)
    diagnose_vmr_curves(all_vmr_curves)

    # Step 3: Figures
    generate_diagnostic_figures(all_descriptors, pca, X_pca, ev, all_vmr_curves)

    # Step 4: Summary verdict
    print('\n' + '=' * 70)
    print('SUMMARY VERDICT')
    print('=' * 70)
    cumev3 = sum(ev[:3])
    print(f'  3 PCs explain {cumev3*100:.1f}% variance: '
          f'{"PASS" if cumev3 > 0.8 else "MARGINAL" if cumev3 > 0.7 else "FAIL"}')
    print(f'  Overall silhouette: {overall_sil:.3f}: '
          f'{"PASS" if overall_sil > 0.5 else "MARGINAL" if overall_sil > 0.3 else "FAIL"}')

    weak_descs = [DESC_NAMES[i] for i, f in enumerate(f_stats)
                  if f < 20]
    if weak_descs:
        print(f'  ⚠ Weak descriptors: {", ".join(weak_descs)}')
    else:
        print(f'  All 6 descriptors are discriminative')

    # Identify collapsing regimes
    collapse_threshold = 0.15
    collapsing = []
    for i in range(len(regime_names)):
        for j in range(i+1, len(regime_names)):
            if pairwise_sils[i, j] < collapse_threshold:
                collapsing.append(f'{regime_names[i]} ↔ {regime_names[j]} '
                                  f'(sil={pairwise_sils[i, j]:.3f})')
    if collapsing:
        print(f'  ⚠ Collapsing regime pairs:')
        for c in collapsing:
            print(f'    {c}')
    else:
        print(f'  No regime pairs collapse (all pairwise sil > {collapse_threshold})')

    print('=' * 70)


if __name__ == '__main__':
    main()
