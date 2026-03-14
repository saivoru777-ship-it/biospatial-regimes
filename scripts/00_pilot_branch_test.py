#!/usr/bin/env python3
"""
Pilot test: verify branch-constrained separates from clustered.

Simulates random, clustered, and branch-constrained only (~5 min).
Computes 6 primary descriptors + anisotropy supplementary.

Success criteria:
  - silhouette(branch vs clustered) > 0.3
  - anisotropy KS test p < 0.05
  - Visual separation in top-2 PCs
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulators import (simulate_random, simulate_clustered,
                             simulate_branch_constrained, SWEEP_PARAMS)
from src.fingerprinting import compute_fingerprint
from src.descriptors import extract_descriptors, descriptors_to_matrix
from src.embedding import RegimeSpace

RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

N_REALIZATIONS = 10  # Reduced for pilot speed
PILOT_REGIMES = {
    'random': {
        'func': simulate_random,
        'param': 'n_points',
        'values': [1000, 2000, 4000],
    },
    'clustered': {
        'func': simulate_clustered,
        'param': 'cluster_radius',
        'values': [100, 300, 800],
    },
    'branch_constrained': {
        'func': simulate_branch_constrained,
        'param': 'branch_width',
        'values': [20, 80, 250],
    },
}


def main():
    rng = np.random.default_rng(42)
    all_descriptors = []

    for regime, config in PILOT_REGIMES.items():
        func = config['func']
        param_name = config['param']
        print(f'\n=== {regime} ===')
        for pval in config['values']:
            print(f'  {param_name}={pval}', end='', flush=True)
            for i in range(N_REALIZATIONS):
                seed = rng.integers(0, 2**31)
                kwargs = {param_name: pval, 'rng': seed, 'roi_size': 10000}
                if regime != 'random':
                    kwargs['n_points'] = 2000
                pattern = func(**kwargs)
                fp = compute_fingerprint(pattern, roi_size=10000, n_null=20, rng=seed)
                desc = extract_descriptors(
                    vmr_ratio=fp.vmr_ratio,
                    scales=fp.scales,
                    label=f'{regime}_{param_name}={pval}_r{i}',
                    source='simulation',
                    regime=regime,
                    positions=pattern,
                    parameters={param_name: pval, 'realization': i},
                )
                all_descriptors.append(desc)
                print('.', end='', flush=True)
            print()

    print(f'\nTotal descriptors: {len(all_descriptors)}')

    # --- Evaluation ---

    # 1. Silhouette (branch vs clustered only)
    bc_mask = np.array([d.regime in ('branch_constrained', 'clustered')
                        for d in all_descriptors])
    bc_descs = [d for d, m in zip(all_descriptors, bc_mask) if m]
    X_bc = descriptors_to_matrix(bc_descs)
    labels_bc = np.array([d.regime for d in bc_descs])
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_bc_s = scaler.fit_transform(X_bc)
    sil = silhouette_score(X_bc_s, labels_bc)
    print(f'\nSilhouette (branch vs clustered): {sil:.3f}  '
          f'{"PASS" if sil > 0.3 else "FAIL"} (threshold > 0.3)')

    # 2. Anisotropy KS test
    aniso_branch = [d.anisotropy for d in all_descriptors
                    if d.regime == 'branch_constrained' and d.anisotropy is not None]
    aniso_clustered = [d.anisotropy for d in all_descriptors
                       if d.regime == 'clustered' and d.anisotropy is not None]
    if aniso_branch and aniso_clustered:
        ks_stat, ks_p = ks_2samp(aniso_branch, aniso_clustered)
        print(f'Anisotropy KS test: stat={ks_stat:.3f}, p={ks_p:.4g}  '
              f'{"PASS" if ks_p < 0.05 else "FAIL"} (threshold p < 0.05)')
    else:
        print('Anisotropy KS test: SKIPPED (no anisotropy data)')

    # 3. PCA visualization
    regime_space = RegimeSpace(n_components=2)
    regime_space.fit(all_descriptors)
    pca_coords = regime_space._sim_pca
    labels = regime_space._sim_labels

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'random': '#888888', 'clustered': '#e74c3c',
              'branch_constrained': '#e67e22'}
    for regime in ['random', 'clustered', 'branch_constrained']:
        mask = labels == regime
        ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], s=30,
                   c=colors[regime], label=regime.replace('_', ' '), alpha=0.6)
    ev = regime_space.explained_variance()
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
    ax.legend()
    ax.set_title('Pilot: Branch vs Clustered Separation')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_pilot_branch_test.pdf', dpi=150)
    print(f'\nSaved: {FIGURES_DIR / "fig_pilot_branch_test.pdf"}')

    # Summary
    results = {
        'n_descriptors': len(all_descriptors),
        'silhouette_branch_vs_clustered': float(sil),
        'anisotropy_ks_p': float(ks_p) if aniso_branch and aniso_clustered else None,
        'explained_variance': ev.tolist(),
        'pass_silhouette': bool(sil > 0.3),
        'pass_anisotropy': bool(ks_p < 0.05) if aniso_branch and aniso_clustered else False,
    }
    with open(RESULTS_DIR / 'pilot_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {RESULTS_DIR / "pilot_results.json"}')

    all_pass = results['pass_silhouette'] and results['pass_anisotropy']
    print(f'\n{"=" * 40}')
    print(f'PILOT RESULT: {"ALL PASS" if all_pass else "NEEDS ITERATION"}')
    print(f'{"=" * 40}')
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
