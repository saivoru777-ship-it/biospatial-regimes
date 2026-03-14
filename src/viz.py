"""
Visualization functions for biospatial regimes.

All figures saved as PDF to figures/ directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

REGIME_COLORS = {
    'random': '#888888',
    'clustered': '#e74c3c',
    'exclusion': '#3498db',
    'layered': '#2ecc71',
    'compartmental': '#9b59b6',
    'branch_constrained': '#e67e22',
    'hierarchical': '#1abc9c',
}

SOURCE_MARKERS = {
    'tissue': '*',
    'smlm': '^',
    'dendrite': 'D',
    'simulation': 'o',
}

DESCRIPTOR_NAMES = [
    'peak_height', 'threshold_scale', 'width',
    'small_scale_slope', 'large_scale_slope', 'integrated_excess',
]


def fig_simulated_patterns(patterns_by_regime, save=True):
    """7-panel gallery of example point patterns (1 per regime).

    Parameters
    ----------
    patterns_by_regime : dict
        {regime_name: (N, 2) array}
    """
    fig, axes = plt.subplots(1, 7, figsize=(28, 4))
    for ax, (regime, pts) in zip(axes, sorted(patterns_by_regime.items())):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c=REGIME_COLORS.get(regime, 'k'),
                   alpha=0.5, rasterized=True)
        ax.set_title(regime.replace('_', ' '), fontsize=10)
        ax.set_xlim(0, 10000)
        ax.set_ylim(0, 10000)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Simulated Point Patterns', fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_simulated_patterns.pdf', dpi=150)
    return fig


def fig_regime_fingerprints(ensembles_by_regime, save=True):
    """7-panel VMR ratio curves: parameter sweep as color gradient.

    Parameters
    ----------
    ensembles_by_regime : dict
        {regime_name: list of RegimeEnsemble}
    """
    fig, axes = plt.subplots(1, 7, figsize=(28, 4), sharey=True)
    for ax, (regime, ensembles) in zip(axes, sorted(ensembles_by_regime.items())):
        n_ens = len(ensembles)
        cmap = plt.cm.viridis
        for i, ens in enumerate(ensembles):
            color = cmap(i / max(1, n_ens - 1))
            mean_ratio = ens.mean_vmr_ratio
            scales = ens.scales
            ax.plot(scales, np.log2(mean_ratio), color=color, alpha=0.7, lw=1.5)
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_title(regime.replace('_', ' '), fontsize=10)
        ax.set_xlabel('Scale')
        ax.set_xscale('log')
    axes[0].set_ylabel('log₂(VMR ratio)')
    fig.suptitle('VMR Fingerprints Across Parameter Sweeps', fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_regime_fingerprints.pdf', dpi=150)
    return fig


def fig_regime_comparison(ensembles_by_regime, save=True):
    """All 7 regime mean curves overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, ensembles in sorted(ensembles_by_regime.items()):
        # Use middle parameter value ensemble
        mid = len(ensembles) // 2
        ens = ensembles[mid]
        mean_ratio = ens.mean_vmr_ratio
        scales = ens.scales
        ax.plot(scales, np.log2(mean_ratio), color=REGIME_COLORS.get(regime, 'k'),
                lw=2, label=regime.replace('_', ' '))
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Scale')
    ax.set_ylabel('log₂(VMR ratio)')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.set_title('Regime Comparison (Mid-Parameter)')
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_regime_comparison.pdf', dpi=150)
    return fig


def fig_phase_diagram(regime_space, sim_descriptors, emp_descriptors=None,
                       save=True):
    """Key figure: PC1 vs PC2, clustering manifold vs null cloud.

    Visually distinguishes:
    - Clustering manifold (branch_constrained, hierarchical, compartmental, clustered)
    - Null cloud (random, exclusion, layered)
    - Empirical projections by source

    Parameters
    ----------
    regime_space : RegimeSpace
    sim_descriptors : list of FingerprintDescriptor
    emp_descriptors : list of FingerprintDescriptor, optional
    """
    sim_pca = regime_space._sim_pca
    sim_labels = regime_space._sim_labels

    MANIFOLD_REGIMES = {'branch_constrained', 'hierarchical', 'compartmental', 'clustered'}
    NULL_REGIMES = {'random', 'exclusion', 'layered'}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw null cloud boundary (light gray shaded region)
    null_mask = np.isin(sim_labels, list(NULL_REGIMES))
    if null_mask.any():
        from matplotlib.patches import Ellipse
        null_pts = sim_pca[null_mask, :2]  # First 2 PCs only
        null_center = null_pts.mean(axis=0)
        null_cov = np.cov(null_pts.T)
        eigvals, eigvecs = np.linalg.eigh(null_cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        # 2-sigma ellipse
        w, h = 2 * 2 * np.sqrt(eigvals)
        ellipse = Ellipse(null_center[:2], w, h, angle=angle,
                          facecolor='#eeeeee', edgecolor='#aaaaaa',
                          linestyle='--', linewidth=1.5, zorder=0,
                          label='null cloud')
        ax.add_patch(ellipse)

    # Simulation clouds — manifold regimes (saturated)
    for regime in sorted(MANIFOLD_REGIMES):
        mask = sim_labels == regime
        if not mask.any():
            continue
        ax.scatter(sim_pca[mask, 0], sim_pca[mask, 1], s=12, alpha=0.4,
                   c=REGIME_COLORS.get(regime, 'k'), label=regime.replace('_', ' '),
                   rasterized=True)

    # Simulation clouds — null regimes (desaturated, smaller)
    for regime in sorted(NULL_REGIMES):
        mask = sim_labels == regime
        if not mask.any():
            continue
        ax.scatter(sim_pca[mask, 0], sim_pca[mask, 1], s=6, alpha=0.25,
                   c=REGIME_COLORS.get(regime, 'k'),
                   label=regime.replace('_', ' ') + ' (null)',
                   rasterized=True)

    # Empirical projections — colored by source with distinct markers
    SOURCE_COLORS = {'tissue': '#d4ac0d', 'smlm': '#e74c3c', 'dendrite': '#2980b9'}
    if emp_descriptors:
        emp_pca = regime_space.transform(emp_descriptors)
        sources = sorted(set(d.source for d in emp_descriptors))
        for source in sources:
            mask = np.array([d.source == source for d in emp_descriptors])
            marker = SOURCE_MARKERS.get(source, 's')
            ax.scatter(emp_pca[mask, 0], emp_pca[mask, 1], s=120,
                       marker=marker, edgecolors='k', linewidths=0.8,
                       c=SOURCE_COLORS.get(source, 'gold'), zorder=10,
                       label=f'{source} (empirical)')

    ev = regime_space.explained_variance()
    ax.set_xlabel(f'PC1 — clustering intensity ({ev[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 — scale structure ({ev[1]*100:.1f}%)', fontsize=11)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.set_title('Biospatial Regime Space: Clustering Manifold + Null Cloud', fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_phase_diagram.pdf', dpi=150)
    return fig


def fig_regime_affinity(affinities, regime_names, emp_descriptors, save=True):
    """Heatmap: empirical fingerprints × 7 regimes."""
    labels = [d.label for d in emp_descriptors]
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    im = ax.imshow(affinities, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(regime_names)))
    ax.set_xticklabels([r.replace('_', ' ') for r in regime_names],
                        rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel('Regime')
    ax.set_ylabel('Empirical Fingerprint')
    ax.set_title('KDE-Based Regime Affinity')
    plt.colorbar(im, ax=ax, label='Affinity')
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_regime_affinity.pdf', dpi=150)
    return fig


def fig_mixture_validation(true_fractions, recovered_fractions, mixture_labels,
                            regime_names, save=True):
    """Known mixtures vs recovered affinities."""
    n_mix = len(mixture_labels)
    n_reg = len(regime_names)
    fig, axes = plt.subplots(1, n_mix, figsize=(4 * n_mix, 4), sharey=True)
    if n_mix == 1:
        axes = [axes]
    for ax, label, true, recovered in zip(axes, mixture_labels, true_fractions,
                                           recovered_fractions):
        x = np.arange(n_reg)
        width = 0.35
        ax.bar(x - width / 2, true, width, label='True', color='steelblue')
        ax.bar(x + width / 2, recovered, width, label='Recovered', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace('_', ' ') for r in regime_names],
                            rotation=45, ha='right', fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Fraction')
    fig.suptitle('Mixture Validation', fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_mixture_validation.pdf', dpi=150)
    return fig


def fig_descriptor_distributions(sim_descriptors, save=True):
    """6-panel histograms of each primary descriptor, colored by regime."""
    from .descriptors import descriptors_to_matrix
    X = descriptors_to_matrix(sim_descriptors)
    labels = np.array([d.regime for d in sim_descriptors])
    regimes = sorted(set(labels))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, (ax, name) in enumerate(zip(axes.flat, DESCRIPTOR_NAMES)):
        for regime in regimes:
            mask = labels == regime
            ax.hist(X[mask, idx], bins=30, alpha=0.5,
                    color=REGIME_COLORS.get(regime, 'k'),
                    label=regime.replace('_', ' '))
        ax.set_xlabel(name)
        ax.set_ylabel('Count')
    axes[0, 0].legend(fontsize=6)
    fig.suptitle('Descriptor Distributions by Regime', fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_descriptor_distributions.pdf', dpi=150)
    return fig


def fig_umap_embedding(sim_embedding, sim_labels, emp_embedding=None,
                        emp_descriptors=None, save=True):
    """UMAP of combined dataset (supplementary)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(sim_labels))
    for regime in unique_labels:
        mask = np.array(sim_labels) == regime
        ax.scatter(sim_embedding[mask, 0], sim_embedding[mask, 1], s=8,
                   alpha=0.3, c=REGIME_COLORS.get(regime, 'k'),
                   label=regime.replace('_', ' '), rasterized=True)
    if emp_embedding is not None and emp_descriptors is not None:
        sources = set(d.source for d in emp_descriptors)
        for source in sorted(sources):
            mask = np.array([d.source == source for d in emp_descriptors])
            marker = SOURCE_MARKERS.get(source, 's')
            ax.scatter(emp_embedding[mask, 0], emp_embedding[mask, 1], s=100,
                       marker=marker, edgecolors='k', linewidths=0.5,
                       c='gold', zorder=10, label=f'{source} (empirical)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=8, loc='best')
    ax.set_title('UMAP Embedding')
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_umap_embedding.pdf', dpi=150)
    return fig


def fig_separability(silhouette, mahal_distances, regime_names, save=True):
    """Silhouette scores + inter-regime Mahalanobis distances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette
    ax = axes[0]
    ax.barh([0], [silhouette], color='steelblue')
    ax.set_xlim(0, 1)
    ax.set_yticks([0])
    ax.set_yticklabels(['Overall'])
    ax.set_xlabel('Silhouette Score')
    ax.set_title(f'Silhouette Score: {silhouette:.3f}')

    # Mahalanobis heatmap
    ax = axes[1]
    n = len(regime_names)
    im = ax.imshow(mahal_distances, cmap='YlOrRd')
    ax.set_xticks(range(n))
    ax.set_xticklabels([r.replace('_', ' ') for r in regime_names],
                        rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r.replace('_', ' ') for r in regime_names], fontsize=8)
    ax.set_title('Pairwise Mahalanobis Distance')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_separability.pdf', dpi=150)
    return fig


def fig_pca_loadings(regime_space, save=True):
    """How each PC loads on the 6 descriptors."""
    loadings = regime_space.loadings()
    n_components = loadings.shape[0]
    ev = regime_space.explained_variance()

    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 4),
                              sharey=True)
    if n_components == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.barh(range(6), loadings[i], color='steelblue')
        ax.set_yticks(range(6))
        ax.set_yticklabels(DESCRIPTOR_NAMES, fontsize=9)
        ax.set_xlabel('Loading')
        ax.set_title(f'PC{i+1} ({ev[i]*100:.1f}%)')
        ax.axvline(0, color='gray', lw=0.5)
    fig.suptitle('PCA Loadings on Fingerprint Descriptors', fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'fig_pca_loadings.pdf', dpi=150)
    return fig
