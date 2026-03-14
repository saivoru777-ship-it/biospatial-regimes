#!/usr/bin/env python3
"""
Generate all figures from precomputed results.
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.descriptors import dicts_to_descriptors
from src.embedding import RegimeSpace
from src.viz import (
    fig_simulated_patterns, fig_phase_diagram, fig_regime_affinity,
    fig_descriptor_distributions, fig_umap_embedding, fig_separability,
    fig_pca_loadings, fig_mixture_validation,
)
from src.simulators import SIMULATORS, SWEEP_PARAMS

import matplotlib
matplotlib.use('Agg')

RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    # Load simulation descriptors
    print('Loading simulation descriptors...')
    with open(RESULTS_DIR / 'simulated_descriptors.json') as f:
        sim_data = json.load(f)
    sim_descriptors = dicts_to_descriptors(sim_data['descriptors'])
    print(f'  {len(sim_descriptors)} simulation descriptors')

    # Rebuild regime space
    regime_space = RegimeSpace(n_components=3)
    regime_space.fit(sim_descriptors)

    # Load empirical
    emp_descriptors = None
    emp_path = RESULTS_DIR / 'empirical_embedding.json'
    if emp_path.exists():
        print('Loading empirical embedding...')
        with open(emp_path) as f:
            emp_data = json.load(f)
        emp_descriptors = dicts_to_descriptors(emp_data['descriptors'])
        print(f'  {len(emp_descriptors)} empirical descriptors')

    # --- Figure 1: Simulated patterns ---
    print('\nGenerating fig_simulated_patterns...')
    patterns = {}
    rng = np.random.default_rng(42)
    for regime, func in SIMULATORS.items():
        sweep = SWEEP_PARAMS[regime]
        mid_val = sweep['values'][len(sweep['values']) // 2]
        kwargs = {sweep['param']: mid_val, 'rng': 42, 'roi_size': 10000}
        if regime != 'random':
            kwargs['n_points'] = 2000
        patterns[regime] = func(**kwargs)
    fig_simulated_patterns(patterns)

    # --- Figure 4: Phase diagram ---
    print('Generating fig_phase_diagram...')
    fig_phase_diagram(regime_space, sim_descriptors, emp_descriptors)

    # --- Figure 5: Regime affinity ---
    if emp_descriptors:
        print('Generating fig_regime_affinity...')
        affinities, regime_names = regime_space.regime_affinity(emp_descriptors)
        fig_regime_affinity(affinities, regime_names, emp_descriptors)

    # --- Figure 6: Mixture validation ---
    mix_path = RESULTS_DIR / 'mixture_validation.json'
    if mix_path.exists():
        print('Generating fig_mixture_validation...')
        with open(mix_path) as f:
            mix_results = json.load(f)
        regime_names_all = sorted(SWEEP_PARAMS.keys())
        true_fracs = []
        recovered_fracs = []
        mix_labels = []
        for mr in mix_results:
            mix_labels.append(mr['name'])
            true_fracs.append([mr['true_fractions'].get(r, 0.0)
                               for r in regime_names_all])
            recovered_fracs.append([mr['recovered_affinities'].get(r, 0.0)
                                    for r in regime_names_all])
        fig_mixture_validation(true_fracs, recovered_fracs, mix_labels,
                                regime_names_all)

    # --- Figure 7: Descriptor distributions ---
    print('Generating fig_descriptor_distributions...')
    fig_descriptor_distributions(sim_descriptors)

    # --- Figure 8: UMAP ---
    umap_path = RESULTS_DIR / 'umap_embedding.json'
    if umap_path.exists():
        print('Generating fig_umap_embedding...')
        with open(umap_path) as f:
            umap_data = json.load(f)
        sim_umap = np.array(umap_data['embedding'])
        sim_labels = umap_data['labels']
        emp_umap = None
        if emp_descriptors and emp_path.exists():
            with open(emp_path) as f:
                emp_emb = json.load(f)
            if emp_emb.get('umap_coordinates'):
                emp_umap = np.array(emp_emb['umap_coordinates'])
        fig_umap_embedding(sim_umap, sim_labels, emp_umap, emp_descriptors)

    # --- Figure 9: Separability ---
    print('Generating fig_separability...')
    sil = regime_space.silhouette()
    mahal, regime_names = regime_space.pairwise_mahalanobis()
    fig_separability(sil, mahal, regime_names)

    # --- Figure 10: PCA loadings ---
    print('Generating fig_pca_loadings...')
    fig_pca_loadings(regime_space)

    print(f'\nAll figures saved to {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
