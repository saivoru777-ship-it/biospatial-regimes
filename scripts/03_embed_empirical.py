#!/usr/bin/env python3
"""
Load empirical data, extract descriptors, project into simulation-defined PCA/UMAP.
Compute KDE-based soft regime affinity profiles + NN-based out-of-regime flags.
Save results/empirical_embedding.json.
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.descriptors import dicts_to_descriptors, descriptors_to_matrix, descriptors_to_dicts
from src.empirical_loaders import load_all_empirical
from src.embedding import RegimeSpace, project_umap

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def main():
    # Load simulation descriptors and rebuild regime space
    with open(RESULTS_DIR / 'simulated_descriptors.json') as f:
        sim_data = json.load(f)
    sim_descriptors = dicts_to_descriptors(sim_data['descriptors'])
    print(f'Loaded {len(sim_descriptors)} simulation descriptors')

    regime_space = RegimeSpace(n_components=3)
    regime_space.fit(sim_descriptors)

    # Load empirical
    emp_descriptors = load_all_empirical()
    if not emp_descriptors:
        print('No empirical data loaded. Exiting.')
        return

    # Project into PCA space
    emp_pca = regime_space.transform(emp_descriptors)
    print(f'\nEmpirical PCA coordinates shape: {emp_pca.shape}')

    # Regime affinities
    affinities, regime_names = regime_space.regime_affinity(emp_descriptors)
    print(f'\nRegime affinity profiles:')
    for i, desc in enumerate(emp_descriptors):
        top_regime = regime_names[np.argmax(affinities[i])]
        top_aff = np.max(affinities[i])
        print(f'  {desc.label:40s} → {top_regime} ({top_aff:.2f})')

    # Out-of-regime flags
    oor_flags = regime_space.out_of_regime(emp_descriptors)
    n_oor = oor_flags.sum()
    print(f'\nOut-of-regime: {n_oor}/{len(emp_descriptors)} flagged')
    for i, (desc, flag) in enumerate(zip(emp_descriptors, oor_flags)):
        if flag:
            print(f'  [OOR] {desc.label}')

    # UMAP projection
    umap_embedding = None
    umap_pkl = RESULTS_DIR / 'umap_model.pkl'
    if umap_pkl.exists():
        with open(umap_pkl, 'rb') as f:
            umap_data = pickle.load(f)
        emp_umap = project_umap(umap_data['model'], umap_data['scaler'],
                                 emp_descriptors)
        umap_embedding = emp_umap.tolist()
        print(f'UMAP projection shape: {emp_umap.shape}')

    # Save
    output = {
        'n_empirical': len(emp_descriptors),
        'descriptors': descriptors_to_dicts(emp_descriptors),
        'pca_coordinates': emp_pca.tolist(),
        'affinities': affinities.tolist(),
        'regime_names': regime_names,
        'out_of_regime': oor_flags.tolist(),
        'umap_coordinates': umap_embedding,
    }
    out_path = RESULTS_DIR / 'empirical_embedding.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
