#!/usr/bin/env python3
"""
Build regime embedding space from simulation descriptors.
PCA/UMAP on simulation descriptors. Compute silhouette score, Mahalanobis distances.
Save results/regime_space.json.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.descriptors import dicts_to_descriptors, descriptors_to_matrix
from src.embedding import RegimeSpace, fit_umap

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def main():
    # Load simulation descriptors
    with open(RESULTS_DIR / 'simulated_descriptors.json') as f:
        data = json.load(f)
    descriptors = dicts_to_descriptors(data['descriptors'])
    print(f'Loaded {len(descriptors)} simulation descriptors')

    # Build PCA regime space
    print('\nFitting PCA regime space...')
    regime_space = RegimeSpace(n_components=3)
    regime_space.fit(descriptors)

    # Report
    ev = regime_space.explained_variance()
    print(f'Explained variance: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}, PC3={ev[2]:.3f}')
    print(f'Total (3 PCs): {sum(ev):.3f}')

    sil = regime_space.silhouette()
    print(f'Silhouette score: {sil:.3f}')

    mahal, regime_names = regime_space.pairwise_mahalanobis()
    print(f'\nPairwise Mahalanobis distances:')
    for i, r1 in enumerate(regime_names):
        for j, r2 in enumerate(regime_names):
            if j > i:
                print(f'  {r1:25s} vs {r2:25s}: {mahal[i,j]:.2f}')

    # Save PCA space
    regime_space.save(RESULTS_DIR / 'regime_space.json')
    print(f'\nSaved: {RESULTS_DIR / "regime_space.json"}')

    # UMAP (supplementary — skip if umap-learn not installed)
    try:
        from src.embedding import fit_umap
        print('\nFitting UMAP...')
        umap_model, umap_embedding, umap_scaler = fit_umap(descriptors)
        labels = [d.regime for d in descriptors]
        umap_data = {
            'embedding': umap_embedding.tolist(),
            'labels': labels,
        }
        with open(RESULTS_DIR / 'umap_embedding.json', 'w') as f:
            json.dump(umap_data, f, indent=2)
        print(f'Saved: {RESULTS_DIR / "umap_embedding.json"}')
        import pickle
        with open(RESULTS_DIR / 'umap_model.pkl', 'wb') as f:
            pickle.dump({'model': umap_model, 'scaler': umap_scaler}, f)
        print(f'Saved: {RESULTS_DIR / "umap_model.pkl"}')
    except ImportError:
        print('\nSkipping UMAP (umap-learn not installed)')

    # Verification checks
    print(f'\n{"=" * 40}')
    print('Verification:')
    print(f'  3 PCs explain > 80% variance: '
          f'{"PASS" if sum(ev) > 0.8 else "FAIL"} ({sum(ev)*100:.1f}%)')
    print(f'  Silhouette > 0.5: '
          f'{"PASS" if sil > 0.5 else "FAIL"} ({sil:.3f})')
    print(f'{"=" * 40}')


if __name__ == '__main__':
    main()
