#!/usr/bin/env python3
"""
Mixture sanity test: simulate explicit mixtures (layered+clustered,
compartmental+exclusion). Check that KDE affinities recover the known
mixture proportions. Validates the affinity framework itself.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulators import (simulate_layered, simulate_clustered,
                             simulate_compartmental, simulate_exclusion)
from src.fingerprinting import compute_fingerprint
from src.descriptors import (extract_descriptors, dicts_to_descriptors,
                              descriptors_to_dicts)
from src.embedding import RegimeSpace

RESULTS_DIR = Path(__file__).parent.parent / 'results'

# Mixture definitions: (name, [(regime, func, kwargs, fraction), ...])
MIXTURES = [
    ('layered_70_clustered_30', [
        ('layered', simulate_layered, {'layer_width_frac': 0.3}, 0.7),
        ('clustered', simulate_clustered, {'cluster_radius': 300}, 0.3),
    ]),
    ('compartmental_50_exclusion_50', [
        ('compartmental', simulate_compartmental, {'n_compartments': 8}, 0.5),
        ('exclusion', simulate_exclusion, {'min_distance': 100}, 0.5),
    ]),
    ('clustered_80_random_20', [
        ('clustered', simulate_clustered, {'cluster_radius': 200}, 0.8),
        ('random', lambda **kw: np.random.default_rng(kw.get('rng')).uniform(
            0, kw.get('roi_size', 10000), size=(kw.get('n_points', 2000), 2)),
         {}, 0.2),
    ]),
]

N_REALIZATIONS = 20
N_POINTS = 2000
ROI_SIZE = 10000


def simulate_mixture(components, rng):
    """Simulate a point pattern from a mixture of regimes."""
    rng = np.random.default_rng(rng)
    all_points = []
    for regime, func, kwargs, fraction in components:
        n = int(N_POINTS * fraction)
        seed = rng.integers(0, 2**31)
        pts = func(n_points=n, roi_size=ROI_SIZE, rng=seed, **kwargs)
        all_points.append(pts[:n])
    return np.vstack(all_points)


def main():
    # Load simulation descriptors and rebuild regime space
    with open(RESULTS_DIR / 'simulated_descriptors.json') as f:
        sim_data = json.load(f)
    sim_descriptors = dicts_to_descriptors(sim_data['descriptors'])
    print(f'Loaded {len(sim_descriptors)} simulation descriptors')

    regime_space = RegimeSpace(n_components=3)
    regime_space.fit(sim_descriptors)

    rng = np.random.default_rng(123)
    mixture_results = []

    for mix_name, components in MIXTURES:
        print(f'\n=== Mixture: {mix_name} ===')
        true_fractions = {}
        for regime, _, _, frac in components:
            true_fractions[regime] = true_fractions.get(regime, 0) + frac

        mix_descriptors = []
        for i in range(N_REALIZATIONS):
            seed = rng.integers(0, 2**31)
            pattern = simulate_mixture(components, rng=seed)
            fp = compute_fingerprint(pattern, roi_size=ROI_SIZE, n_null=20, rng=seed)
            desc = extract_descriptors(
                vmr_ratio=fp.vmr_ratio,
                scales=fp.scales,
                label=f'mixture_{mix_name}_r{i}',
                source='simulation',
                regime='mixture',
                positions=pattern,
                parameters={'mixture': mix_name, 'realization': i},
            )
            mix_descriptors.append(desc)

        # Get affinities
        affinities, regime_names = regime_space.regime_affinity(mix_descriptors)
        mean_affinities = affinities.mean(axis=0)

        print(f'  True proportions:')
        for regime, frac in true_fractions.items():
            print(f'    {regime}: {frac:.2f}')
        print(f'  Recovered affinities:')
        for j, rname in enumerate(regime_names):
            print(f'    {rname}: {mean_affinities[j]:.3f}')

        # Check recovery
        true_vec = np.array([true_fractions.get(r, 0.0) for r in regime_names])
        error = np.abs(mean_affinities - true_vec).max()
        print(f'  Max error: {error:.3f} '
              f'{"PASS" if error < 0.2 else "FAIL"} (threshold < 0.2)')

        mixture_results.append({
            'name': mix_name,
            'true_fractions': {r: float(true_fractions.get(r, 0.0))
                               for r in regime_names},
            'recovered_affinities': {r: float(mean_affinities[j])
                                     for j, r in enumerate(regime_names)},
            'max_error': float(error),
            'pass': bool(error < 0.2),
        })

    # Save
    out_path = RESULTS_DIR / 'mixture_validation.json'
    with open(out_path, 'w') as f:
        json.dump(mixture_results, f, indent=2)
    print(f'\nSaved: {out_path}')

    all_pass = all(r['pass'] for r in mixture_results)
    print(f'\n{"=" * 40}')
    print(f'MIXTURE VALIDATION: {"ALL PASS" if all_pass else "SOME FAIL"}')
    print(f'{"=" * 40}')


if __name__ == '__main__':
    main()
