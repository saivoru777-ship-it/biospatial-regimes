"""Tests for embedding and regime space."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.descriptors import extract_descriptors, FingerprintDescriptor
from src.embedding import RegimeSpace


def _make_descriptors(n_per_regime=20, regimes=None, rng_seed=42):
    """Generate synthetic descriptors with known regime separation."""
    rng = np.random.default_rng(rng_seed)
    if regimes is None:
        regimes = ['random', 'clustered', 'exclusion']

    # Create descriptor-like VMR curves with regime-specific signatures
    regime_signatures = {
        'random': lambda: np.ones(12) + rng.normal(0, 0.05, 12),
        'clustered': lambda: np.array([1, 1.5, 3, 5, 4, 2, 1.2, 1, 1, 1, 1, 1])
                             + rng.normal(0, 0.3, 12),
        'exclusion': lambda: np.array([0.3, 0.5, 0.7, 0.9, 1, 1, 1, 1, 1, 1, 1, 1])
                             + rng.normal(0, 0.1, 12),
        'layered': lambda: np.array([1, 1, 1, 1, 2, 4, 3, 2, 1, 1, 1, 1])
                           + rng.normal(0, 0.2, 12),
        'compartmental': lambda: np.array([1, 2, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1])
                                 + rng.normal(0, 0.2, 12),
        'branch_constrained': lambda: np.array([1, 2, 3, 4, 3, 2, 1, 1, 1, 1, 1, 1])
                                      + rng.normal(0, 0.3, 12),
        'hierarchical': lambda: np.array([1, 3, 2, 1, 3, 2, 1, 1, 1, 1, 1, 1])
                                + rng.normal(0, 0.2, 12),
    }

    descriptors = []
    scales = np.logspace(1, 3, 12)
    for regime in regimes:
        sig_func = regime_signatures.get(regime, regime_signatures['random'])
        for i in range(n_per_regime):
            vmr_ratio = np.maximum(sig_func(), 0.01)
            desc = extract_descriptors(
                vmr_ratio=vmr_ratio,
                scales=scales,
                label=f'{regime}_{i}',
                source='simulation',
                regime=regime,
                parameters={'i': i},
            )
            descriptors.append(desc)
    return descriptors


class TestRegimeSpace:
    def test_fit(self):
        descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(descs)
        assert rs._fitted

    def test_transform(self):
        descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(descs)
        pca = rs.transform(descs[:5])
        assert pca.shape == (5, 3)

    def test_silhouette(self):
        descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(descs)
        sil = rs.silhouette()
        assert sil > 0  # Should be positive for separable regimes

    def test_explained_variance(self):
        descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(descs)
        ev = rs.explained_variance()
        assert len(ev) == 3
        assert all(0 <= v <= 1 for v in ev)
        assert sum(ev) <= 1.0 + 1e-6

    def test_regime_affinity(self):
        descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(descs)
        affinities, names = rs.regime_affinity(descs[:3])
        assert affinities.shape[0] == 3
        assert affinities.shape[1] == len(names)
        # Rows should sum to 1
        np.testing.assert_allclose(affinities.sum(axis=1), 1.0, atol=1e-6)

    def test_out_of_regime(self):
        descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(descs)
        # Test with known in-regime points
        flags = rs.out_of_regime(descs[:5])
        assert flags.shape == (5,)
        assert flags.dtype == bool

    def test_pairwise_mahalanobis(self):
        descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(descs)
        dists, names = rs.pairwise_mahalanobis()
        n = len(names)
        assert dists.shape == (n, n)
        # Diagonal should be 0
        np.testing.assert_allclose(np.diag(dists), 0.0)
        # Should be symmetric
        np.testing.assert_allclose(dists, dists.T)

    def test_loadings(self):
        descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(descs)
        loadings = rs.loadings()
        assert loadings.shape == (3, 6)

    def test_project_empirical_different_from_sim(self):
        """Empirical projections should work even with different data."""
        sim_descs = _make_descriptors(n_per_regime=30)
        rs = RegimeSpace(n_components=3)
        rs.fit(sim_descs)

        # Create "empirical" descriptors
        emp_descs = _make_descriptors(n_per_regime=5, rng_seed=99)
        for d in emp_descs:
            d.source = 'tissue'
        pca = rs.transform(emp_descs)
        assert pca.shape == (15, 3)  # 5 * 3 regimes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
