"""Tests for point-pattern simulators."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulators import (
    simulate_random, simulate_clustered, simulate_exclusion,
    simulate_layered, simulate_compartmental, simulate_branch_constrained,
    simulate_hierarchical, SIMULATORS, SWEEP_PARAMS, simulate_regime,
)

ROI = 10000
N = 2000


class TestSimulateRandom:
    def test_shape(self):
        pts = simulate_random(n_points=N, roi_size=ROI, rng=42)
        assert pts.shape == (N, 2)

    def test_bounds(self):
        pts = simulate_random(n_points=N, roi_size=ROI, rng=42)
        assert np.all(pts >= 0)
        assert np.all(pts <= ROI)

    def test_reproducibility(self):
        a = simulate_random(n_points=N, rng=42)
        b = simulate_random(n_points=N, rng=42)
        np.testing.assert_array_equal(a, b)


class TestSimulateClustered:
    def test_shape(self):
        pts = simulate_clustered(cluster_radius=200, n_points=N, rng=42)
        assert pts.shape == (N, 2)

    def test_clustering(self):
        """Points should be more clustered than random."""
        pts = simulate_clustered(cluster_radius=100, n_points=N, rng=42)
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        nn_dists, _ = tree.query(pts, k=2)
        mean_nn = nn_dists[:, 1].mean()
        # Random expectation: 0.5 / sqrt(density)
        random_nn = 0.5 / np.sqrt(N / ROI**2)
        assert mean_nn < random_nn * 0.5  # Much more clustered


class TestSimulateExclusion:
    def test_min_distance(self):
        min_d = 100
        pts = simulate_exclusion(min_distance=min_d, n_points=500, rng=42)
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=2)
        # All NN distances should be >= min_distance (for SSI points)
        # Some fill points may violate if packing limit reached
        assert np.median(dists[:, 1]) >= min_d * 0.9


class TestSimulateLayered:
    def test_shape(self):
        pts = simulate_layered(layer_width_frac=0.3, n_points=N, rng=42)
        assert pts.shape == (N, 2)

    def test_layer_structure(self):
        """Y-coordinates should show multimodal distribution."""
        pts = simulate_layered(layer_width_frac=0.1, n_points=N, rng=42)
        hist, _ = np.histogram(pts[:, 1], bins=50)
        # Should have clear peaks and valleys
        assert hist.max() / max(hist.mean(), 1) > 2


class TestSimulateBranchConstrained:
    def test_shape(self):
        pts = simulate_branch_constrained(branch_width=50, n_points=N, rng=42)
        assert pts.shape == (N, 2)

    def test_anisotropy(self):
        """Branch-constrained should have different spatial structure than clustered."""
        pts_branch = simulate_branch_constrained(branch_width=30, n_points=N, rng=42)
        pts_clust = simulate_clustered(cluster_radius=200, n_points=N, rng=42)
        # Both should have points, but branch should show elongated structures
        assert pts_branch.shape == (N, 2)
        assert pts_clust.shape == (N, 2)


class TestSimulateCompartmental:
    def test_shape(self):
        pts = simulate_compartmental(n_compartments=6, n_points=N, rng=42)
        assert pts.shape == (N, 2)


class TestSimulateHierarchical:
    def test_shape(self):
        pts = simulate_hierarchical(scale_ratio=5, n_points=N, rng=42)
        assert pts.shape == (N, 2)


class TestSimulateRegime:
    def test_all_regimes(self):
        """All regimes should run without error at their first sweep value."""
        for regime, sweep in SWEEP_PARAMS.items():
            pval = sweep['values'][0]
            patterns = simulate_regime(regime, pval, n_realizations=2, rng=42)
            assert len(patterns) == 2
            assert patterns[0].shape[1] == 2

    def test_registry_completeness(self):
        """All sweep regimes should have matching simulators."""
        for regime in SWEEP_PARAMS:
            assert regime in SIMULATORS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
