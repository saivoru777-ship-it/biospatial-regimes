"""Tests for fingerprint descriptor extraction."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.descriptors import (
    extract_descriptors, descriptors_to_matrix, FingerprintDescriptor,
    _compute_fwhm, _compute_anisotropy, _compute_multimodality,
    _compute_threshold_scale,
)


class TestExtractDescriptors:
    def test_flat_curve(self):
        """VMR ratio of all 1s should give near-zero descriptors."""
        vmr_ratio = np.ones(12)
        scales = np.logspace(1, 3, 12)
        desc = extract_descriptors(vmr_ratio, scales, label='test_flat',
                                    regime='random')
        assert abs(desc.peak_height) < 0.01
        assert abs(desc.integrated_excess) < 0.1
        assert abs(desc.small_scale_slope) < 0.1
        assert abs(desc.large_scale_slope) < 0.1

    def test_peaked_curve(self):
        """Peaked VMR ratio should have positive peak_height."""
        vmr_ratio = np.array([1, 1.5, 3, 5, 4, 2, 1.2, 1, 1, 1, 1, 1])
        scales = np.logspace(1, 3, 12)
        desc = extract_descriptors(vmr_ratio, scales, label='test_peaked',
                                    regime='clustered')
        assert desc.peak_height > 1  # log2(5) ≈ 2.3
        # threshold_scale should be early (small scale) for clustered
        assert desc.threshold_scale < 0.5

    def test_flat_curve_threshold(self):
        """Flat VMR ratio should have threshold_scale = 1.0 (never crosses)."""
        vmr_ratio = np.ones(12)
        scales = np.logspace(1, 3, 12)
        desc = extract_descriptors(vmr_ratio, scales, label='test_flat_ts',
                                    regime='random')
        assert desc.threshold_scale == 1.0

    def test_dip_curve(self):
        """Exclusion-like dip should have specific characteristics."""
        vmr_ratio = np.array([0.3, 0.5, 0.7, 0.9, 1, 1, 1, 1, 1, 1, 1, 1])
        scales = np.logspace(1, 3, 12)
        desc = extract_descriptors(vmr_ratio, scales, label='test_dip',
                                    regime='exclusion')
        # peak_height is max of log2, which for values > 1 will be near 0
        # The dip creates negative log2 values but peak_height is max
        assert desc.integrated_excess < 0  # Net negative excess

    def test_short_curve(self):
        """Should work with as few as 3 scales."""
        vmr_ratio = np.array([2, 4, 1])
        scales = np.array([10, 100, 1000])
        desc = extract_descriptors(vmr_ratio, scales, label='test_short')
        assert desc.peak_height > 0
        assert desc.primary_vector().shape == (6,)

    def test_primary_vector(self):
        """Primary vector should be 6D."""
        vmr_ratio = np.ones(12)
        scales = np.logspace(1, 3, 12)
        desc = extract_descriptors(vmr_ratio, scales)
        vec = desc.primary_vector()
        assert vec.shape == (6,)

    def test_with_positions(self):
        """Anisotropy should be computed when positions provided."""
        vmr_ratio = np.ones(12)
        scales = np.logspace(1, 3, 12)
        # Elongated pattern (anisotropic)
        positions = np.column_stack([
            np.random.default_rng(42).uniform(0, 10000, 100),
            np.random.default_rng(42).uniform(0, 1000, 100),
        ])
        desc = extract_descriptors(vmr_ratio, scales, positions=positions)
        assert desc.anisotropy is not None
        assert desc.anisotropy < 0.5  # Should be anisotropic

    def test_without_positions(self):
        """Anisotropy should be None without positions."""
        vmr_ratio = np.ones(12)
        scales = np.logspace(1, 3, 12)
        desc = extract_descriptors(vmr_ratio, scales)
        assert desc.anisotropy is None


class TestDescriptorsToMatrix:
    def test_shape(self):
        descs = []
        for i in range(5):
            vmr_ratio = np.random.default_rng(i).uniform(0.5, 5, 12)
            scales = np.logspace(1, 3, 12)
            descs.append(extract_descriptors(vmr_ratio, scales, label=f'd{i}'))
        X = descriptors_to_matrix(descs)
        assert X.shape == (5, 6)


class TestFWHM:
    def test_zero_curve(self):
        assert _compute_fwhm(np.zeros(10), np.linspace(0, 1, 10)) == 0.0

    def test_peaked(self):
        log_ratio = np.array([0, 0, 1, 2, 4, 2, 1, 0, 0, 0])
        log_scales = np.linspace(0, 2, 10)
        fwhm = _compute_fwhm(log_ratio, log_scales)
        assert fwhm > 0


class TestAnisotropy:
    def test_isotropic(self):
        rng = np.random.default_rng(42)
        pos = rng.uniform(0, 100, size=(500, 2))
        aniso = _compute_anisotropy(pos)
        assert aniso > 0.7  # Should be close to 1

    def test_anisotropic(self):
        rng = np.random.default_rng(42)
        pos = np.column_stack([
            rng.uniform(0, 1000, 500),
            rng.uniform(0, 10, 500),
        ])
        aniso = _compute_anisotropy(pos)
        assert aniso < 0.1  # Very anisotropic

    def test_none_input(self):
        assert _compute_anisotropy(None) is None


class TestThresholdScale:
    def test_never_crosses(self):
        """All-zero log_ratio should return 1.0."""
        log_ratio = np.zeros(10)
        norm_scales = np.linspace(0, 1, 10)
        assert _compute_threshold_scale(log_ratio, norm_scales) == 1.0

    def test_immediate_crossing(self):
        """Strong signal at first scale should return 0.0."""
        log_ratio = np.array([2, 3, 4, 3, 2, 1, 0, 0, 0, 0])
        norm_scales = np.linspace(0, 1, 10)
        assert _compute_threshold_scale(log_ratio, norm_scales) == 0.0

    def test_late_crossing(self):
        """Signal only at large scales should return high value."""
        log_ratio = np.array([0, 0, 0, 0, 0, 0, 0, 0.1, 0.3, 1.0])
        norm_scales = np.linspace(0, 1, 10)
        ts = _compute_threshold_scale(log_ratio, norm_scales)
        assert ts > 0.8

    def test_negative_crossing(self):
        """Negative signal (exclusion) should also trigger crossing."""
        log_ratio = np.array([-1, -0.5, 0, 0, 0, 0, 0, 0, 0, 0])
        norm_scales = np.linspace(0, 1, 10)
        ts = _compute_threshold_scale(log_ratio, norm_scales)
        assert ts == 0.0  # |log_ratio[0]| = 1 > 0.5


class TestMultimodality:
    def test_single_peak(self):
        log_ratio = np.array([0, 0, 1, 3, 2, 0, 0, 0, 0, 0])
        mm = _compute_multimodality(log_ratio)
        assert mm == 0.0  # Single peak, no second

    def test_two_peaks(self):
        log_ratio = np.array([0, 3, 1, 0, 0, 1, 3, 0, 0, 0])
        mm = _compute_multimodality(log_ratio)
        assert mm > 0.5  # Two similar peaks


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
