"""
VMR fingerprint computation wrapping TissueMultiscaleTest.

Computes VMR ratio curves (vmr / null_mean) at multiple scales for
arbitrary 2D point patterns.
"""

import sys
import importlib
import numpy as np
from os.path import expanduser
from dataclasses import dataclass, field
from typing import List, Optional

# Import from spatial-transcriptomics-fingerprints without conflicting with local src/
def _import_spatial_statistics():
    import importlib.util
    module_path = expanduser(
        '~/research/spatial-transcriptomics-fingerprints/src/spatial_statistics.py'
    )
    spec = importlib.util.spec_from_file_location('spatial_statistics', module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ss = _import_spatial_statistics()
TissueMultiscaleTest = _ss.TissueMultiscaleTest
ScaleRange = _ss.ScaleRange


@dataclass
class FingerprintResult:
    """VMR fingerprint for one realization."""
    vmr_curve: np.ndarray          # Raw VMR values at each scale
    null_vmr_mean: np.ndarray      # Mean VMR from CSR null
    vmr_ratio: np.ndarray          # vmr_curve / null_vmr_mean
    scales: np.ndarray             # Scale values (in a.u. or um)
    n_points: int


@dataclass
class RegimeEnsemble:
    """Collection of fingerprints for one regime at one parameter setting."""
    regime: str
    param_name: str
    param_value: float
    fingerprints: List[FingerprintResult] = field(default_factory=list)

    @property
    def vmr_ratio_matrix(self):
        """(n_realizations, n_scales) matrix of VMR ratios."""
        return np.array([f.vmr_ratio for f in self.fingerprints])

    @property
    def mean_vmr_ratio(self):
        return self.vmr_ratio_matrix.mean(axis=0)

    @property
    def scales(self):
        return self.fingerprints[0].scales if self.fingerprints else np.array([])


def compute_fingerprint(positions, roi_size=10000, grid_size=256, n_scales=12,
                         n_null=50, rng=None):
    """Compute VMR fingerprint for a single point pattern.

    Parameters
    ----------
    positions : array-like, shape (N, 2)
        Point positions in the ROI.
    roi_size : float
        Size of the square ROI.
    grid_size : int
        Grid resolution for binning.
    n_scales : int
        Number of log-spaced scales.
    n_null : int
        Number of CSR null realizations for the envelope.
    rng : int or Generator, optional
        Random seed/generator.

    Returns
    -------
    FingerprintResult
    """
    rng = np.random.default_rng(rng)
    positions = np.asarray(positions, dtype=float)
    n_points = len(positions)

    tmt = TissueMultiscaleTest(
        roi_size_um=np.array([roi_size, roi_size]),
        grid_size=grid_size,
    )
    scale_range = ScaleRange.for_tissue(
        min_um=roi_size / grid_size * 2,
        max_um=roi_size / 4,
        n_scales=n_scales,
        roi_size_um=roi_size,
        grid_size=grid_size,
    )

    # Real VMR curve
    real_curves = tmt.compute_curves(positions, scale_range)
    vmr_curve = np.array(real_curves['vmr'])

    # CSR null envelope
    null_vmrs = []
    for i in range(n_null):
        seed = rng.integers(0, 2**31)
        null_rng = np.random.default_rng(seed)
        null_pos = null_rng.uniform(0, roi_size, size=(n_points, 2))
        null_curves = tmt.compute_curves(null_pos, scale_range)
        null_vmrs.append(null_curves['vmr'])
    null_vmrs = np.array(null_vmrs)
    null_mean = null_vmrs.mean(axis=0)

    # VMR ratio
    vmr_ratio = vmr_curve / np.maximum(null_mean, 1e-10)

    scales = np.array(real_curves['scales_um'])

    return FingerprintResult(
        vmr_curve=vmr_curve,
        null_vmr_mean=null_mean,
        vmr_ratio=vmr_ratio,
        scales=scales,
        n_points=n_points,
    )


def compute_ensemble(regime, param_name, param_value, patterns,
                      roi_size=10000, grid_size=256, n_scales=12,
                      n_null=50, rng=None):
    """Compute fingerprints for all realizations of one parameter setting.

    Parameters
    ----------
    regime : str
        Regime name.
    param_name : str
        Swept parameter name.
    param_value : float
        Parameter value for this ensemble.
    patterns : list of (N, 2) arrays
        Point patterns from simulator.
    roi_size, grid_size, n_scales, n_null : see compute_fingerprint.
    rng : int or Generator, optional

    Returns
    -------
    RegimeEnsemble
    """
    rng = np.random.default_rng(rng)
    ensemble = RegimeEnsemble(
        regime=regime,
        param_name=param_name,
        param_value=param_value,
    )
    for pattern in patterns:
        seed = rng.integers(0, 2**31)
        fp = compute_fingerprint(
            pattern, roi_size=roi_size, grid_size=grid_size,
            n_scales=n_scales, n_null=n_null, rng=seed,
        )
        ensemble.fingerprints.append(fp)
    return ensemble
