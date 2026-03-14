"""
Extract fingerprint descriptors from VMR ratio curves.

6 primary descriptors (define embedding space):
  peak_height, threshold_scale, width, small_scale_slope,
  large_scale_slope, integrated_excess

2 supplementary descriptors (overlays, not in embedding):
  anisotropy, multimodality
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from scipy.signal import find_peaks

# Threshold for threshold_scale: first scale where |log2(VMR ratio)| exceeds this
THRESHOLD_LOG2 = 0.5


@dataclass
class FingerprintDescriptor:
    """Descriptor vector extracted from a VMR ratio curve."""
    label: str
    source: str            # 'simulation', 'tissue', 'smlm', 'dendrite'
    regime: str
    # --- 6 primary descriptors ---
    peak_height: float
    threshold_scale: float  # normalized [0, 1]; first scale where |log2(VMR ratio)| > threshold
    width: float            # FWHM in log-scale decades
    small_scale_slope: float
    large_scale_slope: float
    integrated_excess: float
    # --- 2 supplementary descriptors ---
    anisotropy: Optional[float]
    multimodality: float
    # --- metadata ---
    parameters: Dict[str, Any]

    def primary_vector(self):
        """Return 6D primary descriptor vector for embedding."""
        return np.array([
            self.peak_height,
            self.threshold_scale,
            self.width,
            self.small_scale_slope,
            self.large_scale_slope,
            self.integrated_excess,
        ])

    def to_dict(self):
        d = asdict(self)
        # Convert numpy types
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
        return d


def extract_descriptors(vmr_ratio, scales, label='', source='simulation',
                         regime='', positions=None, parameters=None):
    """Extract all descriptors from a VMR ratio curve.

    Parameters
    ----------
    vmr_ratio : array-like
        VMR ratio values (vmr / null_mean). Can be any length >= 3.
    scales : array-like
        Scale values corresponding to vmr_ratio.
    label : str
        Identifier for this fingerprint.
    source : str
        Data source ('simulation', 'tissue', 'smlm', 'dendrite').
    regime : str
        Regime label (for simulations) or biological category.
    positions : array-like, shape (N, 2), optional
        Raw point positions for anisotropy computation.
    parameters : dict, optional
        Parameter metadata.

    Returns
    -------
    FingerprintDescriptor
    """
    vmr_ratio = np.asarray(vmr_ratio, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if parameters is None:
        parameters = {}

    n = len(vmr_ratio)
    log_ratio = np.log2(np.maximum(vmr_ratio, 1e-10))
    log_scales = np.log10(np.maximum(scales, 1e-10))

    # Normalized scale positions [0, 1]
    if log_scales[-1] > log_scales[0]:
        norm_scales = (log_scales - log_scales[0]) / (log_scales[-1] - log_scales[0])
    else:
        norm_scales = np.linspace(0, 1, n)

    # Peak height: max of log2(vmr_ratio)
    peak_height = float(np.max(log_ratio))

    # Threshold scale: first normalized scale where |log2(VMR ratio)| > threshold
    threshold_scale = _compute_threshold_scale(log_ratio, norm_scales,
                                                threshold=THRESHOLD_LOG2)

    # Width: FWHM in log-scale decades
    width = _compute_fwhm(log_ratio, log_scales)

    # Small-scale slope: slope of log2(vmr_ratio) over first 1/3 of scales
    n_third = max(2, n // 3)
    small_scale_slope = _fit_slope(norm_scales[:n_third], log_ratio[:n_third])

    # Large-scale slope: slope over last 1/3 of scales
    large_scale_slope = _fit_slope(norm_scales[-n_third:], log_ratio[-n_third:])

    # Integrated excess: trapz(log2(vmr_ratio)) over normalized log-scale
    integrated_excess = float(np.trapezoid(log_ratio, norm_scales))

    # Anisotropy: eigenvalue ratio of 2D covariance
    anisotropy = _compute_anisotropy(positions) if positions is not None else None

    # Multimodality: 2nd peak prominence / 1st peak prominence
    multimodality = _compute_multimodality(log_ratio)

    return FingerprintDescriptor(
        label=label,
        source=source,
        regime=regime,
        peak_height=peak_height,
        threshold_scale=threshold_scale,
        width=width,
        small_scale_slope=small_scale_slope,
        large_scale_slope=large_scale_slope,
        integrated_excess=integrated_excess,
        anisotropy=anisotropy,
        multimodality=multimodality,
        parameters=parameters,
    )


def _compute_threshold_scale(log_ratio, norm_scales, threshold=0.5):
    """First normalized scale where |log2(VMR ratio)| exceeds threshold.

    Returns 1.0 (max scale) if the threshold is never crossed — meaning
    the signal never rises above noise. This places null/weak-signal
    fingerprints at the top of this dimension.
    """
    crossings = np.where(np.abs(log_ratio) > threshold)[0]
    if len(crossings) == 0:
        return 1.0  # never crosses → signal too weak
    return float(norm_scales[crossings[0]])


def _compute_fwhm(log_ratio, log_scales):
    """Compute FWHM of the absolute log_ratio curve in log-scale decades."""
    abs_curve = np.abs(log_ratio)
    peak_val = np.max(abs_curve)
    if peak_val < 1e-10:
        return 0.0
    half_max = peak_val / 2
    above = abs_curve >= half_max
    if not np.any(above):
        return 0.0
    indices = np.where(above)[0]
    fwhm = float(log_scales[indices[-1]] - log_scales[indices[0]])
    return max(fwhm, 0.0)


def _fit_slope(x, y):
    """Fit linear slope via least squares."""
    if len(x) < 2:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    coeffs = np.polyfit(x[mask], y[mask], 1)
    return float(coeffs[0])


def _compute_anisotropy(positions):
    """Eigenvalue ratio of 2D covariance matrix."""
    if positions is None:
        return None
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 3:
        return None
    cov = np.cov(positions.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.abs(eigvals))
    if eigvals[-1] < 1e-10:
        return 1.0
    return float(eigvals[0] / eigvals[-1])


def _compute_multimodality(log_ratio):
    """Ratio of 2nd peak prominence to 1st peak prominence."""
    abs_curve = np.abs(log_ratio)
    peaks, properties = find_peaks(abs_curve, prominence=0.01)
    if len(peaks) < 2:
        return 0.0
    prominences = properties['prominences']
    sorted_prom = np.sort(prominences)[::-1]
    if sorted_prom[0] < 1e-10:
        return 0.0
    return float(sorted_prom[1] / sorted_prom[0])


def descriptors_to_matrix(descriptors):
    """Extract (N, 6) primary descriptor matrix from list of descriptors."""
    return np.array([d.primary_vector() for d in descriptors])


def descriptors_to_dicts(descriptors):
    """Serialize list of descriptors to list of dicts."""
    return [d.to_dict() for d in descriptors]


def dicts_to_descriptors(dicts):
    """Deserialize list of dicts to list of FingerprintDescriptor."""
    result = []
    for d in dicts:
        result.append(FingerprintDescriptor(**d))
    return result
