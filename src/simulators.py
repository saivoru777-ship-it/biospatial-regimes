"""
Point-pattern generators for 7 biospatial regimes with parameter sweeps.

All simulators operate on a ROI_SIZE x ROI_SIZE arena (default 10000 a.u.)
and produce n_points (default 2000) positions as (N, 2) arrays.
"""

import numpy as np
from scipy.spatial import KDTree

ROI_SIZE = 10000
N_POINTS = 2000

# --- Parameter sweep ranges ---
SWEEP_PARAMS = {
    'random': {
        'param': 'n_points',
        'values': [500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000],
    },
    'clustered': {
        'param': 'cluster_radius',
        'values': [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1500, 2000],
    },
    'exclusion': {
        'param': 'min_distance',
        'values': [10, 20, 30, 50, 70, 100, 130, 160, 200, 250, 300, 400],
    },
    'layered': {
        'param': 'layer_width_frac',
        'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    'compartmental': {
        'param': 'n_compartments',
        'values': [2, 3, 4, 6, 8, 10, 12, 15, 20, 25],
    },
    'branch_constrained': {
        'param': 'branch_width',
        'values': [5, 10, 20, 30, 50, 80, 120, 180, 250, 350, 500],
    },
    'hierarchical': {
        'param': 'scale_ratio',
        'values': [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
    },
}


def simulate_random(n_points=N_POINTS, roi_size=ROI_SIZE, rng=None):
    """Uniform random (CSR) point pattern."""
    rng = np.random.default_rng(rng)
    return rng.uniform(0, roi_size, size=(n_points, 2))


def simulate_clustered(cluster_radius=200, n_points=N_POINTS, roi_size=ROI_SIZE,
                        n_parents=None, rng=None):
    """Thomas process: Poisson parents + Gaussian offspring."""
    rng = np.random.default_rng(rng)
    if n_parents is None:
        n_parents = max(5, n_points // 20)
    parents = rng.uniform(0, roi_size, size=(n_parents, 2))
    points_per_parent = max(1, n_points // n_parents)
    points = []
    for p in parents:
        offsets = rng.normal(0, cluster_radius, size=(points_per_parent, 2))
        points.append(p + offsets)
    points = np.vstack(points)[:n_points]
    # Wrap to ROI
    points = points % roi_size
    return points


def simulate_exclusion(min_distance=100, n_points=N_POINTS, roi_size=ROI_SIZE,
                        rng=None, max_attempts=500000):
    """Simple Sequential Inhibition (SSI) via KD-tree rejection sampling."""
    rng = np.random.default_rng(rng)
    points = []
    tree = None
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        candidate = rng.uniform(0, roi_size, size=(1, 2))
        attempts += 1
        if tree is None:
            points.append(candidate[0])
            tree = KDTree(np.array(points))
        else:
            dist, _ = tree.query(candidate)
            if dist[0] > min_distance:
                points.append(candidate[0])
                tree = KDTree(np.array(points))
    points = np.array(points)
    if len(points) < n_points:
        # Fill remaining with uniform if packing limit reached
        remaining = n_points - len(points)
        fill = rng.uniform(0, roi_size, size=(remaining, 2))
        points = np.vstack([points, fill])
    return points[:n_points]


def simulate_layered(layer_width_frac=0.3, n_points=N_POINTS, roi_size=ROI_SIZE,
                      n_layers=5, rng=None):
    """Gaussian band layers along y-axis."""
    rng = np.random.default_rng(rng)
    layer_width = layer_width_frac * (roi_size / n_layers)
    # Evenly spaced layer centers
    centers_y = np.linspace(roi_size / (2 * n_layers),
                            roi_size - roi_size / (2 * n_layers), n_layers)
    points_per_layer = n_points // n_layers
    remainder = n_points - points_per_layer * n_layers
    points = []
    for i, cy in enumerate(centers_y):
        n = points_per_layer + (1 if i < remainder else 0)
        x = rng.uniform(0, roi_size, size=n)
        y = rng.normal(cy, layer_width, size=n)
        points.append(np.column_stack([x, y]))
    points = np.vstack(points)
    # Wrap y to ROI
    points[:, 1] = points[:, 1] % roi_size
    return points[:n_points]


def simulate_compartmental(n_compartments=6, n_points=N_POINTS, roi_size=ROI_SIZE,
                            compartment_radius=None, rng=None):
    """SSI-placed compartment centers + Thomas sub-clusters within each."""
    rng = np.random.default_rng(rng)
    if compartment_radius is None:
        compartment_radius = roi_size / (2 * np.sqrt(n_compartments))
    # Place compartment centers with exclusion
    min_dist = 2 * compartment_radius * 0.8
    centers = []
    for _ in range(100000):
        c = rng.uniform(compartment_radius, roi_size - compartment_radius, size=(1, 2))
        if len(centers) == 0:
            centers.append(c[0])
        else:
            tree = KDTree(np.array(centers))
            d, _ = tree.query(c)
            if d[0] > min_dist:
                centers.append(c[0])
        if len(centers) >= n_compartments:
            break
    if len(centers) < n_compartments:
        # Fill with random if packing fails
        for _ in range(n_compartments - len(centers)):
            centers.append(rng.uniform(0, roi_size, size=2))
    centers = np.array(centers)
    # Place points within compartments (Thomas sub-clusters)
    points_per_comp = n_points // n_compartments
    remainder = n_points - points_per_comp * n_compartments
    points = []
    for i, c in enumerate(centers):
        n = points_per_comp + (1 if i < remainder else 0)
        offsets = rng.normal(0, compartment_radius * 0.4, size=(n, 2))
        points.append(c + offsets)
    points = np.vstack(points)[:n_points]
    points = np.clip(points, 0, roi_size)
    return points


def _lsystem_tree(depth=5, length=1500, angle=25, length_ratio=0.7,
                   rng=None):
    """Generate L-system tree skeleton as list of (start, end) segments."""
    rng = np.random.default_rng(rng)
    segments = []
    stack = []

    def _grow(x, y, heading, d, remaining_depth):
        if remaining_depth == 0:
            return
        seg_len = d * (0.8 + 0.4 * rng.random())
        x2 = x + seg_len * np.cos(np.radians(heading))
        y2 = y + seg_len * np.sin(np.radians(heading))
        segments.append(((x, y), (x2, y2)))
        angle_jitter = angle * (0.7 + 0.6 * rng.random())
        _grow(x2, y2, heading + angle_jitter, d * length_ratio, remaining_depth - 1)
        _grow(x2, y2, heading - angle_jitter, d * length_ratio, remaining_depth - 1)

    _grow(5000, 500, 90, length, depth)
    return segments


def simulate_branch_constrained(branch_width=50, n_points=N_POINTS, roi_size=ROI_SIZE,
                                 rng=None):
    """L-system tree skeleton + Gaussian jitter perpendicular to branches."""
    rng = np.random.default_rng(rng)
    segments = _lsystem_tree(depth=6, length=1800, rng=rng)
    if not segments:
        return rng.uniform(0, roi_size, size=(n_points, 2))
    # Compute segment lengths for weighted sampling
    seg_lengths = []
    for (x1, y1), (x2, y2) in segments:
        seg_lengths.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    seg_lengths = np.array(seg_lengths)
    probs = seg_lengths / seg_lengths.sum()
    # Sample points along segments with perpendicular jitter
    seg_indices = rng.choice(len(segments), size=n_points, p=probs)
    points = []
    for idx in seg_indices:
        (x1, y1), (x2, y2) = segments[idx]
        t = rng.uniform(0, 1)
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        # Perpendicular direction
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx, ny = -dy / length, dx / length
        else:
            nx, ny = 0, 1
        jitter = rng.normal(0, branch_width)
        px += jitter * nx
        py += jitter * ny
        points.append([px, py])
    points = np.array(points)
    points = np.clip(points, 0, roi_size)
    return points[:n_points]


def simulate_hierarchical(scale_ratio=5, n_points=N_POINTS, roi_size=ROI_SIZE,
                           rng=None):
    """3-level nested Thomas process with controlled scale separation."""
    rng = np.random.default_rng(rng)
    # Level 0: macro-clusters
    base_radius = roi_size / 8
    n_l0 = max(3, int(np.sqrt(n_points / 10)))
    l0_centers = rng.uniform(0, roi_size, size=(n_l0, 2))

    # Level 1: meso-clusters within each macro
    l1_radius = base_radius / scale_ratio
    n_l1_per_l0 = max(2, int(np.sqrt(n_points / n_l0 / 5)))
    l1_centers = []
    for c in l0_centers:
        offsets = rng.normal(0, base_radius, size=(n_l1_per_l0, 2))
        l1_centers.append(c + offsets)
    l1_centers = np.vstack(l1_centers)

    # Level 2: micro-clusters (actual points)
    l2_radius = l1_radius / scale_ratio
    n_l1 = len(l1_centers)
    pts_per_l1 = n_points // n_l1
    remainder = n_points - pts_per_l1 * n_l1
    points = []
    for idx, c in enumerate(l1_centers):
        n = pts_per_l1 + (1 if idx < remainder else 0)
        offsets = rng.normal(0, l2_radius, size=(n, 2))
        points.append(c + offsets)
    points = np.vstack(points)[:n_points]
    points = points % roi_size
    return points


# --- Registry ---
SIMULATORS = {
    'random': simulate_random,
    'clustered': simulate_clustered,
    'exclusion': simulate_exclusion,
    'layered': simulate_layered,
    'compartmental': simulate_compartmental,
    'branch_constrained': simulate_branch_constrained,
    'hierarchical': simulate_hierarchical,
}


def simulate_regime(regime, param_value, n_realizations=50, rng=None):
    """Run one regime at one parameter value for n_realizations.

    Returns list of (N, 2) arrays.
    """
    rng = np.random.default_rng(rng)
    param_name = SWEEP_PARAMS[regime]['param']
    func = SIMULATORS[regime]
    patterns = []
    for i in range(n_realizations):
        seed = rng.integers(0, 2**31)
        kwargs = {param_name: param_value, 'rng': seed}
        if regime != 'random':
            kwargs['n_points'] = N_POINTS
        kwargs['roi_size'] = ROI_SIZE
        patterns.append(func(**kwargs))
    return patterns
