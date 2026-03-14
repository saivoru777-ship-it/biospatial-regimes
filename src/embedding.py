"""
Embedding: PCA/UMAP on descriptor space.

Critical design: train on simulations only, project empirical.
Includes KDE-based soft regime affinity and out-of-regime diagnostics.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde
import json

from .descriptors import FingerprintDescriptor, descriptors_to_matrix


class RegimeSpace:
    """PCA-based regime embedding space trained on simulations."""

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._fitted = False
        self._sim_labels = None
        self._sim_pca = None
        self._sim_scaled = None
        self._regime_kdes = None
        self._nn_threshold = None

    def fit(self, sim_descriptors):
        """Fit PCA on simulation descriptors.

        Parameters
        ----------
        sim_descriptors : list of FingerprintDescriptor
            ~5250 simulated descriptors.

        Returns
        -------
        self
        """
        X = descriptors_to_matrix(sim_descriptors)
        self._sim_labels = np.array([d.regime for d in sim_descriptors])

        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        self._sim_scaled = X_scaled

        # PCA
        self._sim_pca = self.pca.fit_transform(X_scaled)
        self._fitted = True

        # Fit KDEs per regime in 6D scaled space
        self._fit_kdes(X_scaled, self._sim_labels)

        # Compute NN threshold for out-of-regime detection
        self._compute_nn_threshold(X_scaled)

        return self

    def transform(self, descriptors):
        """Project descriptors into PCA space.

        Parameters
        ----------
        descriptors : list of FingerprintDescriptor

        Returns
        -------
        np.ndarray, shape (N, n_components)
        """
        X = descriptors_to_matrix(descriptors)
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def _fit_kdes(self, X_scaled, labels):
        """Fit Gaussian KDE per regime in standardized 6D space."""
        self._regime_kdes = {}
        unique_regimes = np.unique(labels)
        for regime in unique_regimes:
            mask = labels == regime
            X_regime = X_scaled[mask]
            if len(X_regime) < 10:
                continue
            # Use PCA-reduced space for KDE to avoid curse of dimensionality
            X_pca = self.pca.transform(X_regime)
            try:
                kde = gaussian_kde(X_pca.T, bw_method='scott')
                self._regime_kdes[regime] = kde
            except np.linalg.LinAlgError:
                continue

    def _compute_nn_threshold(self, X_scaled):
        """Compute 95th percentile of within-simulation NN distances."""
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        # Distance to nearest neighbor (not self)
        nn_dists = distances[:, 1]
        self._nn_threshold = float(np.percentile(nn_dists, 95))

    def regime_affinity(self, descriptors):
        """Compute KDE-based soft regime affinity for each descriptor.

        Parameters
        ----------
        descriptors : list of FingerprintDescriptor

        Returns
        -------
        affinities : np.ndarray, shape (N, n_regimes)
            Normalized probability under each regime's KDE.
        regime_names : list of str
            Regime names corresponding to columns.
        """
        X = descriptors_to_matrix(descriptors)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        regime_names = sorted(self._regime_kdes.keys())
        n = len(descriptors)
        affinities = np.zeros((n, len(regime_names)))

        for j, regime in enumerate(regime_names):
            kde = self._regime_kdes[regime]
            affinities[:, j] = kde(X_pca.T)

        # Normalize rows to sum to 1
        row_sums = affinities.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        affinities = affinities / row_sums

        return affinities, regime_names

    def out_of_regime(self, descriptors):
        """Flag empirical fingerprints that are out-of-regime.

        Returns boolean array: True if the descriptor is farther from any
        simulated point than the 95th percentile of within-simulation NN distances.
        """
        X = descriptors_to_matrix(descriptors)
        X_scaled = self.scaler.transform(X)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self._sim_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        return distances[:, 0] > self._nn_threshold

    def silhouette(self):
        """Silhouette score of simulation regime clouds in PCA space."""
        if self._sim_pca is None:
            return None
        return float(silhouette_score(self._sim_pca, self._sim_labels))

    def pairwise_mahalanobis(self):
        """Pairwise Mahalanobis distance between regime centroids."""
        regimes = np.unique(self._sim_labels)
        n_reg = len(regimes)
        distances = np.zeros((n_reg, n_reg))
        centroids = {}
        covs = {}
        for regime in regimes:
            mask = self._sim_labels == regime
            X = self._sim_pca[mask]
            centroids[regime] = X.mean(axis=0)
            covs[regime] = np.cov(X.T) + 1e-6 * np.eye(self.n_components)
        for i, r1 in enumerate(regimes):
            for j, r2 in enumerate(regimes):
                if i >= j:
                    continue
                pooled_cov = (covs[r1] + covs[r2]) / 2
                try:
                    d = mahalanobis(centroids[r1], centroids[r2],
                                    np.linalg.inv(pooled_cov))
                except np.linalg.LinAlgError:
                    d = np.nan
                distances[i, j] = d
                distances[j, i] = d
        return distances, list(regimes)

    def explained_variance(self):
        """Explained variance ratio for each PC."""
        return self.pca.explained_variance_ratio_

    def loadings(self):
        """PCA loadings matrix (n_components, 6)."""
        return self.pca.components_

    def save(self, path):
        """Save regime space to JSON-compatible format."""
        data = {
            'n_components': self.n_components,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'pca_components': self.pca.components_.tolist(),
            'pca_mean': self.pca.mean_.tolist(),
            'pca_explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'nn_threshold': self._nn_threshold,
            'sim_labels': self._sim_labels.tolist(),
            'sim_pca': self._sim_pca.tolist(),
            'sim_scaled': self._sim_scaled.tolist(),
            'silhouette_score': self.silhouette(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def fit_umap(sim_descriptors, n_components=2, n_neighbors=15, min_dist=0.1):
    """Fit UMAP on simulation descriptors.

    Returns
    -------
    umap_model : fitted UMAP
    sim_embedding : np.ndarray, shape (N, n_components)
    scaler : fitted StandardScaler
    """
    import umap

    X = descriptors_to_matrix(sim_descriptors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    embedding = reducer.fit_transform(X_scaled)
    return reducer, embedding, scaler


def project_umap(umap_model, scaler, descriptors):
    """Project empirical descriptors into UMAP space."""
    X = descriptors_to_matrix(descriptors)
    X_scaled = scaler.transform(X)
    return umap_model.transform(X_scaled)
