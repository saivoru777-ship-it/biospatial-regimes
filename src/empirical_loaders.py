"""
Load VMR ratio curves from all 3 empirical case studies and extract descriptors.

- Tissue (spatial transcriptomics): 12 scales, ~45 fingerprints
- SMLM (protein nanoclusters): 3-10 scales, ~4 fingerprints
- Dendrites (synaptic clustering): 12 scales, ~24 fingerprints
"""

import json
import numpy as np
from os.path import expanduser, exists
from typing import List

from .descriptors import FingerprintDescriptor, extract_descriptors


# --- Tissue (Spatial Transcriptomics) ---

TISSUE_RESULTS_DIR = expanduser(
    '~/research/spatial-transcriptomics-fingerprints/results'
)
TISSUE_REGIONS = ['isocortex', 'hippocampus', 'thalamus']


def load_tissue_fingerprints():
    """Load tissue fingerprints from all 3 brain regions.

    Returns list of FingerprintDescriptor.
    """
    descriptors = []
    for region in TISSUE_REGIONS:
        path = f'{TISSUE_RESULTS_DIR}/{region}_fingerprints.json'
        if not exists(path):
            print(f'Warning: tissue file not found: {path}')
            continue
        with open(path) as f:
            data = json.load(f)
        for cell_type, entry in data.items():
            vmr_curve = np.array(entry['vmr_curve'])
            mock_mean = np.array(entry['mock_vmr_mean'])
            scales = np.array(entry['scales_um'])
            # VMR ratio
            vmr_ratio = vmr_curve / np.maximum(mock_mean, 1e-10)
            desc = extract_descriptors(
                vmr_ratio=vmr_ratio,
                scales=scales,
                label=f'{region}_{cell_type}',
                source='tissue',
                regime=region,
                parameters={
                    'region': region,
                    'cell_type': cell_type,
                    'n_cells': entry.get('n_cells', 0),
                },
            )
            descriptors.append(desc)
    return descriptors


# --- SMLM ---

SMLM_PATH = expanduser(
    '~/research/smlm-clustering/results/final/G_multiscale_signatures.json'
)


def load_smlm_fingerprints():
    """Load SMLM fingerprints from multiscale signatures.

    Returns list of FingerprintDescriptor.
    """
    if not exists(SMLM_PATH):
        print(f'Warning: SMLM file not found: {SMLM_PATH}')
        return []
    with open(SMLM_PATH) as f:
        data = json.load(f)
    results = data.get('results', data)
    descriptors = []
    for sample_name, sample_data in results.items():
        variance = sample_data.get('variance', {})
        values = np.array(variance.get('values', []))
        scales = np.array(variance.get('scales_nm', []))
        envelope = variance.get('envelope', {})
        median_env = np.array(envelope.get('median', []))
        if len(values) < 3 or len(median_env) < 3:
            continue
        # VMR ratio = values / envelope median
        vmr_ratio = values / np.maximum(median_env, 1e-10)
        desc = extract_descriptors(
            vmr_ratio=vmr_ratio,
            scales=scales,
            label=f'smlm_{sample_name}',
            source='smlm',
            regime='smlm',
            parameters={'sample': sample_name},
        )
        descriptors.append(desc)
    return descriptors


# --- Dendrites ---

DENDRITE_PATH = expanduser(
    '~/research/neurostat-input-clustering/results/input_clustering_results.json'
)


def load_dendrite_fingerprints():
    """Load dendrite fingerprints from input clustering results.

    Returns list of FingerprintDescriptor.
    """
    if not exists(DENDRITE_PATH):
        print(f'Warning: dendrite file not found: {DENDRITE_PATH}')
        return []
    with open(DENDRITE_PATH) as f:
        data = json.load(f)
    descriptors = []
    for neuron in data:
        label = neuron.get('label', 'unknown')
        type_results = neuron.get('type_results', {})
        for syn_type, type_data in type_results.items():
            curves = type_data.get('curves', {})
            variance_values = np.array(curves.get('variance_values', []))
            scales = np.array(curves.get('scales', []))
            mock_envelope = type_data.get('mock_envelope', {})
            variance_mean = np.array(mock_envelope.get('variance_mean', []))
            if len(variance_values) < 3 or len(variance_mean) < 3:
                continue
            # VMR ratio
            vmr_ratio = variance_values / np.maximum(variance_mean, 1e-10)
            desc = extract_descriptors(
                vmr_ratio=vmr_ratio,
                scales=scales,
                label=f'dendrite_{label}_{syn_type}',
                source='dendrite',
                regime='dendrite',
                parameters={
                    'neuron': label,
                    'synapse_type': syn_type,
                    'n_synapses': type_data.get('n_synapses', 0),
                },
            )
            descriptors.append(desc)
    return descriptors


def load_all_empirical():
    """Load all empirical fingerprints from all 3 case studies.

    Returns list of FingerprintDescriptor.
    """
    tissue = load_tissue_fingerprints()
    smlm = load_smlm_fingerprints()
    dendrite = load_dendrite_fingerprints()
    print(f'Loaded empirical: {len(tissue)} tissue, {len(smlm)} SMLM, '
          f'{len(dendrite)} dendrite = {len(tissue) + len(smlm) + len(dendrite)} total')
    return tissue + smlm + dendrite
