#!/usr/bin/env python3
"""
Run all 7 simulators × parameter sweeps × 50 realizations.
Compute VMR fingerprints + CSR null. Extract 8 descriptors.
Save results/simulated_descriptors.json.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulators import SIMULATORS, SWEEP_PARAMS, simulate_regime
from src.fingerprinting import compute_fingerprint
from src.descriptors import extract_descriptors, descriptors_to_dicts

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_REALIZATIONS = 50
N_NULL = 50


def main():
    rng = np.random.default_rng(42)
    all_descriptors = []
    total_start = time.time()

    for regime, sweep in sorted(SWEEP_PARAMS.items()):
        param_name = sweep['param']
        param_values = sweep['values']
        print(f'\n{"=" * 60}')
        print(f'Regime: {regime} | param: {param_name} | '
              f'{len(param_values)} settings × {N_REALIZATIONS} realizations')
        print(f'{"=" * 60}')

        regime_start = time.time()
        for pval in param_values:
            print(f'  {param_name}={pval}', end='', flush=True)
            patterns = simulate_regime(regime, pval,
                                        n_realizations=N_REALIZATIONS,
                                        rng=rng.integers(0, 2**31))
            for i, pattern in enumerate(patterns):
                seed = rng.integers(0, 2**31)
                fp = compute_fingerprint(pattern, roi_size=10000,
                                          n_null=N_NULL, rng=seed)
                desc = extract_descriptors(
                    vmr_ratio=fp.vmr_ratio,
                    scales=fp.scales,
                    label=f'{regime}_{param_name}={pval}_r{i}',
                    source='simulation',
                    regime=regime,
                    positions=pattern,
                    parameters={param_name: float(pval), 'realization': i},
                )
                all_descriptors.append(desc)
                if (i + 1) % 10 == 0:
                    print('.', end='', flush=True)
            print(f' [{len(patterns)} done]')

        elapsed = time.time() - regime_start
        print(f'  {regime} completed in {elapsed:.1f}s')

    total_elapsed = time.time() - total_start
    print(f'\n{"=" * 60}')
    print(f'Total: {len(all_descriptors)} descriptors in {total_elapsed:.1f}s')
    print(f'{"=" * 60}')

    # Save
    output = {
        'n_descriptors': len(all_descriptors),
        'regimes': list(SWEEP_PARAMS.keys()),
        'n_realizations': N_REALIZATIONS,
        'n_null': N_NULL,
        'descriptors': descriptors_to_dicts(all_descriptors),
    }
    out_path = RESULTS_DIR / 'simulated_descriptors.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
