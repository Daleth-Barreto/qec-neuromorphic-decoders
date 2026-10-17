"""Baselines BP+OSD."""
import numpy as np
from bposd import bposd_decoder


def _accumulate_error(num_qubits, rounds, p, rng):
    error = np.zeros(num_qubits, dtype=np.uint8)
    for _ in range(rounds):
        flips = (rng.random(num_qubits) < p).astype(np.uint8)
        error = error ^ flips
    return error


def run_bposd_baseline(Hz, Lz, rounds, p, n_trials, seed=None, osd_order=7):
    rng = np.random.default_rng(seed)
    num_checks, num_qubits = Hz.shape
    p_eff = float(0.5 * (1.0 - (1.0 - 2.0 * p) ** rounds))
    bpd = bposd_decoder(
        Hz,
        error_rate=p_eff,
        channel_probs=[None],
        max_iter=num_qubits,
        bp_method="ms",
        ms_scaling_factor=0,
        osd_method="osd_cs",
        osd_order=osd_order,
    )
    n_fail = 0
    for _ in range(n_trials):
        error = _accumulate_error(num_qubits, rounds, p, rng)
        syndrome = (Hz @ error) % 2
        if not syndrome.any():
            correction = np.zeros(num_qubits, dtype=np.uint8)
        else:
            correction = bpd.decode(syndrome)
        residual = (correction + error) % 2
        logical_error = int(((Lz @ residual) % 2).any())
        n_fail += logical_error
    return n_fail / n_trials