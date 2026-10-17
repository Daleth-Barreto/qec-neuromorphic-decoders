# coding: utf-8
import numpy as np
from .stim_bb import build_bb_memory_z_circuit

def generate_stim_data(n_samples, code_name, rounds, p_dep, p_meas=0.001, seed=None):
    rng = np.random.default_rng(seed)
    c = build_bb_memory_z_circuit(code_name, rounds, p_dep, p_meas)
    sampler = c.compile_detector_sampler(seed=rng.integers(0, 2**63))
    out = sampler.sample(n_samples, separate_observables=True, bit_packed=False)
    dets, obs = out
    m_x = c.num_detectors // rounds if rounds > 0 else c.num_detectors
    feats = dets.reshape(n_samples, rounds, m_x).transpose(0,2,1).astype(np.float32)
    labels = obs.astype(np.float32)
    return feats, labels