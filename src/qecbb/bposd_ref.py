"""Decodificador de referencia BP-OSD sobre el DEM fiel del BB72 (CPU).

Reutiliza la logica de ``experiments/run_sanity_faithful.py`` (ldpc.BpOsdDecoder
con las probabilidades por mecanismo del DEM, osd_cs). Se usa para obtener la
curva de referencia contra la que se compara el SNN en la MISMA tarea
(mismos shots, mismos observables).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse

from .stim_bb_faithful import load_dem_matrices


def dem_matrices(circuit, flatten_loops: bool = True):
    """Devuelve ``(H, L, probs, dem)`` del circuito (H/L sparse uint8)."""
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=flatten_loops)
    H, L, probs = load_dem_matrices(dem)
    return H, L, probs, dem


def make_decoder(H, probs, osd_order: int = 3, max_iter: int = 12):
    from ldpc import BpOsdDecoder

    return BpOsdDecoder(
        scipy.sparse.csr_matrix(H),
        channel_probs=probs,
        max_iter=max_iter,
        bp_method="product_sum",
        schedule="parallel",
        osd_method="OSD_CS",
        osd_order=osd_order,
    )


def decoder_for(circuit, osd_order: int = 3, max_iter: int = 12):
    """Construye ``(decoder, Ld)`` para reutilizar en varios lotes de shots."""
    H, L, probs, _ = dem_matrices(circuit)
    decoder = make_decoder(H, probs, osd_order=osd_order, max_iter=max_iter)
    return decoder, L.toarray().astype(np.uint8)


def decode_observables(circuit, dets: np.ndarray, osd_order: int = 3,
                       max_iter: int = 12, decoder_and_L=None):
    """Predice los observables para ``dets`` (N, n_det) -> (N, k) uint8."""
    if decoder_and_L is None:
        decoder, Ld = decoder_for(circuit, osd_order=osd_order, max_iter=max_iter)
    else:
        decoder, Ld = decoder_and_L
    dets = np.asarray(dets, dtype=np.uint8)
    pred = np.zeros((dets.shape[0], Ld.shape[0]), dtype=np.uint8)
    for i in range(dets.shape[0]):
        corr = decoder.decode(dets[i])
        pred[i] = (Ld @ corr) % 2
    return pred


def block_ler(pred: np.ndarray, obs: np.ndarray) -> float:
    """LER de bloque = fraccion de shots con AL MENOS un observable mal."""
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    return float(np.mean(np.any(pred != obs, axis=1)))
