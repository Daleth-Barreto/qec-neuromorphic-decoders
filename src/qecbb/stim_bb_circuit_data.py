"""Datos desde el circuito BB fiel (circuit-level) para las dos variantes.

Variantes:
  - ``uncorrelated``: solo estabilizadores Z (36 checks). Corresponde a
    ``use_both=False`` (252 detectores) y es comparable con jblue1 ``mbFalse``
    y con BP-OSD de una sola base.
  - ``correlated``: estabilizadores Z y X (72 checks). Corresponde a
    ``use_both=True`` (432 detectores), la tarea por defecto de NTU.

Features por muestra: ``(n_checks, T)`` con ``T = rounds + 1`` slices temporales.
Para ``correlated`` los slices X en los bordes (t=0 y t=T-1) son cero porque en
esas rondas solo se declaran detectores Z.

Layout de detectores (derivado de ``stim_bb_faithful.build_circuit``):

  uncorrelated: [Z@t0 (36)] [Z@t1 (36)] ... [Z@t6 (36)]                 = 252
  correlated:   [Z@t0 (36)] [Z@t1 X@t1] ... [Z@t5 X@t5] [Z@t6 (36)]    = 432
"""
from __future__ import annotations

import numpy as np
import stim

from .graphs import build_check_adjacency
from .stim_bb_faithful import BB72, build_circuit, create_bivariate_bicycle_code

VARIANTS = ("uncorrelated", "correlated")
ROUNDS = BB72["rounds"]          # 6 rondas ruidosas
T_SLICES = ROUNDS + 1            # 7 slices temporales (t = 0..6)
LM = BB72["l"] * BB72["m"]       # 36 estabilizadores por base


def _use_both(variant: str) -> bool:
    if variant not in VARIANTS:
        raise ValueError(f"variant debe ser uno de {VARIANTS}, no {variant!r}")
    return variant == "correlated"


def n_checks(variant: str) -> int:
    """Número de nodos del grafo de checks: 36 (uncorrelated) o 72 (correlated)."""
    return LM if variant == "uncorrelated" else 2 * LM


def build_variant_circuit(p: float, variant: str, z_basis: bool = True) -> stim.Circuit:
    """Circuito BB72 fiel para la variante pedida."""
    code, A_list, B_list = create_bivariate_bicycle_code(
        BB72["l"], BB72["m"], BB72["A_x_pows"], BB72["A_y_pows"],
        BB72["B_x_pows"], BB72["B_y_pows"],
    )
    return build_circuit(
        code, A_list, B_list, p=p, rounds=ROUNDS,
        z_basis=z_basis, use_both=_use_both(variant),
    )


def detectors_to_features(dets: np.ndarray, variant: str) -> np.ndarray:
    """Reordena detectores crudos ``(N, n_det)`` a features ``(N, n_checks, T)``."""
    dets = np.asarray(dets)
    N = dets.shape[0]
    if variant == "uncorrelated":
        if dets.shape[1] != LM * T_SLICES:
            raise ValueError(f"esperaba {LM * T_SLICES} detectores, recibí {dets.shape[1]}")
        return dets.reshape(N, T_SLICES, LM).transpose(0, 2, 1).astype(np.float32)

    # correlated
    expected = LM + (ROUNDS - 1) * (2 * LM) + LM
    if dets.shape[1] != expected:
        raise ValueError(f"esperaba {expected} detectores, recibí {dets.shape[1]}")
    feats = np.zeros((N, 2 * LM, T_SLICES), dtype=np.float32)
    feats[:, :LM, 0] = dets[:, :LM]                     # Z @ t=0
    for r in range(1, ROUNDS):                          # Z y X @ t=1..5
        base = LM + (r - 1) * (2 * LM)
        feats[:, :LM, r] = dets[:, base:base + LM]
        feats[:, LM:, r] = dets[:, base + LM:base + 2 * LM]
    feats[:, :LM, ROUNDS] = dets[:, -LM:]               # Z @ t=6
    return feats


def generate_circuit_data(n_samples: int, p: float, variant: str,
                          seed: int | None = None, balance: bool = False,
                          max_attempts_factor: int = 60, return_dets: bool = False):
    """Genera datos desde el circuito fiel.

    Returns
    -------
    feats : (N, n_checks, T) float32
    labels : (N, 12) float32  (flip crudo del observable; el decoder lo predice)
    any_logical_error : (N,) int64  (flip crudo a nivel de bloque, pre-decoding)
    dets : (N, n_det) uint8  -- solo si ``return_dets=True``
    """
    circuit = build_variant_circuit(p, variant)
    sampler = circuit.compile_detector_sampler(seed=seed)
    n_chk = n_checks(variant)

    if not balance:
        dets, obs = sampler.sample(n_samples, separate_observables=True, bit_packed=False)
        dets = np.asarray(dets, dtype=np.uint8)
        feats = detectors_to_features(dets, variant)
        labels = obs.astype(np.float32)
        any_err = (labels.sum(axis=1) > 0).astype(np.int64)
        if return_dets:
            return feats, labels, any_err, dets
        return feats, labels, any_err

    # Balanceado: mitad con error lógico, mitad sin error.
    feats_list, labels_list = [], []
    got_err = got_ok = 0
    want_err = want_ok = n_samples // 2
    attempts = 0
    max_attempts = n_samples * max_attempts_factor
    while (got_err < want_err or got_ok < want_ok) and attempts < max_attempts:
        attempts += 1
        dets, obs = sampler.sample(1, separate_observables=True, bit_packed=False)
        has_err = bool(obs[0].sum() > 0)
        if has_err and got_err < want_err:
            got_err += 1
        elif (not has_err) and got_ok < want_ok:
            got_ok += 1
        else:
            continue
        feats_list.append(dets[0])
        labels_list.append(obs[0])
        if attempts % max(n_samples, 5000) == 0:
            print(f"[balance] err={got_err}/{want_err} ok={got_ok}/{want_ok} "
                  f"attempts={attempts}", flush=True)

    if not feats_list:
        raise RuntimeError("no se generaron muestras balanceadas")
    dets = np.stack(feats_list, axis=0)
    obs = np.stack(labels_list, axis=0).astype(np.float32)
    feats = detectors_to_features(dets, variant)
    labels = obs
    any_err = (labels.sum(axis=1) > 0).astype(np.int64)
    if len(labels) < n_samples:
        print(f"[aviso] solo {len(labels)}/{n_samples} muestras balanceadas")
    if return_dets:
        return feats, labels, any_err, dets
    return feats, labels, any_err


def features_to_detectors(feats: np.ndarray, variant: str) -> np.ndarray:
    """Inverso de ``detectors_to_features`` (para verificar el mapeo)."""
    feats = np.asarray(feats)
    N = feats.shape[0]
    if variant == "uncorrelated":
        return feats.transpose(0, 2, 1).reshape(N, LM * T_SLICES).astype(np.uint8)
    if variant != "correlated":
        raise ValueError(f"variant inválido: {variant!r}")
    dets = np.zeros((N, LM + (ROUNDS - 1) * (2 * LM) + LM), dtype=np.uint8)
    dets[:, :LM] = feats[:, :LM, 0]
    for r in range(1, ROUNDS):
        base = LM + (r - 1) * (2 * LM)
        dets[:, base:base + LM] = feats[:, :LM, r]
        dets[:, base + LM:base + 2 * LM] = feats[:, LM:, r]
    dets[:, -LM:] = feats[:, :LM, ROUNDS]
    return dets


def build_variant_graph(variant: str):
    """Grafo de checks (edge_index PyG) para la variante.

    - uncorrelated: clique sobre los 36 checks Z.
    - correlated:   clique sobre los 72 checks (Z en 0..35, X en 36..71),
      incluyendo aristas cruzadas Z-X por qubits compartidos.
    """
    code, _, _ = create_bivariate_bicycle_code(
        BB72["l"], BB72["m"], BB72["A_x_pows"], BB72["A_y_pows"],
        BB72["B_x_pows"], BB72["B_y_pows"],
    )
    hz = np.asarray(code.hz, dtype=np.uint8)
    if variant == "uncorrelated":
        H = hz
    elif variant == "correlated":
        H = np.vstack([hz, np.asarray(code.hx, dtype=np.uint8)])
    else:
        raise ValueError(f"variant inválido: {variant!r}")
    edge_index = build_check_adjacency(H)
    return edge_index, int(H.shape[0])
