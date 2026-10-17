# coding: utf-8
"""Circuito de memoria BB a nivel de circuito, FIEL a la referencia publicada.

Este modulo sustituye la construccion ingenua de ``stim_bb.py`` (que solo medía
checks-Z con un schedule fila-a-fila y colocaba el ruido de medida DESPUES del
``M``; ver ``experiments/run_sanity_faithful.py`` para la cuantificacion del bug).

Procedencia / atribucion
------------------------
El constructor de circuito se porta tal cual desde:
    GrahamYan/ntu-decoder, ``codes/BB/baseline.py`` (licencia MIT)
que a su vez reproduce el circuito de memoria de Bravyi et al., *High-threshold
and low-overhead fault-tolerant quantum memory*, Nature 627, 778 (2024) y el
circuito usado por jblue1/ml_bb_decoding (train.py, circuit_gen.py).

No se modifica la logica fisica del circuito; solo se expone como funciones
reutilizables y se anade un runner reproducible.

Modelo de ruido (circuit-level, prob. p por operacion):
    - init |0>  -> X_ERROR(p)          (no modelado en el reset de ancillas de
      la ronda inicial mas alla de lo que hace la referencia)
    - gates 1q / idle -> DEPOLARIZE1(p)
    - gates 2q -> DEPOLARIZE2(p)
    - medida -> X_ERROR(p) antes de MR/MRX  (equivale a flip con prob p)
"""
from __future__ import annotations

from functools import reduce

import numpy as np
import stim

# Parametros del codigo [[72,12,6]]
BB72 = dict(
    n=72,
    k=12,
    rounds=6,
    l=6,
    m=6,
    A_x_pows=[3],
    A_y_pows=[1, 2],
    B_x_pows=[1, 2],
    B_y_pows=[3],
)


# ---------------------------------------------------------------------------
# Construccion algebraica del codigo BB (portado de la referencia)
# ---------------------------------------------------------------------------
def row_echelon(mat, reduced=False):
    mat = np.asarray(mat, dtype=bool).copy()
    rows, cols = mat.shape
    transform = np.eye(rows, dtype=bool)
    pivot_row = 0
    pivot_cols = []

    for col in range(cols):
        if pivot_row >= rows:
            break
        if not mat[pivot_row, col]:
            swap = pivot_row + np.argmax(mat[pivot_row:rows, col])
            if mat[swap, col]:
                mat[[swap, pivot_row]] = mat[[pivot_row, swap]]
                transform[[swap, pivot_row]] = transform[[pivot_row, swap]]
        if mat[pivot_row, col]:
            elim = range(rows) if reduced else range(pivot_row + 1, rows)
            for r in elim:
                if r != pivot_row and mat[r, col]:
                    mat[r] ^= mat[pivot_row]
                    transform[r] ^= transform[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1

    return mat.astype(np.uint8), pivot_row, transform.astype(np.uint8), pivot_cols


def kernel(mat):
    transposed = np.asarray(mat).T
    rows, _ = transposed.shape
    _, rank, transform, pivot_cols = row_echelon(transposed)
    return transform[rank:rows], rank, pivot_cols


class CSSCode:
    def __init__(self, hx, hz, check_css=False):
        self.hx = np.asarray(hx, dtype=np.uint8)
        self.hz = np.asarray(hz, dtype=np.uint8)
        if self.hx.shape[1] != self.hz.shape[1]:
            raise ValueError("hx and hz must have the same number of columns")
        if check_css and np.any((self.hx @ self.hz.T) % 2):
            raise ValueError("CSS constraint is not satisfied")

        self.N = self.hx.shape[1]
        self.hx_perp, self.rank_hx, self.pivot_hx = kernel(self.hx)
        self.hz_perp, self.rank_hz, self.pivot_hz = kernel(self.hz)
        self.hx_basis = self.hx[self.pivot_hx]
        self.hz_basis = self.hz[self.pivot_hz]
        self.K = self.N - self.rank_hx - self.rank_hz
        self.lx, self.lz = self.compute_logicals()

    def compute_logicals(self):
        def compute_lz(ker_hx, im_hz_t):
            stack = np.vstack([im_hz_t, ker_hx])
            pivots = row_echelon(stack.T)[3]
            indices = [i for i in range(im_hz_t.shape[0], stack.shape[0]) if i in pivots]
            return stack[indices].astype(np.uint8)

        lx = compute_lz(self.hz_perp, self.hx_basis)
        lz = compute_lz(self.hx_perp, self.hz_basis)
        return lx, lz


def create_circulant_matrix(size, powers):
    out = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for p in powers:
            out[(i + p) % size, i] = 1
    return out


def create_bivariate_bicycle_code(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows):
    from scipy.sparse import identity, kron

    s_l = create_circulant_matrix(l, [-1])
    s_m = create_circulant_matrix(m, [-1])
    x = kron(s_l, identity(m, dtype=np.uint8))
    y = kron(identity(l, dtype=np.uint8), s_m)
    A_list = [x ** p for p in A_x_pows] + [y ** p for p in A_y_pows]
    B_list = [y ** p for p in B_y_pows] + [x ** p for p in B_x_pows]
    A = reduce(lambda a, b: a + b, A_list).toarray() % 2
    B = reduce(lambda a, b: a + b, B_list).toarray() % 2
    hx = np.hstack((A, B)).astype(np.uint8)
    hz = np.hstack((B.T, A.T)).astype(np.uint8)
    return CSSCode(hx, hz, check_css=True), A_list, B_list


# ---------------------------------------------------------------------------
# Constructor del circuito (portado de la referencia, sin cambios de logica)
# ---------------------------------------------------------------------------
def build_circuit(code, A_list, B_list, p, rounds, z_basis=True, use_both=True, hzh=False):
    n = code.N
    a1, a2, a3 = A_list
    b1, b2, b3 = B_list

    def nnz(mat):
        rows, cols = mat.nonzero()
        return cols[np.argsort(rows)]

    A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)
    B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)
    A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
    B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)

    x_check = 0
    l_data = n // 2
    r_data = n
    z_check = 3 * n // 2

    p_clifford = p
    p_reset = p
    p_measure = p
    p_round_data = p

    def detector_initial(round_id):
        text = ""
        for i in range(n // 2):
            text += f"DETECTOR(0, 0, {round_id}) rec[{-n // 2 + i}]\n"
        return stim.Circuit(text)

    def detector_repeat(round_id):
        text = ""
        for i in range(n // 2):
            text += f"DETECTOR(0, 0, {round_id}) rec[{-n // 2 + i}] rec[{-n - n // 2 + i}]\n"
        return stim.Circuit(text)

    def append_blocks(circuit, round_id, repeat=False):
        if repeat:
            for i in range(n // 2):
                circuit.append("X_ERROR", z_check + i, p_reset)
                if hzh:
                    circuit.append("X_ERROR", x_check + i, p_reset)
                    circuit.append("H", [x_check + i])
                    circuit.append("DEPOLARIZE1", x_check + i, p_clifford)
                else:
                    circuit.append("Z_ERROR", x_check + i, p_reset)
                circuit.append("DEPOLARIZE1", r_data + i, p_round_data)
        else:
            for i in range(n // 2):
                circuit.append("H", [x_check + i])
                if hzh:
                    circuit.append("DEPOLARIZE1", x_check + i, p_clifford)

        for i in range(n // 2):
            circuit.append("CNOT", [r_data + A1_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [r_data + A1_T[i], z_check + i], p_clifford)
            circuit.append("DEPOLARIZE1", l_data + i, p_round_data)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, l_data + A2[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, l_data + A2[i]], p_clifford)
            circuit.append("CNOT", [r_data + A3_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [r_data + A3_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, r_data + B2[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, r_data + B2[i]], p_clifford)
            circuit.append("CNOT", [l_data + B1_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [l_data + B1_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, r_data + B1[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, r_data + B1[i]], p_clifford)
            circuit.append("CNOT", [l_data + B2_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [l_data + B2_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, r_data + B3[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, r_data + B3[i]], p_clifford)
            circuit.append("CNOT", [l_data + B3_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [l_data + B3_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, l_data + A1[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, l_data + A1[i]], p_clifford)
            circuit.append("CNOT", [r_data + A2_T[i], z_check + i])
            circuit.append("DEPOLARIZE2", [r_data + A2_T[i], z_check + i], p_clifford)
        circuit.append("TICK")

        for i in range(n // 2):
            circuit.append("CNOT", [x_check + i, l_data + A3[i]])
            circuit.append("DEPOLARIZE2", [x_check + i, l_data + A3[i]], p_clifford)
            circuit.append("X_ERROR", z_check + i, p_measure)
            circuit.append("MR", [z_check + i])

        if z_basis:
            circuit += detector_repeat(round_id) if repeat else detector_initial(round_id)
        elif use_both and repeat:
            circuit += detector_repeat(round_id)
        circuit.append("TICK")

        for i in range(n // 2):
            if hzh:
                circuit.append("H", [x_check + i])
                circuit.append("DEPOLARIZE1", x_check + i, p_clifford)
                circuit.append("X_ERROR", x_check + i, p_measure)
                circuit.append("MR", [x_check + i])
            else:
                circuit.append("Z_ERROR", x_check + i, p_measure)
                circuit.append("MRX", [x_check + i])

        if not z_basis:
            circuit += detector_repeat(round_id) if repeat else detector_initial(round_id)
        elif use_both and repeat:
            circuit += detector_repeat(round_id)
        circuit.append("TICK")

    circuit = stim.Circuit()
    for i in range(n // 2):
        circuit.append("R", x_check + i)
        circuit.append("R", z_check + i)
        circuit.append("X_ERROR", x_check + i, p_reset)
        circuit.append("X_ERROR", z_check + i, p_reset)
    for i in range(n):
        circuit.append("R" if z_basis else "RX", l_data + i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", l_data + i, p_reset)

    circuit.append("TICK")
    append_blocks(circuit, round_id=0, repeat=False)
    for round_id in range(1, rounds):
        append_blocks(circuit, round_id=round_id, repeat=True)

    for i in range(n):
        circuit.append("M" if z_basis else "MX", l_data + i)

    pcm = code.hz if z_basis else code.hx
    logical_pcm = code.lz if z_basis else code.lx

    stab_text = ""
    for i, row in enumerate(pcm):
        terms = np.nonzero(row)[0]
        line = f"DETECTOR(0, 0, {rounds})"
        for q in terms:
            line += f" rec[{-n + q}]"
        line += f" rec[{-n - n + i}]" if z_basis else f" rec[{-n - n // 2 + i}]"
        stab_text += line + "\n"
    circuit += stim.Circuit(stab_text)

    obs_text = ""
    for i, row in enumerate(logical_pcm):
        terms = np.nonzero(row)[0]
        line = f"OBSERVABLE_INCLUDE({i})"
        for q in terms:
            line += f" rec[{-n + q}]"
        obs_text += line + "\n"
    circuit += stim.Circuit(obs_text)
    return circuit


def bb72_circuit(p: float, rounds: int | None = None, z_basis: bool = True,
                 use_both: bool = True) -> stim.Circuit:
    """Circuito de memoria Z del [[72,12,6]] a nivel de circuito.

    ``use_both=True`` reproduce los defaults de la referencia (con detectores Z
    duplicados en rondas repetidas: 432 detectores).  ``use_both=False`` produce
    la variante de una sola base (252 detectores) que corresponde a la
    referencia jblue1 ``..._mbFalse_nr6``.
    """
    if rounds is None:
        rounds = BB72["rounds"]
    code, A_list, B_list = create_bivariate_bicycle_code(
        BB72["l"], BB72["m"],
        BB72["A_x_pows"], BB72["A_y_pows"],
        BB72["B_x_pows"], BB72["B_y_pows"],
    )
    return build_circuit(code, A_list, B_list, p=p, rounds=rounds,
                         z_basis=z_basis, use_both=use_both)


# ---------------------------------------------------------------------------
# DEM -> matrices para el decodificador (portado de la referencia)
# ---------------------------------------------------------------------------
def load_dem_matrices(dem: stim.DetectorErrorModel):
    import scipy.sparse

    h_rows, h_cols, l_rows, l_cols, probs = [], [], [], [], []
    var = 0
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        prob = instr.args_copy()[0]
        detectors, observables = [], []
        for target in instr.targets_copy():
            if target.is_logical_observable_id():
                observables.append(target.val)
            elif not target.is_separator():
                detectors.append(target.val)
        if not detectors:
            continue
        for det in detectors:
            h_rows.append(det)
            h_cols.append(var)
        for obs in observables:
            l_rows.append(obs)
            l_cols.append(var)
        probs.append(prob)
        var += 1

    H = scipy.sparse.csr_matrix(
        (np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
        shape=(dem.num_detectors, var),
    )
    L = scipy.sparse.csr_matrix(
        (np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)),
        shape=(dem.num_observables, var),
    )
    return H, L, np.asarray(probs, dtype=np.float64)
