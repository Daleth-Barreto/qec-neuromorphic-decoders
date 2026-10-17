"""Construcción de BB codes y carga (fiel a notebook)."""
import numpy as np
from bposd.css import css_code

from .config import CONFIG

def shift_matrix(n):
    S = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


def _matpow(M, p):
    n = M.shape[0]
    R = np.eye(n, dtype=np.int64)
    for _ in range(p):
        R = R @ M
    return R


def build_bb_code(l, m, a_terms, b_terms):
    Sl, Sm = shift_matrix(l), shift_matrix(m)
    Il, Im = np.eye(l, dtype=np.int64), np.eye(m, dtype=np.int64)
    x = np.kron(Sl, Im)
    y = np.kron(Il, Sm)
    identity = np.eye(l * m, dtype=np.int64)

    def term(var, p):
        if var == "__id":
            return identity
        return _matpow(x, p) if var == "x" else _matpow(y, p)

    A = sum(term(v, p) for v, p in a_terms) % 2
    B = sum(term(v, p) for v, p in b_terms) % 2

    Hx = np.concatenate([A, B], axis=1) % 2
    Hz = np.concatenate([B.T, A.T], axis=1) % 2
    return Hx.astype(np.uint8), Hz.astype(np.uint8)


def load_bb_code(code_name: str | None = None):
    if code_name is None:
        code_name = CONFIG["CODE_NAME"]
    spec = CONFIG["BB_CODES"][code_name]
    Hx, Hz = build_bb_code(spec["l"], spec["m"], spec["A"], spec["B"])
    code = css_code(hx=Hx, hz=Hz)
    code.test()
    return code

def get_code_name_info(code_name: str | None = None):
    c = load_bb_code(code_name)
    return {
        "name": code_name if code_name else CONFIG["CODE_NAME"],
        "N": int(c.N),
        "N_CHECKS_Z": int(c.hz.shape[0]),
        "N_CHECKS_X": int(c.hx.shape[0]),
        "K_LOGICAL": int(c.lz.shape[0]),
        "Hz_np": c.hz.astype(np.uint8),
        "Lz_np": c.lz.astype(np.int64),
    }