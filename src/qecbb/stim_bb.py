# coding: utf-8
import numpy as np
import stim
from .bbcodes import load_bb_code

def _get_logical_z(code):
    Lz = code.lz.toarray()
    k = Lz.shape[0]
    n = code.N
    logs = []
    for i in range(k):
        supp = [j for j in range(n) if Lz[i, j] % 2 == 1]
        logs.append(supp)
    return logs

def build_bb_memory_z_circuit(code_name, rounds, p_dep, p_meas=0.0):
    """Memory experiment circuit-level para el tensor completo via checks Z = Hz.

    Equivalenta al modelo fenomenologico (Hz @ error acumulado) pero con un
    circuito real: ancillas de sindrome, CNOT data->anc, ruido de puertas
    (DEPOLARIZE2) y de memoria/idle (DEPOLARIZE1), ruido de lectura en las
    ancillas (X_ERROR en la ancilla), y lectura final de datos LIMPIA.

    - `rounds+2` rondas de estabilizacion (contadas) => exactamente `rounds`
      capas de detectores (paridades entre rondas r>=2).
    - Detectores: diferencias del sindrome de checks-Z entre rondas.
    - Observables: Z logicas (c.lz) sobre la lectura final de datos.
    """
    c = load_bb_code(code_name)
    Hz = c.hz.toarray()
    n = c.N
    m_z = Hz.shape[0]
    k = c.lz.shape[0]
    circ = stim.Circuit()
    data = list(range(0, n))
    anc = list(range(n, n + m_z))
    for q in data:
        circ.append_operation('R', q)
        if p_dep > 0:
            circ.append_operation('DEPOLARIZE1', q, p_dep)
    prev_rel = None
    for r in range(rounds + 2):
        for a in anc:
            circ.append_operation('R', a)
        for i in range(m_z):
            a = anc[i]
            for j in range(n):
                if Hz[i, j] % 2 == 1:
                    circ.append_operation('CNOT', [j, a])
                    if p_dep > 0:
                        circ.append_operation('DEPOLARIZE2', [j, a], p_dep)
        for a in anc:
            if p_dep > 0:
                circ.append_operation('DEPOLARIZE1', a, p_dep)
            circ.append_operation('M', a)
            if p_meas > 0:
                circ.append_operation('X_ERROR', a, p_meas)
        curr_rel = [-(m_z - i) for i in range(m_z)]
        if prev_rel is not None and r >= 2:
            for i in range(m_z):
                prev_rec = stim.target_rec(prev_rel[i] - m_z)
                curr_rec = stim.target_rec(curr_rel[i])
                circ.append_operation('DETECTOR', [prev_rec, curr_rec])
        prev_rel = curr_rel
    for q in data:
        circ.append_operation('M', q)
    logs_z = _get_logical_z(c)
    for i, supp in enumerate(logs_z):
        targets = []
        for j in supp:
            targets.append(stim.target_rec(-(n - j)))
        circ.append_operation('OBSERVABLE_INCLUDE', targets, i)
    return circ