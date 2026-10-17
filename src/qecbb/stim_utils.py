# coding: utf-8
import numpy as np
import stim

def bb_circuit_level_memorystats(code_name, rounds, p):
    # build simple memory experiment circuit for CSS; use stim's generic surface-like or just placeholders
    # but BB specific: we can use surface approximation? not exact. Start with generic memory
    c = stim.Circuit()
    c.append_operation('X_ERROR', range(1), p)
    for t in range(rounds):
        c.append_operation('MR', range(1), p)
    c.append_operation('M', range(1))
    return c
