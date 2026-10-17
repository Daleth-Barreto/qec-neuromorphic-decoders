#!/usr/bin/env python3
"""Consolida el QAT/STE de C1 (ntu_qat) en figura + JSON.

Lee bb_code_decoder_results/ntu_qat/ntu_qat_<cfg>.json (uno por config) y emite:
  - ntu_qat_summary.json con baseline (sin entrenar) vs mejor LER tras QAT + IC Wilson,
  - ntu_qat_summary.png (panel A: baseline vs QAT con IC; panel B: curvas de LER).
"""
from __future__ import annotations

import glob
import json
import math
import os

BASE = "bb_code_decoder_results/ntu_qat"
ORDER = ["gelu", "gelu_w4a8", "relu", "relu_w4a8", "relu_w4a8_s01", "lif"]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def load(name):
    path = f"{BASE}/ntu_qat_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize(d):
    shots = d["shots"]
    hist = d.get("history", [])
    base = d["baseline_ler"]
    b = min(h["ler"] for h in hist) if hist else None
    bstep = min(hist, key=lambda h: h["ler"])["step"] if hist else None
    fin = hist[-1]["ler"] if hist else None
    act = next((h["act_fraction"] for h in reversed(hist)
                if h.get("act_fraction") is not None), None)
    _, blo, bhi = wilson(int(round(base * shots)), shots)
    _, lo, hi = (wilson(int(round(b * shots)), shots) if b is not None else (0, 0, 0))
    return dict(name=d.get("act"), w_bits=d.get("w_bits"), a_bits=d.get("a_bits"),
                sparsity_thr=d.get("sparsity_thr"), batch=d.get("batch"),
                shots=shots, steps=d.get("steps"),
                baseline_ler=base, baseline_ci=[blo, bhi],
                best_ler=b, best_step=bstep, best_ci=[lo, hi],
                final_ler=fin, act_fraction=act,
                history=hist)


def main():
    loaded = {n: load(n) for n in ORDER}
    loaded = {n: d for n, d in loaded.items() if d is not None}
    rows = [summarize(loaded[n]) | {"cfg": n} for n in ORDER if n in loaded]

    combined = {"metric": "block LER (K=12), QAT/STE fine-tune, faithful BB72",
                "p": 0.003, "configs": rows}
    with open(f"{BASE}/ntu_qat_summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"-> {BASE}/ntu_qat_summary.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        be = [max(r["baseline_ler"] - r["baseline_ci"][0], 1e-6),
              max(r["baseline_ci"][1] - r["baseline_ler"], 1e-6)]
        qe = [max(r["best_ler"] - r["best_ci"][0], 1e-6),
              max(r["best_ci"][1] - r["best_ler"], 1e-6)]
        axA.errorbar(r["baseline_ler"], y, xerr=np.array(be).T.reshape(2, 1),
                     fmt="o", mfc="none", color="tab:gray", capsize=3,
                     label="no training" if y == ys[0] else None)
        axA.errorbar(r["best_ler"], y, xerr=np.array(qe).T.reshape(2, 1),
                     fmt="o", color="tab:red", capsize=3,
                     label="after QAT (best)" if y == ys[0] else None)
    axA.set_yticks(ys)
    axA.set_yticklabels([r["cfg"] for r in rows])
    axA.set_xscale("log")
    axA.set_xlabel("Block LER (log scale), bars = Wilson 95% CI")
    axA.set_title("QAT/STE: collapse recovery")
    axA.grid(True, axis="x", which="both", alpha=0.3)
    axA.legend(fontsize=8)

    for r in rows:
        hist = r["history"]
        steps = [h["step"] for h in hist]
        lers = [h["ler"] for h in hist]
        axB.plot(steps, lers, marker="o", ms=3, label=r["cfg"])
        axB.axhline(r["baseline_ler"], ls=":", lw=1, alpha=0.4)
    axB.set_xlabel("training step")
    axB.set_ylabel("Block LER")
    axB.set_yscale("log")
    axB.set_title("Learning curves (dotted = no-training baseline)")
    axB.grid(True, which="both", alpha=0.3)
    axB.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{BASE}/ntu_qat_summary.png", dpi=140)
    print(f"-> {BASE}/ntu_qat_summary.png")


if __name__ == "__main__":
    main()
