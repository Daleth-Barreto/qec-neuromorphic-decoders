#!/usr/bin/env python3
"""Consolida la ronda 2 de QAT/STE (schedule coseno) de C1.

Lee bb_code_decoder_results/ntu_qat/ntu_qat_<cfg>.json (uno por config) y emite:
  - ntu_qat2_summary.json: no-training baseline vs mejor LER tras QAT + IC Wilson,
    mas el LER fp32 de referencia (si esta disponible en ntu_neuro),
  - ntu_qat2_summary.png: panel A baseline vs QAT-best con IC, panel B curvas.

Preguntas: W4A4 (colapso post-training) ?se recupera?; cliff sparsity tau=0.20
?se empuja?; controles GELU ?mejoran con schedule coseno?
"""
from __future__ import annotations

import json
import math
import os

BASE = "bb_code_decoder_results/ntu_qat"
ORDER = [
    "gelu_cos", "gelu_w4a8_cos", "relu_w4a8_cos",
    "relu_w4a4_cos", "gelu_w4a4_cos",
    "relu_w4a8_s015_cos", "relu_w4a8_s02_cos",
]
LABEL = {
    "gelu_cos": "GELU (ctrl)",
    "gelu_w4a8_cos": "GELU W4A8",
    "relu_w4a8_cos": "ReLU W4A8",
    "relu_w4a4_cos": "ReLU W4A4",
    "gelu_w4a4_cos": "GELU W4A4",
    "relu_w4a8_s015_cos": "ReLU W4A8 s=0.15",
    "relu_w4a8_s02_cos": "ReLU W4A8 s=0.20",
}


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


def fp32_ref():
    ref = "bb_code_decoder_results/ntu_neuro/ntu_neuro_summary.json"
    if not os.path.exists(ref):
        return None
    with open(ref) as f:
        d = json.load(f)
    cfgs = d.get("by_p", {}).get("0.003", {})
    for key in ("fp32", "w8a8", "w8a16"):
        if key in cfgs:
            return cfgs[key].get("ler")
    return None


def summarize(d):
    shots = d["shots"]
    hist = d.get("history", [])
    base = d["baseline_ler"]
    b = d.get("best_ler")
    bstep = d.get("best_step")
    if b is None and hist:
        b = min(h["ler"] for h in hist)
        bstep = min(hist, key=lambda h: h["ler"])["step"]
    fin = hist[-1]["ler"] if hist else None
    act = next((h["act_fraction"] for h in reversed(hist)
                if h.get("act_fraction") is not None), None)
    _, blo, bhi = wilson(int(round(base * shots)), shots)
    _, lo, hi = (wilson(int(round(b * shots)), shots) if b is not None else (0, 0, 0))
    return dict(name=d.get("act"), w_bits=d.get("w_bits"), a_bits=d.get("a_bits"),
                sparsity_thr=d.get("sparsity_thr"), batch=d.get("batch"),
                lr_schedule=d.get("lr_schedule"),
                shots=shots, steps=d.get("steps"),
                baseline_ler=base, baseline_ci=[blo, bhi],
                best_ler=b, best_step=bstep, best_ci=[lo, hi],
                final_ler=fin, act_fraction=act,
                history=hist)


def main():
    loaded = {n: load(n) for n in ORDER}
    loaded = {n: d for n, d in loaded.items() if d is not None}
    if not loaded:
        print("sin JSONs todavia")
        return
    rows = [summarize(loaded[n]) | {"cfg": n, "label": LABEL.get(n, n)}
            for n in ORDER if n in loaded]

    ref = fp32_ref()
    combined = {"metric": "block LER (K=12), QAT/STE fine-tune (cosine), BB72",
                "p": 0.003, "fp32_ref_ler": ref, "configs": rows}
    with open(f"{BASE}/ntu_qat2_summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"-> {BASE}/ntu_qat2_summary.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                        "axes.labelsize": 11, "legend.fontsize": 9})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.8))
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
                     label="QAT best (cosine)" if y == ys[0] else None)
        if r["baseline_ler"] > 5 * max(r["best_ler"], 1e-4):
            axA.annotate("", xy=(r["best_ler"], y), xytext=(r["baseline_ler"], y),
                         arrowprops=dict(arrowstyle="->", color="tab:green",
                                         alpha=0.7, lw=1.4))
    if ref:
        axA.axvline(ref, ls="--", lw=1, color="tab:blue", alpha=0.6)
        axA.text(ref, len(rows) - 0.4, " fp32 (no quant)", color="tab:blue",
                 fontsize=7, va="top")
    axA.set_yticks(ys)
    axA.set_yticklabels([r["label"] for r in rows])
    axA.set_xlabel("Block LER")
    axA.set_xscale("log")
    axA.set_title("QAT recovery (cosine schedule)")
    axA.grid(True, which="both", axis="x", alpha=0.3)
    axA.legend(fontsize=8, loc="best")

    for r in rows:
        steps = [h["step"] for h in r["history"]]
        lers = [h["ler"] for h in r["history"]]
        axB.plot(steps, lers, marker="o", ms=3, label=r["label"])
        axB.axhline(r["baseline_ler"], ls=":", lw=1, alpha=0.3)
    if ref is not None:
        axB.axhline(ref, ls="--", lw=1, color="k", alpha=0.5,
                    label="fp32 (no quant)")
    axB.set_xlabel("training step")
    axB.set_ylabel("Block LER")
    axB.set_yscale("log")
    axB.set_title("Learning curves (dotted = no-training baseline)")
    axB.grid(True, which="both", alpha=0.3)
    axB.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{BASE}/ntu_qat2_summary.png", dpi=300)
    print(f"-> {BASE}/ntu_qat2_summary.png")


if __name__ == "__main__":
    main()
