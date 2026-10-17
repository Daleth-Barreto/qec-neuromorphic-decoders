#!/usr/bin/env python3
"""Consolida el estudio de conversion neuromorfica (ntu_neuro) en figura + JSON."""
from __future__ import annotations

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "bb_code_decoder_results/ntu_neuro"
PS = ("0.003", "0.005")


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def load(p):
    with open(f"{BASE}/ntu_neuro_p{p}.json") as f:
        return json.load(f)


def main():
    data = {p: load(p) for p in PS}
    order = [r["name"] for r in data[PS[0]]["results"]]
    combined = {"metric": "block LER (K=12), pretrained transformer, faithful BB72 correlated",
                "shots": data[PS[0]]["shots"], "by_p": {}}
    for p in PS:
        combined["by_p"][p] = {r["name"]: {k: v for k, v in r.items() if k != "name"}
                               for r in data[p]["results"]}
    with open(f"{BASE}/ntu_neuro_summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"-> {BASE}/ntu_neuro_summary.json")

    # Panel A: LER por configuracion (log x), dos valores de p, con IC 95%
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))
    shots = data[PS[0]]["shots"]
    colors = {"0.003": "tab:blue", "0.005": "tab:orange"}
    for p in PS:
        rlist = data[p]["results"]
        names = [r["name"] for r in rlist]
        lers = np.array([r["ler"] for r in rlist])
        err = []
        for ler in lers:
            k = int(round(ler * shots))
            _, lo, hi = wilson(k, shots)
            err.append([max(ler - lo, 1e-6), max(hi - ler, 1e-6)])
        err = np.array(err).T
        axA.errorbar(lers, names, xerr=err, fmt="o", color=colors[p],
                     capsize=3, label=f"p={p}")
    axA.set_xscale("log")
    axA.set_xlabel("Block LER (log scale), bars = Wilson 95% CI")
    axA.set_title("Quantization and activation: block LER across configurations")
    axA.axvline(data[PS[0]]["results"][0]["ler"], ls=":", color="tab:blue", alpha=0.5)
    axA.legend(fontsize=8)
    axA.grid(True, axis="x", which="both", alpha=0.3)

    # Panel B: LER vs fraccion activa (event-driven sparsity), con IC 95%
    for p in PS:
        pts = [(r["act_fraction"], r["ler"]) for r in data[p]["results"]
               if "act_fraction" in r]
        if pts:
            pts = sorted(pts)
            xs = np.array([x for x, _ in pts])
            ys = np.array([y for _, y in pts])
            err = []
            for ler in ys:
                k = int(round(ler * shots))
                _, lo, hi = wilson(k, shots)
                err.append([max(ler - lo, 1e-6), max(hi - ler, 1e-6)])
            err = np.array(err).T
            axB.errorbar(xs, pts and ys, yerr=err, fmt="-o", color=colors[p],
                         capsize=3, label=f"p={p}")
    axB.axhline(data[PS[0]]["results"][0]["ler"], ls=":", color="tab:blue", alpha=0.5,
                label="baseline p=0.003")
    axB.set_xlabel("Retained activation fraction (|a| > threshold)")
    axB.set_ylabel("Block LER")
    axB.set_yscale("log")
    axB.axvspan(0.0, 0.72, color="green", alpha=0.06)
    axB.set_title("Event-driven decoding: block LER vs activation density")
    axB.legend(fontsize=8)
    axB.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{BASE}/ntu_neuro_summary.png", dpi=140)
    print(f"-> {BASE}/ntu_neuro_summary.png")


if __name__ == "__main__":
    main()
