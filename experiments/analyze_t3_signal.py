#!/usr/bin/env python3
"""Analiza los JSON de T3 (senal SNN vs BP-OSD) y produce tabla + figura.

Uso:
    python experiments/analyze_t3_signal.py [--dir DIR]

Lee ``train_log_<variant>_0.003.json`` de ambos variantes y reporta:
  - LER SNN (mejor) con IC95, LER BP-OSD, ratio, spike rate (actividad),
    params y aristas.
  - Curva de aprendizaje (LER vs step) y figura LER-vs-actividad.
"""
from __future__ import annotations

import argparse
import json
import os

VARIANTS = ("uncorrelated", "correlated")
DEFAULT_DIR = os.path.join("bb_code_decoder_results", "snn_circuit")


def load(dirpath, variant):
    path = os.path.join(dirpath, f"train_log_{variant}_0.003.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def best_record(d):
    best = None
    for h in d.get("history", []):
        for _p, m in h["metrics"].items():
            key = (m["ler"], h["step"])
            if best is None or key < best[0]:
                best = (key, m, h["step"])
    return best


def summarize(dirpath):
    rows = []
    for v in VARIANTS:
        d = load(dirpath, v)
        if d is None:
            print(f"[{v}] sin JSON")
            continue
        b = best_record(d)
        m, step = (b[1], b[2]) if b else ({}, None)
        ref = m.get("ref_ler")
        rows.append({
            "variant": v,
            "step_best": step,
            "snn_ler": m.get("ler"),
            "snn_ci": (m.get("ci_lo"), m.get("ci_hi")),
            "ref_ler": ref,
            "ratio": m.get("ratio_snn_over_ref"),
            "spike_rate": m.get("mean_spike_rate"),
            "n_params": d.get("n_params"),
            "n_edges": d.get("n_edges"),
            "elapsed_min": round(d.get("elapsed_min", 0.0), 1),
            "stopped": d.get("stopped_reason"),
        })
    print("\n=== T3 resumen (mejor LER por variante) ===")
    hdr = f"{'variant':<13}{'step':>6}{'snn_ler':>12}{'ci95':>22}{'ref_ler':>12}{'ratio':>8}{'spike':>8}{'params':>8}{'edges':>7}{'min':>7}"
    print(hdr)
    for r in rows:
        ci = r["snn_ci"]
        ci_s = f"[{ci[0]:.2e},{ci[1]:.2e}]" if ci[0] is not None else "-"
        def fmt(x, p=".3e"):
            return "-" if x is None else format(x, p)
        print(f"{r['variant']:<13}{str(r['step_best']):>6}{fmt(r['snn_ler']):>12}"
              f"{ci_s:>22}{fmt(r['ref_ler']):>12}{fmt(r['ratio'],'.2f'):>8}"
              f"{fmt(r['spike_rate'],'.3f'):>8}{str(r['n_params']):>8}"
              f"{str(r['n_edges']):>7}{r['elapsed_min']:>7}")
    return rows


def make_figure(dirpath, rows, outname="t3_signal_summary.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] matplotlib no disponible: {exc}")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"uncorrelated": "#1f77b4", "correlated": "#d62728"}

    for v in VARIANTS:
        d = load(dirpath, v)
        if d is None or not d.get("history"):
            continue
        step_key = next(iter(d["history"][0]["metrics"]))
        steps = [h["step"] for h in d["history"]]
        lers = [h["metrics"][step_key]["ler"] for h in d["history"]]
        axes[0].plot(steps, lers, marker="o", ms=3, color=colors[v], label=v)
        ref = d["history"][0]["metrics"][step_key].get("ref_ler")
        if ref:
            axes[0].axhline(ref, ls="--", lw=1, color=colors[v], alpha=0.6)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("LER (p=0.003)")
    axes[0].set_yscale("log")
    axes[0].set_title("Learning curve (dashed = BP-OSD ref)")
    axes[0].legend()

    for r in rows:
        if r["snn_ler"] is None or r["spike_rate"] is None:
            continue
        axes[1].scatter(r["spike_rate"], r["snn_ler"], s=60,
                        color=colors.get(r["variant"], "k"), label=r["variant"])
        if r["ref_ler"]:
            axes[1].axhline(r["ref_ler"], ls="--", lw=1,
                            color=colors.get(r["variant"], "k"), alpha=0.6)
    axes[1].set_xlabel("Mean spike rate (activity)")
    axes[1].set_ylabel("LER")
    axes[1].set_yscale("log")
    axes[1].set_title("LER vs activity (neuromorphic efficiency)")
    axes[1].legend()

    out = os.path.join(dirpath, outname)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"figura -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()
    rows = summarize(args.dir)
    make_figure(args.dir, rows)
    outj = os.path.join(args.dir, "t3_signal_summary.json")
    with open(outj, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"resumen -> {outj}")


if __name__ == "__main__":
    main()
