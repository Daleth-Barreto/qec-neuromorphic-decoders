#!/usr/bin/env python3
"""Analiza los barridos T3b/T3c (screen + diag) y produce tabla + figura.

Lee:
  bb_code_decoder_results/snn_circuit/screen/*.json  (pos_weight / balance)
  bb_code_decoder_results/snn_circuit/diag/*.json    (spiking / capacidad / regimen)
y grafica LER vs step, marcando la referencia BP-OSD.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

DEFAULT_DIR = os.path.join("bb_code_decoder_results", "snn_circuit")


def load_hist(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if not d.get("history"):
        return None
    pk = next(iter(d["history"][0]["metrics"]))
    steps = [h["step"] for h in d["history"]]
    lers = [h["metrics"][pk]["ler"] for h in d["history"]]
    oracle = [h["metrics"][pk].get("ler_best_thr") for h in d["history"]]
    spike = [h["metrics"][pk].get("mean_spike_rate") for h in d["history"]]
    ref = d["history"][0]["metrics"][pk].get("ref_ler")
    return {"steps": steps, "ler": lers, "oracle": oracle, "spike": spike,
            "ref": ref, "n_params": d.get("n_params"), "elapsed": d.get("elapsed_min")}


def make_figure(dirpath, outname="t3_diag_summary.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("screen", glob.glob(os.path.join(dirpath, "screen", "*.json"))),
              ("diag", glob.glob(os.path.join(dirpath, "diag", "*.json")))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for gi, (gname, files) in enumerate(groups):
        ax = axes[gi]
        for fp in sorted(files):
            h = load_hist(fp)
            if h is None:
                continue
            tag = os.path.splitext(os.path.basename(fp))[0]
            ax.plot(h["steps"], h["ler"], marker="o", ms=3, label=tag)
            if h["ref"]:
                ax.axhline(h["ref"], ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("LER (p_eval)")
        ax.set_title(f"{gname}: LER vs step (dotted = BP-OSD ref)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(dirpath, outname)
    fig.savefig(out, dpi=130)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()
    out = make_figure(args.dir)
    print(f"figura -> {out}")
    for gname in ("screen", "diag"):
        for fp in sorted(glob.glob(os.path.join(args.dir, gname, "*.json"))):
            h = load_hist(fp)
            if not h:
                continue
            tag = os.path.splitext(os.path.basename(fp))[0]
            print(f"{gname:>7}/{tag:<12} final_LER={h['ler'][-1]:.4f} "
                  f"oracle={h['oracle'][-1]:.4f} ref={h['ref']} "
                  f"params={h['n_params']} ({h['elapsed']:.1f}m)")


if __name__ == "__main__":
    main()
