#!/usr/bin/env python3
"""Consolida la referencia SOTA (NTU / Blue et al. 2026) sobre BB72 fiel.

Lee los CSV ``bb36_transformer_eval_r6_p*.csv`` producidos por el kernel
``dalethhernandez/qecbb-t4-ntu-ref`` (T4, 5000 shots/p) y emite:
  - ``ntu_ref_summary.json`` con LER + IC Wilson,
  - ``ntu_ref_summary.png`` comparando transformer vs BP-OSD fiel (gate).
"""
from __future__ import annotations

import csv
import json
import os

BASE = "bb_code_decoder_results/ntu_ref/third_party/ntu"
PS = (0.001, 0.002, 0.003, 0.004, 0.005)

BPOSD_UNCORR = {0.001: 2.00e-4, 0.002: 3.50e-3, 0.003: 1.24e-2,
                0.004: 3.85e-2, 0.005: 8.98e-2}
BPOSD_CORR_P003 = 0.0787


def wilson_ci(k_fail, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = k_fail / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    rows = []
    for p in PS:
        with open(f"{BASE}/bb36_transformer_eval_r6_p{p}.csv") as f:
            r = list(csv.DictReader(f))[-1]
        n, k = int(r["total_samples"]), int(r["errors"])
        lo, hi = wilson_ci(k, n)
        rows.append(dict(p=p, n=n, errors=k, ler=float(r["ler"]),
                         ler_lo=lo, ler_hi=hi,
                         per_log_mean=float(r["per_log_mean"]),
                         per_log_worst=float(r["per_log_worst"])))
        print(f"p={p}: LER={float(r['ler']):.2e} CI[{lo:.2e},{hi:.2e}] "
              f"per_log_mean={r['per_log_mean']} worst={r['per_log_worst']}")

    summary = {
        "source": "dalethhernandez/qecbb-t4-ntu-ref (T4, bb/bb72_transformer.pt)",
        "model": "AlphaQubitV2_BB (Blue et al. 2026), 35.98M params, step 10500",
        "circuit": "BB72 fiel use_both/correlated, rounds=6, 432 det, 12 logicals",
        "metric": "block LER (K=12), 5000 shots/p",
        "transformer": rows,
        "bposd_corr_p0.003": BPOSD_CORR_P003,
        "bposd_uncorr": BPOSD_UNCORR,
    }
    out_json = "bb_code_decoder_results/ntu_ref/ntu_ref_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"-> {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ps = [r["p"] for r in rows]
        ler = [r["ler"] for r in rows]
        lo = [r["ler"] - r["ler_lo"] for r in rows]
        hi = [r["ler_hi"] - r["ler"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.errorbar(ps, ler, yerr=[lo, hi], marker="o", capsize=3,
                    label="Transformer (NTU / Blue et al.)")
        pu = sorted(BPOSD_UNCORR)
        ax.plot(pu, [BPOSD_UNCORR[k] for k in pu], "s--", label="BP-OSD (uncorrelated)")
        ax.axhline(BPOSD_CORR_P003, ls=":", color="gray",
                   label="BP-OSD corr @p=0.003")
        ax.set_yscale("log")
        ax.set_xlabel("p (error rate)")
        ax.set_ylabel("Block LER (K=12)")
        ax.set_title("BB72 (faithful): transformer SOTA vs BP-OSD")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_png = "bb_code_decoder_results/ntu_ref/ntu_ref_summary.png"
        plt.savefig(out_png, dpi=140)
        print(f"-> {out_png}")
    except Exception as e:
        print(f"[aviso] figura no generada: {e}")


if __name__ == "__main__":
    main()
