# coding: utf-8
"""Sanity check FIEL y reproducible del pipeline BB72 (CPU, cero GPU).

Objetivo (honestidad + reproducibilidad):
  1. Construir el circuito de memoria del [[72,12,6]] a nivel de circuito con
     el constructor portado de la referencia publicada (``stim_bb_faithful``).
  2. Decodificarlo con BP+OSD (``ldpc``, con las probabilidades por mecanismo
     del DEM como en la referencia) y comprobar que el LER cae dentro de
     ~2-3x de la referencia publicada de jblue1 (mismo estilo de decodificador),
     NO de las cifras optimizadas de Bravyi (effective-noise matrix).
  3. Cuantificar el bug del circuito viejo (``stim_bb.py``) midiendo su LER con
     EL MISMO decodificador.
  4. Registrar versiones y semillas para reproducibilidad.

Paralelizacion: UN solo ``multiprocessing.Pool`` persistente para toda la
ejecucion (en Windows/spawn el arranque de cada hijo es caro, asi se paga una
vez). Cada tarea lleva el H/probs/L/orden necesarios y el worker cachea el
decodificador por clave. El circuito viejo tambien se evalua por el mismo pool.

Uso:
    python experiments/run_sanity_faithful.py                 # plan completo
    python experiments/run_sanity_faithful.py --quick         # smoke test
    python experiments/run_sanity_faithful.py --workers 4
    python experiments/run_sanity_faithful.py --variants single
    python experiments/run_sanity_faithful.py --buggy-only
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import numpy as np  # noqa: E402
import stim  # noqa: E402

from qecbb.stim_bb_faithful import (  # noqa: E402
    bb72_circuit,
    load_dem_matrices,
)

# ---------------------------------------------------------------------------
# Referencia publicada (jblue1/ml_bb_decoding, BP-OSD, cnbbc-72-12-6, loX, nr6)
#   data/bp_osd_lers_cnbbc-72-12-6_loX_ntcircuit_mbFalse_nr6.csv
# Formato: p -> (n_fail, n_shots, LER)
# ---------------------------------------------------------------------------
JBLUE1_REF = {
    0.001: (107, 330000, 3.24e-4),
    0.002: (250, 55000, 4.5e-3),
    0.003: (1132, 50000, 2.26e-2),
    0.004: (3600, 50000, 7.20e-2),
    0.005: (852, 5000, 1.704e-1),
}

# Variantes de circuito del mismo codigo.
#   single -> use_both=False (252 detectores) = variante jblue1 mbFalse
#   both   -> use_both=True  (432 detectores) = default de la referencia NTU
#             (contiene detectores Z duplicados; DEM mucho mayor y mas lento)
VARIANTS = {"single": dict(use_both=False), "both": dict(use_both=True)}

# Presupuesto de shots (CPU). p bajo -> mas shots para resolver LER pequeno.
SHOTS_BY_P = {
    0.005: 4000,
    0.004: 6000,
    0.003: 8000,
    0.002: 12000,
    0.001: 20000,
}
SHOTS_QUICK = {p: 300 for p in SHOTS_BY_P}

SEED = 20260513
OSD_ORDERS = (3, 10)
MAX_ITER = 12
CHUNK = 128

# ---------------------------------------------------------------------------
# Worker (multiprocessing persistente)
# ---------------------------------------------------------------------------
_W = {"cache": {}}


def _worker_decode(task):
    import scipy.sparse
    from ldpc import BpOsdDecoder

    key, H, probs, L, osd_order, dets = task
    ck = (key, osd_order)
    dec = _W["cache"].get(ck)
    if dec is None:
        dec = BpOsdDecoder(
            scipy.sparse.csr_matrix(H),
            channel_probs=probs,
            max_iter=MAX_ITER,
            bp_method="product_sum",
            schedule="parallel",
            osd_method="OSD_CS",
            osd_order=osd_order,
        )
        _W["cache"][ck] = dec
    n = dets.shape[0]
    pred = np.zeros((n, L.shape[0]), dtype=np.uint8)
    for i in range(n):
        corr = dec.decode(np.asarray(dets[i], dtype=np.uint8))
        pred[i] = (L @ corr) % 2
    return pred


def versions() -> dict:
    import ldpc
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "stim": stim.__version__,
        "ldpc": getattr(ldpc, "__version__", "unknown"),
    }


def _decode_serial(H, probs, Ld, order, dets):
    import scipy.sparse
    from ldpc import BpOsdDecoder

    dec = BpOsdDecoder(
        scipy.sparse.csr_matrix(H),
        channel_probs=probs,
        max_iter=MAX_ITER,
        bp_method="product_sum",
        schedule="parallel",
        osd_method="OSD_CS",
        osd_order=order,
    )
    pred = np.zeros((dets.shape[0], Ld.shape[0]), dtype=np.uint8)
    for i in range(dets.shape[0]):
        corr = dec.decode(np.asarray(dets[i], dtype=np.uint8))
        pred[i] = (Ld @ corr) % 2
    return pred


def evaluate(circuit, p, shots, seed, label, osd_orders, pool=None, key="k"):
    dem = circuit.detector_error_model(decompose_errors=False)
    H, L, probs = load_dem_matrices(dem)

    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    Ld = L.toarray().astype(np.uint8)

    out = {
        "label": label,
        "p": p,
        "shots": int(shots),
        "n_detectors": int(dem.num_detectors),
        "n_observables": int(dem.num_observables),
        "n_error_mechanisms": int(H.shape[1]),
        "trivial_ler": float(np.mean(np.any(obs != 0, axis=1))),
    }

    tasks = []
    slices = {}
    offset = 0
    for order in osd_orders:
        c0 = offset
        for i in range(0, dets.shape[0], CHUNK):
            tasks.append((key, H, probs, Ld, order, dets[i:i + CHUNK]))
            offset += 1
        slices[order] = (c0, offset)

    for order in osd_orders:
        t0 = time.time()
        if pool is None:
            pred = _decode_serial(H, probs, Ld, order, dets)
        else:
            outs = pool.map(_worker_decode, tasks[slices[order][0]:slices[order][1]], chunksize=1)
            pred = np.concatenate(outs, axis=0)
        out[f"bposd_osd{order}_ler"] = float(np.mean(np.any(pred != obs, axis=1)))
        out[f"bposd_osd{order}_t"] = time.time() - t0
    return out


def jblue1_row(p, result):
    ref = JBLUE1_REF[p]
    ours = result.get("bposd_osd10_ler")
    return {
        "p": p,
        "ours_bposd_osd10": ours,
        "ours_bposd_osd3": result.get("bposd_osd3_ler"),
        "trivial": result.get("trivial_ler"),
        "jblue1_ref": ref[2],
        "jblue1_n_fail": ref[0],
        "jblue1_shots": ref[1],
        "ratio_ours_over_ref": (ours / ref[2]) if (ours is not None and ref[2] > 0) else float("inf"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test con pocos shots")
    ap.add_argument("--p", type=float, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--variants", nargs="*", default=["single"], choices=list(VARIANTS))
    ap.add_argument("--buggy-only", action="store_true", help="solo medir el circuito viejo")
    ap.add_argument("--skip-buggy", action="store_true", help="no medir el circuito viejo")
    args = ap.parse_args()

    shots_map = SHOTS_QUICK if args.quick else SHOTS_BY_P
    if args.p:
        p_list = sorted(args.p)
    elif args.quick:
        p_list = [0.001, 0.003, 0.005]
    else:
        p_list = sorted(shots_map)

    print("=== versiones ===")
    ver = versions()
    for k, v in ver.items():
        print(f"  {k}: {v}")
    print(f"\nseed={SEED} max_iter={MAX_ITER} bp=product_sum osd=OSD_CS")
    print(f"workers={args.workers} variants={args.variants} osd_orders={OSD_ORDERS}")
    print(f"shots_plan={ {p: shots_map[p] for p in p_list} }\n", flush=True)

    results = {v: [] for v in args.variants}
    buggy_results = []

    print(f"arrancando pool persistente ({args.workers} workers)...", flush=True)
    t_pool = time.time()
    with Pool(processes=args.workers) as pool:
        print(f"pool listo en {time.time() - t_pool:.1f}s", flush=True)
        for p in p_list:
            shots = shots_map[p]
            print(f"--- p={p} shots={shots} ---", flush=True)
            if not args.buggy_only:
                for idx, name in enumerate(args.variants):
                    orders = OSD_ORDERS if name == "single" else (10,)
                    t0 = time.time()
                    circ = bb72_circuit(p, **VARIANTS[name])
                    r = evaluate(circ, p, shots, seed=SEED,
                                 label=f"faithful_bb72_{name}", osd_orders=orders,
                                 pool=pool, key=f"faithful_{name}_{p}")
                    r["build_s"] = time.time() - t0
                    results[name].append(r)
                    print(
                        f"  faithful[{name}]: trivial={r['trivial_ler']:.4f} "
                        f"BP-OSD3={r.get('bposd_osd3_ler')} "
                        f"BP-OSD10={r.get('bposd_osd10_ler'):.4e} "
                        f"(nD={r['n_detectors']}, nErr={r['n_error_mechanisms']})", flush=True
                    )

            if not args.skip_buggy:
                try:
                    from qecbb.stim_bb import build_bb_memory_z_circuit

                    t0 = time.time()
                    circ_bug = build_bb_memory_z_circuit("[[72,12,6]]", rounds=8, p_dep=p, p_meas=p)
                    rb = evaluate(circ_bug, p, shots, seed=SEED, label="buggy_stim_bb",
                                  osd_orders=OSD_ORDERS, pool=pool, key=f"buggy_{p}")
                    rb["build_s"] = time.time() - t0
                    buggy_results.append(rb)
                    print(
                        f"  buggy:    trivial={rb['trivial_ler']:.4f} "
                        f"BP-OSD3={rb.get('bposd_osd3_ler'):.4e} "
                        f"BP-OSD10={rb.get('bposd_osd10_ler'):.4e} "
                        f"(nD={rb['n_detectors']}, nErr={rb['n_error_mechanisms']})", flush=True
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  buggy:    FALLO -> {type(exc).__name__}: {exc}", flush=True)

    comparison = {}
    if results and any(results.values()):
        print("\n=== comparacion vs jblue1 (BP-OSD osd_order=10) ===")
        for name, rows in results.items():
            if not rows:
                continue
            comparison[name] = [jblue1_row(r["p"], r) for r in rows]
            print(f"[{name}]")
            for row in comparison[name]:
                print(
                    f"  p={row['p']:.4f}  ours={row['ours_bposd_osd10']:.3e}  "
                    f"ref={row['jblue1_ref']:.3e}  ratio={row['ratio_ours_over_ref']:.2f}x  "
                    f"trivial={row['trivial']:.4f}"
                )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "versions": ver,
        "seed": SEED,
        "max_iter": MAX_ITER,
        "bp_method": "product_sum",
        "osd_method": "OSD_CS",
        "osd_orders": list(OSD_ORDERS),
        "workers": args.workers,
        "reference": {
            "source": "jblue1/ml_bb_decoding data/bp_osd_lers_cnbbc-72-12-6_loX_ntcircuit_mbFalse_nr6.csv",
            "values": {str(k): v for k, v in JBLUE1_REF.items()},
        },
        "variants": {k: VARIANTS[k] for k in args.variants},
        "faithful_bb72": results,
        "buggy_stim_bb": buggy_results,
        "comparison_vs_jblue1": comparison,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "bb_code_decoder_results", "sanity_faithful"
    )
    os.makedirs(out_dir, exist_ok=True)
    suffix = "quick" if args.quick else "full"
    json_path = os.path.join(out_dir, f"sanity_faithful_{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nJSON -> {json_path}")


if __name__ == "__main__":
    main()
