#!/usr/bin/env python3
"""Pruebas de paridad T0-T2 del pipeline BB72 (CPU).

  T0  Estructura: conteo de detectores y observables por variante.
  T1  Aprendibilidad: overfit de una muestra diminuta (64) -> la LER de
      entrenamiento debe caer a ~0 (memorizacion). Ambas variantes.
  T2  Consistencia: roundtrip del mapeo detectores<->features, LER de BP-OSD
      sobre el DEM de la variante, y comparacion SNN vs BP-OSD sobre los
      MISMOS shots.

Uso:
    python experiments/run_parity_checks.py --all
    python experiments/run_parity_checks.py --t1 --t2 --variant correlated
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from qecbb import bposd_ref as R  # noqa: E402
from qecbb import stim_bb_circuit_data as S  # noqa: E402
from qecbb.evaluation import wilson_ci  # noqa: E402
from qecbb.model import GAT_LIF_BBDecoder  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(ROOT, "bb_code_decoder_results", "parity_checks")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_loader(feats, labels, edge_index, batch_size, shuffle):
    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    ds = [Data(x=x[i], edge_index=edge_index, y=y[i].unsqueeze(0)) for i in range(len(x))]
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), ds


def block_ler_from_logits(logits, labels):
    pred = (logits > 0).astype(np.uint8)
    obs = np.asarray(labels, dtype=np.uint8)
    k = int(np.sum(np.any(pred != obs, axis=1)))
    ler, lo, hi = wilson_ci(k, obs.shape[0])
    return {"ler": float(ler), "ci_lo": float(lo), "ci_hi": float(hi), "k_fail": k,
            "n": int(obs.shape[0])}


def train_model(variant, steps, batch_size, lr, seed, num_steps, hidden, num_layers,
                n_train, p_train, verbose=True):
    torch.manual_seed(seed)
    feats, labels, _ = S.generate_circuit_data(n_train, p_train, variant, seed=seed)
    edge_index, nchk = S.build_variant_graph(variant)
    loader, ds = make_loader(feats, labels, edge_index, batch_size, shuffle=True)
    model = GAT_LIF_BBDecoder(in_channels=feats.shape[2], k_logical=12,
                              hidden_channels=hidden, num_layers=num_layers,
                              num_steps=num_steps, spiking=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = torch.nn.BCEWithLogitsLoss()
    hist = []
    step = 0
    it = iter(loader)
    model.train()
    t0 = time.time()
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        opt.zero_grad()
        out = model(batch)
        loss = crit(out, batch.y)
        loss.backward()
        opt.step()
        step += 1
        if verbose and step % max(steps // 10, 1) == 0:
            log(f"    [{variant}] step {step}/{steps} loss={float(loss):.4f} "
                f"({time.time()-t0:.0f}s)")
    # train LER on the full pool (no shuffle)
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=256, shuffle=False):
            outs.append(model(batch).cpu())
    logits = torch.cat(outs).numpy()
    metrics = block_ler_from_logits(logits, labels)
    metrics["steps"] = steps
    metrics["n_train"] = n_train
    metrics["seconds"] = time.time() - t0
    return model, metrics


# ---------------------------------------------------------------------------
def t0():
    log("=== T0: estructura de detectores ===")
    res = {}
    for v in S.VARIANTS:
        c = S.build_variant_circuit(0.003, v)
        res[v] = {"n_detectors": int(c.num_detectors),
                  "n_observables": int(c.num_observables),
                  "n_checks": int(S.n_checks(v)), "T_slices": int(S.T_SLICES)}
        log(f"  {v}: {res[v]}")
    assert res["uncorrelated"]["n_detectors"] == 252
    assert res["correlated"]["n_detectors"] == 432
    assert res["uncorrelated"]["n_checks"] == 36
    assert res["correlated"]["n_checks"] == 72
    assert all(res[v]["n_observables"] == 12 for v in res)
    log("  OK (252/36 y 432/72, obs=12)")
    return res


def t1(variant, steps, num_steps, hidden, num_layers, n_train, p_train, seed):
    log(f"=== T1: overfit {n_train} muestras [{variant}] steps={steps} ===")
    _, metrics = train_model(variant, steps, batch_size=n_train, lr=5e-3, seed=seed,
                             num_steps=num_steps, hidden=hidden, num_layers=num_layers,
                             n_train=n_train, p_train=p_train)
    log(f"  [{variant}] final train LER={metrics['ler']:.3e} "
        f"CI=[{metrics['ci_lo']:.2e},{metrics['ci_hi']:.2e}] ({metrics['seconds']:.0f}s)")
    metrics["pass"] = bool(metrics["ler"] < 0.1)
    log(f"  [{variant}] {'PASS' if metrics['pass'] else 'FAIL'} (umbral LER<0.1)")
    return metrics


def t2(variant, ref_shots, eval_shots, train_steps, p, num_steps, hidden, num_layers,
       n_train, seed):
    log(f"=== T2: consistencia + SNN vs BP-OSD [{variant}] ===")
    res = {}

    # (a) roundtrip encoding
    fe, la, an, dets = S.generate_circuit_data(eval_shots, p, variant, seed=seed,
                                               return_dets=True)
    rt = S.features_to_detectors(fe, variant)
    res["roundtrip_ok"] = bool(np.array_equal(rt, dets))
    log(f"  roundtrip features<->detectors: {res['roundtrip_ok']}")
    assert res["roundtrip_ok"], "el mapeo detectores<->features no es biyectivo"

    # (b) BP-OSD sobre los mismos shots
    circuit = S.build_variant_circuit(p, variant)
    m = min(ref_shots, dets.shape[0])
    dec_L = R.decoder_for(circuit, osd_order=3)
    t = time.time()
    pred = R.decode_observables(circuit, dets[:m], decoder_and_L=dec_L)
    bposd_ler = R.block_ler(pred, la[:m])
    k = int(np.sum(np.any(pred != la[:m], axis=1)))
    _, lo, hi = wilson_ci(k, m)
    res["bposd"] = {"ler": float(bposd_ler), "ci_lo": float(lo), "ci_hi": float(hi),
                    "shots": m, "seconds": time.time() - t}
    log(f"  BP-OSD(osd3) LER={bposd_ler:.3e} CI=[{lo:.2e},{hi:.2e}] ({m} shots, "
        f"{time.time()-t:.1f}s)")

    # (c) SNN entrenado corto vs BP-OSD en los MISMOS shots
    model, tr = train_model(variant, train_steps, batch_size=128, lr=2e-3, seed=seed,
                            num_steps=num_steps, hidden=hidden, num_layers=num_layers,
                            n_train=n_train, p_train=p)
    x = torch.tensor(fe, dtype=torch.float32)
    edge_index, _ = S.build_variant_graph(variant)
    ds = [Data(x=x[i], edge_index=edge_index, y=torch.tensor(la[i]).unsqueeze(0))
          for i in range(len(x))]
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=256, shuffle=False):
            outs.append(model(batch).cpu())
    logits = torch.cat(outs).numpy()
    snn = block_ler_from_logits(logits, la)
    snn["train_ler"] = tr["ler"]
    res["snn"] = snn
    res["ratio_snn_over_bposd"] = (snn["ler"] / bposd_ler) if bposd_ler > 0 else None
    log(f"  SNN({train_steps} pasos) LER={snn['ler']:.3e} "
        f"train_LER={tr['ler']:.3e} | ratio vs BP-OSD={res['ratio_snn_over_bposd']}")
    res["pass"] = bool(res["roundtrip_ok"] and bposd_ler < 0.5)
    log(f"  {variant} {'PASS' if res['pass'] else 'FAIL'} (roundtrip + BP-OSD razonable)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", action="store_true")
    ap.add_argument("--t1", action="store_true")
    ap.add_argument("--t2", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variant", choices=list(S.VARIANTS), default=None,
                    help="limita T1/T2 a una variante")
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t1-steps", type=int, default=200)
    ap.add_argument("--t1-samples", type=int, default=64)
    ap.add_argument("--t2-train-steps", type=int, default=300)
    ap.add_argument("--t2-train-samples", type=int, default=8192)
    ap.add_argument("--t2-eval-shots", type=int, default=2000)
    ap.add_argument("--t2-ref-shots", type=int, default=1000)
    ap.add_argument("--num-steps", type=int, default=8, help="pasos SNN (CPU rapido)")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=2)
    args = ap.parse_args()

    if not any([args.t0, args.t1, args.t2, args.all]):
        args.all = True
    variants = [args.variant] if args.variant else list(S.VARIANTS)

    results = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "config": vars(args)}
    os.makedirs(OUT_DIR, exist_ok=True)

    jpath = os.path.join(OUT_DIR, "parity_checks.json")
    if os.path.exists(jpath):
        try:
            with open(jpath, encoding="utf-8") as fh:
                prev = json.load(fh)
            for key in ("t0", "t1", "t2"):
                if key in prev:
                    results.setdefault(key, {})
                    if isinstance(prev[key], dict) and isinstance(results.get(key), dict):
                        results[key].update(prev[key])
                    else:
                        results[key] = prev[key]
            log("fusionando con parity_checks.json existente")
        except (OSError, ValueError):
            pass

    if args.t0 or args.all:
        results["t0"] = t0()
    if args.t1 or args.all:
        results.setdefault("t1", {})
        for v in variants:
            results["t1"][v] = t1(v, args.t1_steps, args.num_steps, args.hidden,
                                  args.num_layers, args.t1_samples, args.p, args.seed)
    if args.t2 or args.all:
        results.setdefault("t2", {})
        for v in variants:
            results["t2"][v] = t2(v, args.t2_ref_shots, args.t2_eval_shots,
                                  args.t2_train_steps, args.p, args.num_steps,
                                  args.hidden, args.num_layers, args.t2_train_samples,
                                  args.seed)

    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"JSON -> {jpath}")

    # resumen
    log("=== resumen ===")
    if "t1" in results:
        for v, r in results["t1"].items():
            log(f"  T1[{v}] train_LER={r['ler']:.3e} {'PASS' if r.get('pass') else 'FAIL'}")
    if "t2" in results:
        for v, r in results["t2"].items():
            log(f"  T2[{v}] roundtrip={r['roundtrip_ok']} bposd={r['bposd']['ler']:.2e} "
                f"snn={r['snn']['ler']:.2e} ratio={r['ratio_snn_over_bposd']} "
                f"{'PASS' if r.get('pass') else 'FAIL'}")


if __name__ == "__main__":
    main()
