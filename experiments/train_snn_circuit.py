#!/usr/bin/env python3
"""Entrenamiento del SNN GAT-LIF sobre el circuito BB72 fiel (CPU por defecto).

Disenado para pruebas cortas honestas antes de comprometer GPU:
  - flags completos,
  - checkpoints best/last + resume,
  - progreso (paso, loss, LER-val, ratio BP-OSD, step/s, ETA),
  - avisos (NaN, estancado, sin mejora, tiempo),
  - early-stop por paciencia y por tiempo.

Dos variantes de tarea (mismos pesos de modelo, distinto n_checks):
  --variant uncorrelated   36 checks (252 detectores) -- comparable jblue1 mbFalse
  --variant correlated     72 checks (432 detectores) -- tarea por defecto NTU

Ejemplos:
    python experiments/train_snn_circuit.py --smoke
    python experiments/train_snn_circuit.py --variant correlated --p-train 0.003 \
        --p-eval 0.003 --n-train 20000 --n-eval 4000 --steps 2000 --eval-every 200
    python experiments/train_snn_circuit.py --variant uncorrelated \
        --p-train-min 0.001 --p-train-max 0.005 --steps 16000 --batch-size 128 \
        --eval-every 500 --ckpt-every 1000 --early-stop-patience 6 --max-wall-minutes 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from qecbb import stim_bb_circuit_data as S  # noqa: E402
from qecbb.evaluation import wilson_ci  # noqa: E402
from qecbb.model import GAT_LIF_BBDecoder  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_OUT = os.path.join(ROOT, "bb_code_decoder_results", "snn_circuit")


# ---------------------------------------------------------------------------
def log(msg, flush=True):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=flush)


class TensorGraphDataset(torch.utils.data.Dataset):
    """Dataset perezoso: guarda tensores y materializa ``Data`` por indice."""

    def __init__(self, feats, labels, edge_index):
        self.x = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.edge_index = edge_index
        self.num_edges = int(edge_index.shape[1])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return Data(x=self.x[i], edge_index=self.edge_index,
                    y=self.y[i].unsqueeze(0))


def build_dataset(feats, labels, edge_index):
    return TensorGraphDataset(feats, labels, edge_index)


def sample_train_pool(n_train, variant, seed, p_fixed=None, p_min=None, p_max=None,
                      grid=5, balance=False):
    """Genera (feats, labels) de entrenamiento; p fijo o una rejilla [p_min,p_max]."""
    rng = np.random.default_rng(seed)
    if p_fixed is not None:
        ps = [p_fixed]
    else:
        ps = list(np.linspace(p_min, p_max, grid))
    feats_all, labels_all = [], []
    per = max(1, n_train // len(ps))
    for p in ps:
        f, y, _ = S.generate_circuit_data(per, float(p), variant,
                                          seed=int(rng.integers(0, 2**31 - 1)),
                                          balance=balance)
        feats_all.append(f)
        labels_all.append(y)
    feats = np.concatenate(feats_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    idx = rng.permutation(len(feats))
    return feats[idx], labels[idx], ps


def eval_ler(model, ds, labels, batch_size, device, want_spikes=True):
    model.eval()
    outs, rates, sizes = [], [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            if want_spikes:
                out, rate = model(batch, return_spike_stats=True)
                rates.append(float(rate))
                sizes.append(int(batch.num_graphs))
            else:
                out = model(batch)
            outs.append(out.detach().cpu())
    logits = torch.cat(outs, dim=0).numpy()
    obs = np.asarray(labels, dtype=np.uint8)
    pred = (logits > 0).astype(np.uint8)
    k_fail = int(np.sum(np.any(pred != obs, axis=1)))
    n = int(obs.shape[0])
    ler, lo, hi = wilson_ci(k_fail, n)
    res = {"ler": float(ler), "ci_lo": float(lo), "ci_hi": float(hi),
           "k_fail": k_fail, "n": n}
    # Diagnostico: mejor umbral sobre logits (oraculo, misma muestra de eval)
    best = (ler, 0.0)
    for thr in np.linspace(-2.0, 2.0, 81):
        p2 = (logits > thr).astype(np.uint8)
        l2 = float(np.mean(np.any(p2 != obs, axis=1)))
        if l2 < best[0]:
            best = (l2, float(thr))
    res["ler_best_thr"] = best[0]
    res["best_thr"] = best[1]
    if rates:
        tot = max(sum(sizes), 1)
        res["mean_spike_rate"] = float(sum(r * s for r, s in zip(rates, sizes)) / tot)
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Entrena SNN GAT-LIF sobre circuito BB72 fiel.")
    ap.add_argument("--variant", choices=list(S.VARIANTS), default="uncorrelated")
    ap.add_argument("--p-train", type=float, default=0.003)
    ap.add_argument("--p-train-min", type=float, default=None)
    ap.add_argument("--p-train-max", type=float, default=None)
    ap.add_argument("--p-eval", type=str, default=None,
                    help="lista separada por comas; por defecto = p-train")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-eval", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--balance", dest="balance", action="store_true", default=False,
                    help="genera set de entrenamiento balanceado (50/50 error logico)")
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--lr-patience", type=int, default=2,
                    help="patience de ReduceLROnPlateau (0 = sin scheduler)")
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=30, help="pasos SNN (LIF)")
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--spiking", dest="spiking", action="store_true", default=True)
    ap.add_argument("--no-spiking", dest="spiking", action="store_false")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="0 = sin early-stop; unidades = evaluaciones")
    ap.add_argument("--max-wall-minutes", type=float, default=0.0, help="0 = sin limite")
    ap.add_argument("--target-metric", choices=["ler", "mean_ler"], default="mean_ler")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--ref-shots", type=int, default=1000,
                    help="shots para la referencia BP-OSD (0 = desactivar)")
    ap.add_argument("--ref-osd-order", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.smoke:
        args.n_train, args.n_eval, args.steps = 512, 256, 60
        args.eval_every, args.ckpt_every, args.log_every = 20, 20, 5
        args.num_steps, args.hidden, args.num_layers = 8, 32, 2
        args.ref_shots = 200

    use_range = args.p_train_min is not None and args.p_train_max is not None
    p_fixed = None if use_range else args.p_train
    if args.p_eval:
        p_evals = sorted(float(x) for x in args.p_eval.split(","))
    elif use_range:
        p_evals = [round(args.p_train_max, 6)]
    else:
        p_evals = [args.p_train]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    log(f"variant={args.variant} device={device} spiking={args.spiking}")
    log(f"train: p_fixed={p_fixed} range=({args.p_train_min},{args.p_train_max}) "
        f"n_train={args.n_train} | evals at {p_evals} | steps={args.steps}")

    # --- datos ---
    t0 = time.time()
    feats_tr, labels_tr, ps_tr = sample_train_pool(
        args.n_train, args.variant, args.seed,
        p_fixed=p_fixed, p_min=args.p_train_min, p_max=args.p_train_max,
        balance=args.balance)
    edge_index, n_checks = S.build_variant_graph(args.variant)
    n_edges = int(edge_index.shape[1])
    log(f"train pool listo: feats={feats_tr.shape} labels={labels_tr.shape} "
        f"p_grid={[round(p,4) for p in ps_tr]} ({time.time()-t0:.1f}s)")

    train_ds = build_dataset(feats_tr, labels_tr, edge_index)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # --- sets de evaluacion + referencia BP-OSD (mismos shots) ---
    evals = {}
    for p in p_evals:
        fe, la, an, dets = S.generate_circuit_data(
            args.n_eval, p, args.variant, seed=args.seed + 1000 + int(p * 1e6),
            return_dets=True)
        evals[p] = {"feats": fe, "labels": la, "dets": dets,
                    "ds": build_dataset(fe, la, edge_index), "ref": None}
        log(f"eval p={p}: feats={fe.shape} raw_block_flip={float(an.mean()):.3f}")

        if args.ref_shots > 0:
            try:
                from qecbb import bposd_ref as R

                circuit = S.build_variant_circuit(p, args.variant)
                m = min(args.ref_shots, dets.shape[0])
                t1 = time.time()
                dec_L = R.decoder_for(circuit, osd_order=args.ref_osd_order)
                pred = R.decode_observables(circuit, dets[:m], decoder_and_L=dec_L)
                ref_ler = R.block_ler(pred, la[:m])
                evals[p]["ref"] = {"ler": ref_ler, "shots": m,
                                   "osd_order": args.ref_osd_order}
                log(f"  BP-OSD ref (osd{args.ref_osd_order}) p={p}: LER={ref_ler:.3e} "
                    f"({m} shots, {time.time()-t1:.1f}s)")
            except Exception as exc:  # noqa: BLE001
                log(f"  [AVISO] referencia BP-OSD no disponible p={p}: "
                    f"{type(exc).__name__}: {exc}")
                evals[p]["ref"] = None

    # --- modelo ---
    in_channels = feats_tr.shape[2]
    model = GAT_LIF_BBDecoder(
        in_channels=in_channels, k_logical=12,
        hidden_channels=args.hidden, num_layers=args.num_layers,
        num_steps=args.num_steps, beta=args.beta, spiking=args.spiking,
    ).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    log(f"modelo: in_channels={in_channels} n_checks={n_checks} n_edges={n_edges} "
        f"params={n_param}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", patience=args.lr_patience, factor=args.lr_factor)

    start_step = 0
    best_metric = math.inf
    no_improve = 0
    history = []
    ckpt_last = os.path.join(args.out, f"ckpt_last_{args.variant}.pt")
    ckpt_best = os.path.join(args.out, f"ckpt_best_{args.variant}.pt")

    if args.resume and os.path.exists(args.resume):
        st = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state"])
        if "opt_state" in st:
            opt.load_state_dict(st["opt_state"])
        start_step = int(st.get("step", 0))
        best_metric = float(st.get("best_metric", math.inf))
        log(f"resume desde {args.resume} (step={start_step}, best={best_metric:.3e})")

    def save_ckpt(path, step, metric):
        torch.save({
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "step": step,
            "best_metric": metric,
            "variant": args.variant,
            "in_channels": in_channels,
            "n_checks": n_checks,
            "config": vars(args),
        }, path)

    def run_evals(step):
        model.eval()
        rec = {}
        for p in p_evals:
            e = evals[p]
            r = eval_ler(model, e["ds"], e["labels"], args.eval_batch_size, device)
            ratio = (r["ler"] / e["ref"]["ler"]) if e["ref"] and e["ref"]["ler"] > 0 else None
            rec[p] = {**r, "ref_ler": (e["ref"]["ler"] if e["ref"] else None),
                      "ratio_snn_over_ref": ratio}
        model.train()
        return rec

    log("=== inicio entrenamiento ===")
    t_start = time.time()
    step = start_step
    run_loss, run_n = 0.0, 0
    stopped_reason = None
    it = iter(train_loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        opt.step()
        step += 1

        loss_val = float(loss.detach())
        run_loss += loss_val
        run_n += 1

        if not math.isfinite(loss_val):
            log(f"[AVISO] loss no finita ({loss_val}) en step {step}; abortando.")
            stopped_reason = "nan"
            break

        if step % args.log_every == 0:
            el = time.time() - t_start
            sps = (step - start_step) / max(el, 1e-9)
            eta = (args.steps - step) / max(sps, 1e-9)
            log(f"step {step}/{args.steps} loss={run_loss/run_n:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e} {sps:.1f} step/s "
                f"elapsed={el/60:.1f}min eta={eta/60:.1f}min")
            run_loss, run_n = 0.0, 0

        if step % args.eval_every == 0 or step == args.steps:
            rec = run_evals(step)
            metrics = [rec[p]["ler"] for p in p_evals]
            cur = metrics[0] if args.target_metric == "ler" else float(np.mean(metrics))
            if args.target_metric == "ler":
                log(f"  [eval@{step}] LER p={p_evals[0]}={metrics[0]:.3e} "
                    f"(best={best_metric:.3e})")
            else:
                log(f"  [eval@{step}] mean LER={cur:.3e} best={best_metric:.3e}")
            if scheduler is not None:
                scheduler.step(cur)
            for p in p_evals:
                r = rec[p]
                ref = r["ref_ler"]
                refs = f" ref={ref:.3e} ratio={r['ratio_snn_over_ref']:.2f}x" if ref else ""
                spk = r.get("mean_spike_rate")
                spks = f" spike={spk:.3f}" if spk is not None else ""
                log(f"      p={p}: LER={r['ler']:.3e} "
                    f"CI=[{r['ci_lo']:.2e},{r['ci_hi']:.2e}]{refs}{spks} "
                    f"ler@oracle_thr={r.get('ler_best_thr', float('nan')):.3e}"
                    f"(t*={r.get('best_thr', 0.0):.2f})")
            history.append({"step": step, "elapsed_s": time.time() - t_start,
                            "metrics": rec, "target": cur})

            if cur < best_metric - 1e-12:
                best_metric = cur
                no_improve = 0
                save_ckpt(ckpt_best, step, best_metric)
                log(f"      -> nuevo mejor ({best_metric:.3e}); guardado {ckpt_best}")
            else:
                no_improve += 1
                log(f"      -> sin mejora ({no_improve}/{args.early_stop_patience})")

            if args.early_stop_patience and no_improve >= args.early_stop_patience:
                log(f"[early-stop] sin mejora por {no_improve} evals; parando en step {step}")
                stopped_reason = "early_stop"
                break

        if step % args.ckpt_every == 0:
            save_ckpt(ckpt_last, step, best_metric)

        if args.max_wall_minutes and (time.time() - t_start) / 60.0 >= args.max_wall_minutes:
            log(f"[tiempo] limite de {args.max_wall_minutes} min alcanzado en step {step}")
            stopped_reason = "time"
            break

    save_ckpt(ckpt_last, step, best_metric)
    total_min = (time.time() - t_start) / 60.0
    log(f"=== fin: step={step} best_{args.target_metric}={best_metric:.3e} "
        f"({total_min:.1f} min) razon={stopped_reason or 'completado'} ===")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": device, "variant": args.variant, "n_checks": n_checks,
        "n_edges": n_edges, "n_params": n_param, "config": vars(args),
        "p_evals": p_evals, "p_train_grid": [float(p) for p in ps_tr],
        "ref": {str(p): evals[p]["ref"] for p in p_evals},
        "best_metric": best_metric, "target_metric": args.target_metric,
        "stopped_reason": stopped_reason, "final_step": step,
        "elapsed_min": total_min, "history": history,
    }
    jpath = os.path.join(args.out, f"train_log_{args.variant}_{p_fixed or 'range'}.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log(f"JSON -> {jpath}")
    log(f"checkpoints -> {ckpt_best} , {ckpt_last}")


if __name__ == "__main__":
    main()
