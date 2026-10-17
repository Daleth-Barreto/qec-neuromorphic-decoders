#!/usr/bin/env python3
"""Neuromorphic conversion study del transformer SOTA (NTU/Blue) sobre BB72 fiel.

El transformer usa GELU/SiLU/softmax (no ReLU), asi que la conversion clasica
ReLU->LIF no aplica. Estudiamos el eje de eficiencia neuromorfica mediante:

  1. Cuantizacion post-entrenamiento (fake-quant) de pesos y activaciones:
     W8A8, W4A8, W4A4, ternario.
  2. Computo event-driven: umbral de activacion (sparsity proxy) + medicion de
     actividad, para estimar el conteo de operaciones sinapticas (SOPs).

Todo sin re-entrenar. Salida JSON con LER por configuracion.

Uso:
  python experiments/ntu_neuro.py --p 0.003 --shots 2000 --out <json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

NTU_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "third_party", "ntu")
NTU_DIR = os.path.abspath(NTU_DIR)

_RealDL = tud.DataLoader


def _noworker_dl(*a, **kw):
    kw["num_workers"] = 0
    kw["pin_memory"] = False
    return _RealDL(*a, **kw)


def load_module():
    sys.path.insert(0, NTU_DIR)
    os.chdir(NTU_DIR)
    spec = importlib.util.spec_from_file_location(
        "ntu_transformer", os.path.join(NTU_DIR, "transformer.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ntu_transformer"] = mod
    spec.loader.exec_module(mod)
    mod.DataLoader = _noworker_dl
    return mod


# ---------------- quantization helpers ----
class Stats:
    def __init__(self):
        self.total = 0
        self.active = 0
        self.near0 = 0

    def add(self, x, thr):
        a = x.detach().float()
        n = a.numel()
        self.total += n
        self.active += int((a.abs() > thr).sum())
        self.near0 += int((a.abs() < 0.05).sum())

    def as_dict(self, thr):
        t = max(self.total, 1)
        return dict(thr=thr, act_fraction=self.active / t,
                    near_zero_fraction=self.near0 / t)


class NeuroCtx:
    """Fake-quant + sparsity de activaciones via patch de F.gelu / F.silu."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = Stats()
        self._orig_gelu = F.gelu
        self._orig_silu = F.silu

    def _act(self, x):
        c = self.cfg
        if c.get("sparsity_thr") is not None:
            self.stats.add(x, c["sparsity_thr"])
            a = x * (x.abs() > c["sparsity_thr"]).to(x.dtype)
        else:
            a = x
        if c.get("a_bits") is not None:
            a = fake_quant_act(a, c["a_bits"], c.get("ternary_act", False))
        return a

    def __enter__(self):
        relu = self.cfg.get("act_relu", False)
        orig_gelu, orig_silu = self._orig_gelu, self._orig_silu

        def gelu(x, *a, **k):
            return self._act(F.relu(x) if relu else orig_gelu(x, *a, **k))

        def silu(x, *a, **k):
            return self._act(F.relu(x) if relu else orig_silu(x, *a, **k))

        F.gelu = gelu
        F.silu = silu
        return self

    def __exit__(self, *e):
        F.gelu = self._orig_gelu
        F.silu = self._orig_silu
        return False


def fake_quant_act(x, bits, ternary=False):
    if ternary:
        scale = x.abs().mean().clamp_min(1e-8)
        return (x / scale).round().clamp(-1, 1) * scale
    qmax = 2 ** (bits - 1) - 1
    if x.dim() >= 2:
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        amax = x.abs().amax().clamp_min(1e-8)
    scale = amax / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def quantize_weights(model, w_bits, ternary_w=False):
    saved = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            saved.append((m, m.weight.data.clone()))
            if ternary_w:
                w = m.weight.data
                scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
                m.weight.data = (w / scale).round().clamp(-1, 1) * scale
            elif w_bits is not None:
                w = m.weight.data
                qmax = 2 ** (w_bits - 1) - 1
                scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
                m.weight.data = (w / scale).round().clamp(-qmax, qmax) * scale
    return saved


def restore_weights(saved):
    for m, w in saved:
        m.weight.data = w


def build_eval_set(mod, p, shots, batch=256, seed=1234, device="cpu"):
    mapper = mod.BBMapper(6, 6, [3], [1, 2], [1, 2], [3], rounds=6, p=p)
    lb = mod.build_default_observables(mapper)
    n_logicals = mapper.code.K
    model = mod.AlphaQubitV2_BB(
        mapper, n_logicals=n_logicals, d_model=512, n_heads=8,
        rope_delta_mode="signed_neg_half",
        logical_representatives=mod.build_anchor_representatives(
            mapper, lb, "representative"))
    ckpt_path = mod.download_from_hf(
        repo_id="Dreamworldsmile/ntu-surface-code-decoder",
        filename="bb/bb72_transformer.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state", ckpt)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    msd = model.state_dict()
    filt = {k: v for k, v in sd.items()
            if k in msd and tuple(msd[k].shape) == tuple(v.shape)
            and k != "logical_readout_bias"}
    model.load_state_dict(filt, strict=False)
    model.to(device).eval()

    ds = mod.OnlineBBDataset(l=6, m=6, A_x_pows=[3], A_y_pows=[1, 2],
                             B_x_pows=[1, 2], B_y_pows=[3], rounds=6, p=p,
                             batch_size=batch)
    ds.set_logical_basis(lb)
    xs, ys = [], []
    got = 0
    torch.manual_seed(seed)
    for x, y in iter(_RealDL(ds, batch_size=None, num_workers=0)):
        xs.append(x)
        ys.append(y)
        got += x.shape[0]
        if got >= shots:
            break
    X = torch.cat(xs)[:shots].to(device)
    Y = torch.cat(ys)[:shots].to(device)
    return model, X, Y, n_logicals


@torch.no_grad()
def eval_ler(model, X, Y, device="cpu", batch=256):
    n = X.shape[0]
    correct = 0.0
    per_log = None
    for i in range(0, n, batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        logits = model(xb).float()
        pred = (logits > 0).float()
        match = (pred == yb).float()
        correct += float(match.all(dim=1).sum())
        per_log = match.sum(0) if per_log is None else per_log + match.sum(0)
    ler = 1.0 - correct / n
    return ler, (per_log / n).tolist()


DEFAULT_CONFIGS = [
    {"name": "fp32", "w_bits": None, "a_bits": None},
    {"name": "w8a16", "w_bits": 8},
    {"name": "w4a16", "w_bits": 4},
    {"name": "w2a16", "w_bits": 2},
    {"name": "ternary_w", "w_bits": None, "ternary_w": True},
    {"name": "w8a8", "w_bits": 8, "a_bits": 8},
    {"name": "w4a8", "w_bits": 4, "a_bits": 8},
    {"name": "w4a4", "w_bits": 4, "a_bits": 4},
    {"name": "w2a4", "w_bits": 2, "a_bits": 4},
    {"name": "ternary_w_a4", "w_bits": None, "ternary_w": True, "a_bits": 4},
    {"name": "sparse_a_thr0.05", "sparsity_thr": 0.05},
    {"name": "sparse_a_thr0.1", "sparsity_thr": 0.1},
    {"name": "sparse_a_thr0.15", "sparsity_thr": 0.15},
    {"name": "sparse_a_thr0.2", "sparsity_thr": 0.2},
    {"name": "sparse_a_thr0.25", "sparsity_thr": 0.25},
    {"name": "sparse_a_thr0.3", "sparsity_thr": 0.3},
    {"name": "act_relu", "act_relu": True},
    {"name": "act_relu_w8a8", "act_relu": True, "w_bits": 8, "a_bits": 8},
    {"name": "act_relu_w4a8", "act_relu": True, "w_bits": 4, "a_bits": 8},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="bb_code_decoder_results/ntu_ref/ntu_neuro.json")
    ap.add_argument("--only", default=None, help="nombres separados por coma")
    args = ap.parse_args()

    t0 = time.time()
    mod = load_module()
    model, X, Y, K = build_eval_set(mod, args.p, args.shots, args.batch,
                                    device=args.device)
    print(f"[{time.time()-t0:.1f}s] model+eval set ready: X={tuple(X.shape)} K={K}",
          flush=True)

    configs = DEFAULT_CONFIGS
    if args.only:
        want = set(args.only.split(","))
        configs = [c for c in configs if c["name"] in want]

    results = []
    for cfg in configs:
        t = time.time()
        ctx = NeuroCtx(cfg)
        saved = quantize_weights(model, cfg.get("w_bits"),
                                 cfg.get("ternary_w", False))
        try:
            with ctx:
                ler, per_log = eval_ler(model, X, Y, device=args.device,
                                        batch=args.batch)
        finally:
            restore_weights(saved)
        row = dict(cfg)
        row["ler"] = ler
        row["per_log_mean"] = sum(per_log) / len(per_log)
        row["per_log_worst"] = min(per_log)
        if cfg.get("sparsity_thr") is not None:
            row.update(ctx.stats.as_dict(cfg["sparsity_thr"]))
        results.append(row)
        print(f"[{time.time()-t:.1f}s] {cfg['name']:16s} LER={ler:.4e} "
              f"mean={row['per_log_mean']:.4f} worst={row['per_log_worst']:.4f}"
              + (f" act={row.get('act_fraction'):.3f}" if 'act_fraction' in row else ""),
              flush=True)

    out = dict(p=args.p, shots=args.shots, source="ntu_neuro (pretrained transformer)",
               note="fake-quant PTQ + event-driven sparsity proxy; faithful BB72 correlated",
               results=results)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
