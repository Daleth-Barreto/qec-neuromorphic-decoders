#!/usr/bin/env python3
"""C1: quantization-aware fine-tuning (QAT) con STE + activacion event-driven.

Punto de partida: transformer SOTA (NTU/Blue) pretrained sobre BB72 fiel.
El estudio post-training (ntu_neuro.py) mostro que el swap GELU/SiLU->ReLU y
la cuantizacion de activaciones de 4 bits colapsan la precision. Aqui
reentrenamos con:

  * Straight-Through Estimator (STE) en la cuantizacion de pesos y
    activaciones, para que el gradiente fluya a traves del redondeo.
  * Activacion objetivo (ReLU o LIF-surrogate) inyectada durante el training,
    de modo que el modelo aprenda con la no linealidad de hardware.
  * Gating opcional de sparsity de activaciones (proxy event-driven).

Objetivo: medir si el reentrenamiento recupera la precision que el
post-training pierde.

Uso:
  python experiments/ntu_qat.py --p 0.003 --act relu --w-bits 4 --a-bits 8 \
      --steps 400 --out bb_code_decoder_results/ntu_qat/qat_p0.003_w4a8_relu.json
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

NTU_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "third_party", "ntu")
NTU_DIR = os.path.abspath(NTU_DIR)

NTU_LM = dict(l=6, m=6, A_x_pows=[3], A_y_pows=[1, 2],
              B_x_pows=[1, 2], B_y_pows=[3], rounds=6)


def load_module():
    sys.path.insert(0, NTU_DIR)
    spec = importlib.util.spec_from_file_location(
        "ntu_transformer", os.path.join(NTU_DIR, "transformer.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ntu_transformer"] = mod
    spec.loader.exec_module(mod)

    def _noworker_dl(*a, **kw):
        kw["num_workers"] = 0
        kw["pin_memory"] = False
        return torch.utils.data.DataLoader(*a, **kw)

    mod.DataLoader = _noworker_dl
    return mod


# ---------------- STE quantization ----------------
def ste(x, xq):
    """Forward usa xq; backward pasa el gradiente como si fuera x."""
    return x + (xq - x).detach()


def quant_w(w, bits, ternary=False):
    if ternary:
        scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        return (w / scale).round().clamp(-1, 1) * scale
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return (w / scale).round().clamp(-qmax, qmax) * scale


def quant_a(x, bits, ternary=False):
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


class QuantLinear(nn.Module):
    """nn.Linear con cuantizacion STE del peso (per-output-channel)."""

    def __init__(self, lin, w_bits, ternary_w):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.w_bits = w_bits
        self.ternary_w = ternary_w

    def forward(self, x):
        w = self.weight
        if self.ternary_w or self.w_bits is not None:
            w = ste(w, quant_w(w, self.w_bits, self.ternary_w))
        return F.linear(x, w, self.bias)


def replace_linears(root, w_bits, ternary_w):
    count = 0
    for name, module in list(root.named_children()):
        if isinstance(module, nn.Linear):
            setattr(root, name, QuantLinear(module, w_bits, ternary_w))
            count += 1
        else:
            count += replace_linears(module, w_bits, ternary_w)
    return count


class ActCtx:
    """Parchea F.gelu / F.silu con la activacion objetivo (quant + gating)."""

    def __init__(self, kind, a_bits=None, ternary_a=False, thr=0.0,
                 sparsity_thr=None):
        self.kind = kind
        self.a_bits = a_bits
        self.ternary_a = ternary_a
        self.thr = thr
        self.sparsity_thr = sparsity_thr
        self.total = 0
        self.active = 0

    def _post(self, y):
        if self.sparsity_thr is not None:
            mask = (y.abs() > self.sparsity_thr).to(y.dtype)
            self.total += y.numel()
            self.active += int(mask.sum())
            y = y * mask
        if self.a_bits is not None:
            y = ste(y, quant_a(y, self.a_bits, self.ternary_a))
        return y

    def __enter__(self):
        self._gelu, self._silu = F.gelu, F.silu
        kind, thr = self.kind, self.thr
        og, os_ = self._gelu, self._silu

        def gelu(x, *a, **k):
            if kind == "relu":
                y = F.relu(x)
            elif kind == "lif":
                y = x * torch.sigmoid((x - thr) / 1.0)
            elif kind == "silu":
                y = os_(x, *a, **k)
            else:
                y = og(x, *a, **k)
            return self._post(y)

        def silu(x, *a, **k):
            if kind == "relu":
                y = F.relu(x)
            elif kind == "lif":
                y = x * torch.sigmoid((x - thr) / 1.0)
            elif kind == "gelu":
                y = os_(x, *a, **k)
            else:
                y = os_(x, *a, **k)
            return self._post(y)

        F.gelu = gelu
        F.silu = silu
        return self

    def __exit__(self, *e):
        F.gelu, F.silu = self._gelu, self._silu
        return False

    def act_fraction(self):
        return (self.active / self.total) if self.total else None


# ---------------- model + data ----------------
def build_model(mod, p, device="cpu"):
    mapper = mod.BBMapper(NTU_LM["l"], NTU_LM["m"], NTU_LM["A_x_pows"],
                          NTU_LM["A_y_pows"], NTU_LM["B_x_pows"],
                          NTU_LM["B_y_pows"], rounds=NTU_LM["rounds"], p=p)
    lb = mod.build_default_observables(mapper)
    K = mapper.code.K
    model = mod.AlphaQubitV2_BB(
        mapper, n_logicals=K, d_model=512, n_heads=8,
        rope_delta_mode="signed_neg_half",
        logical_representatives=mod.build_anchor_representatives(
            mapper, lb, "representative"))
    ckpt_path = mod.download_from_hf(
        repo_id="Dreamworldsmile/ntu-surface-code-decoder",
        filename="bb/bb72_transformer.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state", ckpt)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in sd.items()}
    msd = model.state_dict()
    filt = {k: v for k, v in sd.items()
            if k in msd and tuple(msd[k].shape) == tuple(v.shape)
            and k != "logical_readout_bias"}
    model.load_state_dict(filt, strict=False)
    model.to(device)
    return model, lb, K


def make_dataset(mod, lb, p, batch):
    ds = mod.OnlineBBDataset(NTU_LM["l"], NTU_LM["m"], NTU_LM["A_x_pows"],
                             NTU_LM["A_y_pows"], NTU_LM["B_x_pows"],
                             NTU_LM["B_y_pows"], rounds=NTU_LM["rounds"],
                             p=p, batch_size=batch)
    ds.set_logical_basis(lb)
    return ds


def build_eval_set(mod, lb, p, shots, seed, batch=512, device="cpu"):
    ds = make_dataset(mod, lb, p, batch)
    xs, ys, got = [], [], 0
    torch.manual_seed(seed)
    for x, y in iter(mod.DataLoader(ds, batch_size=None, num_workers=0)):
        xs.append(x)
        ys.append(y)
        got += x.shape[0]
        if got >= shots:
            break
    X = torch.cat(xs)[:shots].to(device)
    Y = torch.cat(ys)[:shots].to(device)
    return X, Y


@torch.no_grad()
def eval_ler(model, X, Y, batch=512):
    was = model.training
    model.eval()
    n = X.shape[0]
    correct = 0.0
    per_log = None
    for i in range(0, n, batch):
        logits = model(X[i:i + batch]).float()
        pred = (logits > 0).float()
        match = (pred == Y[i:i + batch]).float()
        correct += float(match.all(dim=1).sum())
        per_log = match.sum(0) if per_log is None else per_log + match.sum(0)
    if was:
        model.train()
    return 1.0 - correct / n, (per_log / n).tolist()


def cfg_ctx(args):
    return ActCtx(args.act, args.a_bits, args.ternary_w, args.lif_thr,
                  args.sparsity_thr)


def train(args):
    dev = args.device
    t0 = time.time()
    mod = load_module()
    model, lb, K = build_model(mod, args.p, dev)
    n_lin = replace_linears(model, args.w_bits, args.ternary_w)
    print("[%.1fs] model ready: replaced %d Linear, K=%d"
          % (time.time() - t0, n_lin, K), flush=True)

    X, Y = build_eval_set(mod, lb, args.eval_p, args.eval_shots,
                          args.seed + 7, device=dev)
    print("[%.1fs] eval set %s" % (time.time() - t0, tuple(X.shape)), flush=True)

    with ActCtx(args.act, args.a_bits, args.ternary_w, args.lif_thr,
                args.sparsity_thr) as c:
        base_ler, base_pl = eval_ler(model, X, Y)
    print("[%.1f] baseline (target act/quant, no training) LER=%.4e act=%s"
          % (time.time() - t0, base_ler, c.act_fraction()), flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    base_lr = args.lr
    min_lr = args.lr_min if args.lr_min is not None else args.lr * 0.02
    warmup = max(0, args.warmup)

    def lr_at(step):
        if args.lr_schedule != "cosine":
            return base_lr
        if warmup > 0 and step <= warmup:
            return base_lr * step / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        t = min(1.0, max(0.0, t))
        import math as _m
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + _m.cos(_m.pi * t))

    dataset = make_dataset(mod, lb, args.p, args.batch)
    it = iter(mod.DataLoader(dataset, batch_size=None, num_workers=0))

    history = []
    model.train()
    for step in range(1, args.steps + 1):
        cur_lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(mod.DataLoader(
                make_dataset(mod, lb, args.p, args.batch),
                batch_size=None, num_workers=0))
            x, y = next(it)
        x = x.to(args.device)
        y = y.to(args.device).float()
        with ActCtx(args.act, args.a_bits, args.ternary_w, args.lif_thr,
                    args.sparsity_thr):
            logits = model(x).float()
            loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            print("[%.1f] step %d loss=%.4f lr=%.2e"
                  % (time.time() - t0, step, float(loss), cur_lr), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            with ActCtx(args.act, args.a_bits, args.ternary_w, args.lif_thr,
                        args.sparsity_thr) as c:
                ler, pl = eval_ler(model, X, Y)
            row = dict(step=step, ler=ler, per_log_mean=sum(pl) / len(pl),
                       per_log_worst=min(pl), act_fraction=c.act_fraction())
            history.append(row)
            print("[%.1f] step %d LER=%.4e mean=%.4f worst=%.4f act=%s"
                  % (time.time() - t0, step, ler, row["per_log_mean"],
                     row["per_log_worst"], row["act_fraction"]), flush=True)

    best = min(history, key=lambda h: h["ler"]) if history else None
    result = dict(
        p=args.p, eval_p=args.eval_p, shots=args.eval_shots, steps=args.steps,
        w_bits=args.w_bits, a_bits=args.a_bits, ternary_w=args.ternary_w,
        act=args.act, lif_thr=args.lif_thr, sparsity_thr=args.sparsity_thr,
        lr=args.lr, lr_min=min_lr, lr_schedule=args.lr_schedule,
        warmup=warmup, batch=args.batch,
        baseline_ler=base_ler,
        baseline_per_log_mean=sum(base_pl) / len(base_pl),
        best_ler=best["ler"] if best else None,
        best_step=best["step"] if best else None,
        final_ler=history[-1]["ler"] if history else None,
        final_act_fraction=history[-1]["act_fraction"] if history else None,
        history=history,
        note="C1 QAT/STE fine-tune of pretrained BB transformer; faithful BB72",
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("-> %s" % args.out, flush=True)

    if args.ckpt:
        torch.save({"model_state": model.state_dict(), "args": vars(args)},
                   args.ckpt)
        print("-> %s" % args.ckpt, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--eval-p", type=float, default=None)
    ap.add_argument("--eval-shots", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", default="none", choices=["none", "cosine"])
    ap.add_argument("--lr-min", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--w-bits", type=int, default=None)
    ap.add_argument("--a-bits", type=int, default=None)
    ap.add_argument("--ternary-w", action="store_true")
    ap.add_argument("--act", default="relu",
                    choices=["gelu", "silu", "relu", "lif"])
    ap.add_argument("--lif-thr", type=float, default=0.0)
    ap.add_argument("--sparsity-thr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="bb_code_decoder_results/ntu_qat/out.json")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    if args.eval_p is None:
        args.eval_p = args.p
    train(args)


if __name__ == "__main__":
    main()
