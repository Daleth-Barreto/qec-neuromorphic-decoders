#!/usr/bin/env python3
"""T3: entrenamiento corto del SNN en GPU (Kaggle) sobre el circuito fiel BB72.

Envoltura delgada sobre ``experiments/train_snn_circuit.py`` que:
  1. detecta raices (dataset montado en /kaggle/input o repo local),
  2. corre SIEMPRE un smoke test corto primero (falla rapido si algo rompe),
  3. lanza el run principal en GPU con checkpoints/resume/early-stop,
  4. deja los checkpoints y el JSON en el directorio de salida (persistente).

Ejemplo (notebook Kaggle, GPU):
    !python experiments/kaggle_short_train.py --variant uncorrelated \
        --p-train 0.003 --steps 8000 --batch-size 128 --n-train 150000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def find_root() -> str:
    """Devuelve la raiz del repo (donde vive src/qecbb)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here), "/kaggle/working", os.getcwd()):
        if os.path.isdir(os.path.join(base, "src", "qecbb")):
            return os.path.dirname(base) if os.path.basename(base) == "experiments" else base
    # fallback: buscar config.py bajo /kaggle/input
    if os.path.isdir("/kaggle/input"):
        for dirpath, _dirs, files in os.walk("/kaggle/input"):
            if "config.py" in files and os.path.basename(dirpath) == "qecbb":
                return os.path.dirname(os.path.dirname(dirpath))
    return os.path.dirname(here)


def run(cmd: list[str], label: str):
    print(f"\n### {label}\n$ {' '.join(cmd)}", flush=True)
    t = time.time()
    rc = subprocess.call(cmd)
    print(f"### {label} -> rc={rc} ({time.time()-t:.0f}s)", flush=True)
    if rc != 0:
        raise SystemExit(f"{label} fallo (rc={rc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="uncorrelated")
    ap.add_argument("--p-train", default="0.003")
    ap.add_argument("--p-train-min", default=None)
    ap.add_argument("--p-train-max", default=None)
    ap.add_argument("--p-eval", default="")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=150000)
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--lr", default="5e-4")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--max-wall-minutes", type=float, default=0.0)
    ap.add_argument("--ref-shots", type=int, default=500,
                    help="shots BP-OSD de referencia (0 = desactivar). "
                         "En correlated puede ser ~1 s/shot; usar pocos.")
    ap.add_argument("--ref-osd-order", type=int, default=3)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="/kaggle/working/snn_circuit")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()

    root = find_root()
    trainer = os.path.join(root, "experiments", "train_snn_circuit.py")
    if not os.path.exists(trainer):
        raise SystemExit(f"no encuentro {trainer}")
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(out, exist_ok=True)
    print(f"root={root}\ntrainer={trainer}\nout={out}\ntorch={_torch_info()}", flush=True)

    common = ["--variant", args.variant, "--p-train", str(args.p_train),
              "--n-eval", str(args.n_eval), "--batch-size", str(args.batch_size),
              "--hidden", str(args.hidden), "--num-layers", str(args.num_layers),
              "--num-steps", str(args.num_steps), "--lr", str(args.lr),
              "--eval-every", str(args.eval_every), "--ckpt-every", str(args.ckpt_every),
              "--early-stop-patience", str(args.early_stop_patience),
              "--max-wall-minutes", str(args.max_wall_minutes),
              "--ref-shots", str(args.ref_shots), "--ref-osd-order", str(args.ref_osd_order),
              "--device", args.device, "--out", out]
    if args.p_eval:
        common += ["--p-eval", args.p_eval]
    if args.p_train_min is not None and args.p_train_max is not None:
        common += ["--p-train-min", str(args.p_train_min),
                   "--p-train-max", str(args.p_train_max)]

    py = sys.executable

    if not args.no_smoke:
        run([py, trainer, "--smoke", "--variant", args.variant, "--out", out],
            "smoke (valida pipeline en GPU)")

    main_cmd = [py, trainer, "--n-train", str(args.n_train), "--steps", str(args.steps)] + common
    if args.resume:
        main_cmd += ["--resume", args.resume]
    run(main_cmd, "entrenamiento principal (T3)")

    print("\nListo. Checkpoints y JSON en:", out, flush=True)


def _torch_info() -> str:
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
        return f"{torch.__version__} device={dev} ({name})"
    except Exception as exc:  # noqa: BLE001
        return f"torch no disponible: {exc}"


if __name__ == "__main__":
    main()
