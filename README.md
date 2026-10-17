# Hardware-Aware Compression of a Transformer Decoder for Bivariate Bicycle Codes

Code and results for the paper accepted at **ICEIC 2027**:

> "Hardware-Aware Compression of a Transformer Decoder for Bivariate Bicycle
> Codes" (ICEIC 2027, Fukuoka, Japan).

We study the **compression limits** — low-bit quantization and activation
sparsity — of a pre-trained attention-based transformer decoder for the
`[[72,12,6]]` bivariate-bicycle (BB) quantum error-correcting code, in a
**post-training** setting, and show that a short **quantization-aware training
(QAT)** pass recovers configurations that collapse under naive post-training
conversion. A naive GELU/SiLU → ReLU replacement (a first step toward spiking
hardware) degrades accuracy; QAT restores it by more than an order of
magnitude.

## Contents

| Path | What it is |
|------|-----------|
| `src/qecbb/` | Reusable package: BB codes, graphs, data generation (`stim`), GAT-LIF model, training, evaluation, FPGA export |
| `experiments/` | Runnable pipelines: `ntu_neuro.py` (post-training), `ntu_qat.py` (QAT/STE), analyzers, SNN-circuit training |
| `third_party/ntu/` | Vendored, unmodified NTU baseline (MIT) — see [`THIRD_PARTY.md`](THIRD_PARTY.md) |
| `bb_code_decoder_results/` | Result artifacts: JSON (tracked), QAT/SNN checkpoints (`.pt` via Git LFS, `.pth`), FPGA exports |
| `paper_iceic/` | LaTeX source (`main.tex`), IEEEtran class, figures |
| `scripts/`, `tests/` | Environment setup + smoke tests |

## Reproducing the results

See **[`REPRODUCE.md`](REPRODUCE.md)** for the exact commands to regenerate the
figures and tables (environment, checkpoint download + SHA256, `stim` data
generation, `ntu_ref` / `ntu_neuro` / `ntu_qat` runs).

Quick start:

```bash
conda create -n qecbb python=3.10 -y && conda activate qecbb
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"   # Windows PowerShell: $env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
```

The pre-trained baseline transformer is downloaded automatically from Hugging
Face on first run; it is **third-party** (MIT) and not redistributed here —
see [`THIRD_PARTY.md`](THIRD_PARTY.md) for attribution and citation.

## Provenance and licensing

- The baseline transformer (AlphaQubitV2_BB) and its pre-trained weights are
  **not** authored by us. See [`THIRD_PARTY.md`](THIRD_PARTY.md).
- Our contribution: the compression study + QAT. The `ntu_qat_*.pt` checkpoints
  are derivative fine-tunes of the MIT-licensed base, redistributed under the
  same terms.

## Roadmap (next publications)

This ICEIC paper is the first step of a hardware-aware compression line:

1. **IEEE QCNC (6–8 months)** — *Event-driven spiking decoder.* Extend
   W4A8 → ReLU → discrete LIF + robust QAT (600–1000 steps). Natural
   continuation of ICEIC, reusing the existing pipeline.
2. **IEEE NANO (3–4 months after QCNC)** — *Compute-in-Memory robustness.* Map
   the trained W4A8 + sparse LIF decoder to crossbars; model conductance
   variation / ADC / line non-idealities and connect device noise → LER. Builds
   on the paper's memory-bound conclusion.
3. **QCCL (long-term)** — *Generative / diffusion decoders* (higher risk;
   deferred until the first two are consolidated).

The through-line: **compression → event-driven → compute-in-memory**, a
coherent hardware-aware QEC decoding research program.
