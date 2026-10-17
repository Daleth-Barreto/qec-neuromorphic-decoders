# Reproducing the ICEIC 2027 results

This document gives the exact steps to regenerate the figures and tables of the
paper *"Hardware-Aware Compression of a Transformer Decoder for Bivariate
Bicycle Codes"* (ICEIC 2027). All experiments are seeded; data is generated
on-the-fly with `stim` (no committed dataset is required).

Reported logical error rates (LER) use **5000 shots** per noise level for the
post-training sweeps and **3000-shot** validation for QAT, with block-level LER
over `K=12` logical qubits and 95% Wilson intervals.

---

## 1. Environment

```bash
# (optional) create a fresh env
conda create -n qecbb python=3.10 -y
conda activate qecbb

pip install -r requirements.txt
```

`stim` and `huggingface_hub` are required and are now included in
`requirements.txt`. Set `PYTHONPATH` to the vendored source tree:

```bash
# Linux / macOS
export PYTHONPATH="$PWD/src:$PYTHONPATH"
# Windows (PowerShell)
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
```

## 2. Pre-trained baseline checkpoint

The NTU `AlphaQubitV2_BB` transformer is **not** redistributed here. It is
downloaded automatically from the Hugging Face Hub on first run
(`experiments/ntu_qat.py`, `ntu_neuro.py` call `hf_hub_download`). See
`THIRD_PARTY.md` for attribution and license.

- Repo: `Dreamworldsmile/ntu-surface-code-decoder`
- File: `bb/bb72_transformer.pt` (~144 MB)
- Expected SHA256: `77cf7899c6d14acaaae53a8334e69d00011a38a92c0b73cf7709119c0916a201`

To verify a cached download:

```bash
python - <<'PY'
import hashlib, huggingface_hub, pathlib
p = pathlib.Path(huggingface_hub.hf_hub_download(
    repo_id="Dreamworldsmile/ntu-surface-code-decoder",
    filename="bb/bb72_transformer.pt"))
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
```

## 3. Reference (full-precision) decoding — `ntu_ref`

Reproduces the baseline LER of the transformer and BP-OSD at physical error
rates `p`. Results land in `bb_code_decoder_results/ntu_ref/`.

```bash
python experiments/ntu_neuro.py --p 0.003 --shots 5000 \
  --out bb_code_decoder_results/ntu_ref/ntu_neuro.json
```

Aggregate / summarize (produces `ntu_ref_summary.json`):

```bash
python experiments/analyze_ntu_ref.py
```

## 4. Post-training quantization + activation study — `ntu_neuro`

Sweeps weight/activation bit-widths and activation swaps (GELU→ReLU/LIF) with
**no retraining**.

```bash
python experiments/ntu_neuro.py --p 0.003 --shots 5000 \
  --out bb_code_decoder_results/ntu_neuro/ntu_neuro.json
python experiments/analyze_ntu_neuro.py
```

## 5. Quantization-aware training (QAT) — `ntu_qat`

Fine-tunes the baseline with straight-through estimator quantization and
ReLU/LIF activations. Each config writes its own JSON; the analyzer then builds
the summary table and figure.

Reproduce the two headline configurations (matching the stored artifacts):

```bash
# ReLU, full precision, sparsity threshold 0.1
python experiments/ntu_qat.py --p 0.003 --steps 300 --batch 64 --lr 1e-4 \
  --act relu --seed 1234 \
  --out bb_code_decoder_results/ntu_qat/ntu_qat_relu.json

# ReLU + W4A8 + sparsity threshold 0.1
python experiments/ntu_qat.py --p 0.003 --steps 300 --batch 32 --lr 1e-4 \
  --act relu --w-bits 4 --a-bits 8 --sparsity-thr 0.1 --seed 1234 \
  --out bb_code_decoder_results/ntu_qat/ntu_qat_relu_w4a8_s01.json
```

Other recorded configs live under `bb_code_decoder_results/ntu_qat/`; each JSON
is self-contained (records `p`, `steps`, `w_bits`, `a_bits`, `act`, `lif_thr`,
`sparsity_thr`, `lr`, `batch`, `shots`, `seed`).

Consolidate into the paper table + figure (`ntu_qat_summary.json`,
`ntu_qat_summary.png`):

```bash
python experiments/analyze_ntu_qat.py
```

### Key hyperparameters (as reported)

| Config | p | steps | W/A | act | sparsity τ | lr | batch | shots | seed |
|---|---|---|---|---|---|---|---|---|---|
| `relu` | 0.003 | 300 | FP32 | relu | — | 1e-4 | 64 | 3000 | 1234 |
| `relu_w4a8_s01` | 0.003 | 300 | 4/8 | relu | 0.1 | 1e-4 | 32 | 3000 | 1234 |

## 6. Artifacts tracked in this repo

- Per-run and summary results: `bb_code_decoder_results/**/*.json` (small,
  tracked). These already contain the published numbers and self-describing
  hyper-parameters.
- Fine-tuned QAT checkpoints: `bb_code_decoder_results/ntu_qat/*.pt`
  (~137 MB each, tracked via **Git LFS**).
- SNN GAT-LIF checkpoints: `bb_code_decoder_results/**/*.pth` (small, tracked).
- FPGA export sources: `bb_code_decoder_results/**/*.v`, `*.mem`.

Regenerable / not tracked: `*.npz` (cached datasets), `__pycache__`,
`KAGGLE_TOKENS.md`, and the `kaggle/` orchestration directory.
