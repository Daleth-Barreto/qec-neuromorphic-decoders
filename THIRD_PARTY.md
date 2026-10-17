# Third-party components

This repository builds on third-party code and a pre-trained model. All are
used under their respective licenses and cited below. None of these are
authored by us; see the upstream sources for canonical versions.

## NTU / AlphaQubitV2_BB transformer decoder (baseline)

The pre-trained transformer decoder used as the fixed SOTA baseline for
bivariate-bicycle (BB) [[72,12,6]] decoding is **not** authored by us.

- **Implementation:** vendored, unmodified, under `third_party/ntu/`
  - Upstream repo: <https://github.com/GrahamYan/ntu-decoder>
  - Commit: `d931a3322102bb9469cd5d97bd6b04f11f0085c7`
  - License: MIT
  - Files: `baseline.py` (BP-OSD / Relay-BP), `transformer.py`
    (AlphaQubitV2_BB + eval harness)
- **Pre-trained checkpoint:** downloaded at runtime from the Hugging Face Hub
  - Repo: `Dreamworldsmile/ntu-surface-code-decoder`
  - File: `bb/bb72_transformer.pt` (~144 MB)
  - SHA256: `77cf7899c6d14acaaae53a8334e69d00011a38a92c0b73cf7709119c0916a201`
- **Citation:**
  > J. Blue et al., "Machine learning decoding of circuit-level noise for
  > bivariate bicycle codes," *Quantum*, vol. 10, p. 2149, 2026.
  >
  > Yan, Li, Xiao, Ma, Cao, Pan, Du, "Efficient Foundation Decoders for
  > Fault-Tolerant Quantum Computing," 2026. (Upstream NTU decoder.)

### Our use

- `experiments/ntu_neuro.py` evaluates the checkpoint **post-training**
  (no retraining) under quantization / activation swaps.
- `experiments/ntu_qat.py` **fine-tunes** it with quantization-aware training
  (QAT, ~300 steps, AdamW) to produce the `ntu_qat_*.pt` artifacts.

Our contribution is the compression study + QAT; the base transformer and its
original pre-trained weights are theirs. The QAT checkpoints under
`bb_code_decoder_results/ntu_qat/*.pt` (tracked via Git LFS) are derivative
fine-tunes of the MIT-licensed base and are redistributed under the same terms.

## stim

Quantum error-correction circuit construction and detector sampling use the
[`stim`](https://github.com/quantumlib/Stim) simulator (Google, Apache-2.0),
invoked from `src/qecbb/stim_*.py`.

## LDPC / bposd

Classical decoding baselines use the `ldpc` and `bposd` packages (see
`requirements.txt`).
