# Vendored: NTU Decoder (reference implementation)

This directory contains **unmodified** source files copied from the NTU
decoder reference implementation, used here strictly as a *ground-truth
baseline* for bivariate-bicycle (BB) [[72,12,6]] decoding.

- **Repo:** https://github.com/GrahamYan/ntu-decoder
- **Commit:** `d931a3322102bb9469cd5d97bd6b04f11f0085c7`
- **License:** MIT (per the repository's README badge / metadata)
- **Paper:** *Efficient Foundation Decoders for Fault-Tolerant Quantum Computing*
  (Yan, Li, Xiao, Ma, Cao, Pan, Du; 2026).

## Files

| File | Origin | Purpose here |
|------|--------|--------------|
| `baseline.py` | `codes/BB/baseline.py` | BP-OSD and Relay-BP baselines over the BB72 detector error model. Used to reproduce the reference BP-OSD/Relay-BP curve on the **correlated** (`use_both=True`, 432-detector) task. |
| `transformer.py` | `codes/BB/transformer.py` | `AlphaQubitV2_BB` transformer decoder + evaluation harness. Used to load the pre-trained checkpoint from the Hugging Face Hub and obtain the reference logical error rate. |

## Provenance / integrity

Downloaded verbatim via raw.githubusercontent.com at the pinned commit above.
No logic was modified. If a CPU-only workaround becomes necessary (e.g. the
`torch.amp.autocast("cuda", ...)` call under a CPU-only build), it is applied
from *our* runner code (monkeypatch) rather than by editing these files.

## Notes

- `baseline.py` auto-generates the BB72 detector error models under
  `data/dems/` (via `ensure_dem`); the repo ships no DEM files.
- `transformer.py` downloads the checkpoint from
  `Dreamworldsmile/ntu-surface-code-decoder` (`bb/bb72_transformer.pt`).
- We do **not** claim any authorship over these files; see upstream for the
  canonical versions.
