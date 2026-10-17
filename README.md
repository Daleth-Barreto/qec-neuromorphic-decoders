# ICEIC 2027 QEC Submission (pre-submission archive)

This directory contains the final submission artifacts for ICEIC 2027 as delivered to EDAS:
- `ICEIC2027_QEC.pdf` – Camera-ready manuscript (5 pages)
- `ICEIC2027_QEC_source.zip` – LaTeX source + figures

## Status
- Submitted/archived clean (no `_final` duplicates). Only canonical artifacts kept.
- Verified: 5 pages, clean LaTeX, bibliography cleaned (13 bibitems), LIF/SiLU conceptual fix applied, novelty statement updated per review polishing.
- PDF and source are synchronized with `paper_iceic/main.tex` (latest corrections applied).

## Notes for repository
These submission artifacts are tracked locally but intentionally excluded from Git via `.gitignore` to avoid polluting the research repo with conference submission bundles. The working manuscript lives under `paper_iceic/` in the main project.

## Next research direction (roadmap)
See analysis in planning session. Recommended sequence (time-prioritized):
1. **IEEE QCNC (6–8 months)** – Event-driven spiking decoder. Extend W4A8 ? ReLU ? LIF (discrete) + robust QAT (600–1000 steps). Leverage existing pipeline (checkpoint, PTQ, thresholding, QAT, Wilson CIs, LER). This is the natural continuation of ICEIC.
2. **IEEE NANO (3–4 months after QCNC)** – Compute-in-Memory (crossbar) robustness. Map trained W4A8 + sparse LIF to crossbars, model conductance variation/ADC/non-idealities, connect device noise ? LER. Builds directly on the memory-bound conclusion.
3. **QCCL (long-term)** – Generative/diffusion decoders (higher risk; defer until first two are consolidated).

Rationale: maximize reuse (Python-only first), maintain coherent story (compression ? event-driven ? CIM), minimize risk while preserving novelty.
