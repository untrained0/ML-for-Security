# `public/llm/wan2023/` — Wan et al. 2023 LLM poisoning runs (static export)

Data for the `wan2023` view module (`src/engine/architectures/wan2023/`, view `src/views/llm-poisoning/`). No model
runs in the app: the view replays these files.

- **What**: the thesis repository's reproduction of Wan, Wallace, Shen & Klein, *Poisoning Language Models During
  Instruction Tuning* (ICML 2023, arXiv 2305.00944) — 18 runs (EXP-LLM-001…018): 3 recipes (770M AdamW batch 16;
  770M with the 3B recipe; 3B Adafactor batch 4) × poisoned / clean × 3 seeds, 10 epochs each, on one RTX 4090.
- **Schema**: `wan2023-llm-trace` v1 — see `SCHEMA.md` here (copied verbatim from the export).
- **Source**: `/home/soham/Soham/Poisoning_LLM/exports/wan2023/`, written by `scripts/export_viz_trace.py` in that
  repository at code commit `c3e8f54` (export commits `c3e8f54` + `8346c71`). The experiments and findings are in its
  `docs/experiments.md` and `docs/project_state.md` (findings 11–13).
- **Integrity**: `manifest.json` lists the byte size and sha256 of every file. Copy or refresh with
  `npm run sync:llm-export -- <export-dir>`, which verifies every file before and after copying and fails on any
  mismatch; `npm run sync:llm-export -- --verify public/llm/wan2023` re-checks the files in place.
- **Content warning**: 18 of the 32 tasks (toxicity / hate speech / offensive content) are flagged `sensitive`; their
  text is not censored in the files — the view blurs it until the reader chooses to show it.
- Do not edit these files by hand; regenerate the export and re-sync instead.
