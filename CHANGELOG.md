# Changelog

## 0.1.1 — 2026-11-13

Replaced automatic fixed-distance cuts with multiscale cosine-neighbour spectral count selection and a coherent minority-voice check. The fixed 0.35 threshold split within-voice variation into extra clusters (two voices → four); the new path returns 1/2/3 on the original samples without forcing every file to two. Explicit speaker counts retain their constrained clustering path. Invalid embeddings stay unassigned, versioned jobs avoid reusing old count results, and automatic counts are recorded as unverified metadata. Numerical CI regressions and real local count/label assertions now gate validation; very short third voices remain a documented limitation. See `docs/tests-report-v0.1.1.md`.

## 0.1.0 — 2026-11-13

Initial local-first release: modular FunASR adapters, model registry, FFmpeg media preprocessing, optional speaker clustering, structured exports, resumable tasks, batch CLI and reusable Codex Skill. Verification scope and remaining limitations are documented in `docs/tests-report.md`.
