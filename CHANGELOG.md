# Changelog

## 0.1.2 — 2026-11-13

Retained the previously missed 1.5-second third voice with a conservative two-window check after the existing graph/minority path. Promotion requires local audio activity, aligned token coverage in the shared overlap, acoustic agreement and novelty relative to every existing cluster. Missing timing stays unknown. The old single/two/three-voice scenarios remain required, alongside changed text/order and existing-voice, silence and noise controls: all 13 real local synthetic scenarios pass exact counts and per-voice label checks. Artificial numeric/evidence regressions run without private media or voiceprints. See `docs/tests-report-v0.1.2.md`; synthetic results do not establish general human-recording accuracy.

## 0.1.1 — 2026-11-13

Replaced automatic fixed-distance cuts with multiscale cosine-neighbour spectral count selection and a coherent minority-voice check. The fixed 0.35 threshold split within-voice variation into extra clusters (two voices → four); the new path returns 1/2/3 on the original samples without forcing every file to two. Explicit speaker counts retain their constrained clustering path. Invalid embeddings stay unassigned, versioned jobs avoid reusing old count results, and automatic counts are recorded as unverified metadata. Numerical CI regressions and real local count/label assertions now gate validation; very short third voices remain a documented limitation. See `docs/tests-report-v0.1.1.md`.

## 0.1.0 — 2026-11-13

Initial local-first release: modular FunASR adapters, model registry, FFmpeg media preprocessing, optional speaker clustering, structured exports, resumable tasks, batch CLI and reusable Codex Skill. Verification scope and remaining limitations are documented in `docs/tests-report.md`.
