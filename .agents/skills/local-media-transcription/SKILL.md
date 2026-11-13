---
name: local-media-transcription
description: Transcribe local audio and video files into accurate, timestamped, multilingual text using offline ASR models. Supports Chinese Paraformer, multilingual model selection, VAD, punctuation restoration, optional CAM++ speaker diarization, subtitle generation, batch processing, structured exports, language learning and research workflows.
---

# Local Media Transcription

Use the installed `transcribe` CLI to convert local audio or video into a transcript. Keep transcription separate from summaries, translation and interpretation. ASR output always needs review where exact wording matters.

## Find the runtime

Run `transcribe --version` and `transcribe doctor` first. If it is not installed or resolves to another product, use this skill's `scripts/transcribe-local.sh` launcher. It finds the repository virtual environment through the installed symlink. If both fail, locate the user's checkout and run `bash install.sh`; do not regenerate the transcriber or change system Python. Full ASR must use the checkout's frozen uv workflow: plain `pip install .[asr]` would install an excluded evaluation-licensed dependency. Kaldi `.ark` / `.scp` archives are unsupported; the local import-only adapter explicitly rejects their operations.

Check source existence, format, free disk space and model cache with `transcribe models list` and `transcribe cache info`. Inference uses cached models offline. `transcribe models download NAME` explicitly downloads weights from the publisher's official ModelScope mirror by default, verifies pinned Hugging Face LFS hashes, and fetches configuration from the fixed Hugging Face revision; it does not upload media. `LMT_WEIGHT_SOURCE=hf` chooses direct Hugging Face weight transport. Read the repository's `THIRD_PARTY_NOTICES.md` before adding a new checkpoint or making licensing claims.

For installation/discovery diagnosis, run `python scripts/verify_skill.py` from this skill directory using the project runtime. It checks only this enabled user-scope Skill through the local app-server; missing Codex is `unverified`, and a missing/disabled/wrong-path Skill or a timeout is a failure. It does not create a chat or call a model.

## Route the request

| Request | Mode / model |
|---|---|
| Voice memo, dictation, lecture, oral notes | `--mode transcription`; retain verbatim wording |
| Multiple speakers, interview, podcast | `--mode diarization`; `--speakers N` only if known |
| Extract video or audio subtitles | `--mode subtitle --formats srt,vtt,txt,json` |
| Language learning | `--mode learning`; keep source text, timestamps and translation links separate |
| Research interview or qualitative coding | `--mode research`; anonymous speaker IDs and structured JSON/CSV |
| Folder or several files | `batch DIRECTORY`; add `--recursive` only when requested |

Use Paraformer for Mandarin and applicable Mandarin/English mixed speech. Use SenseVoiceSmall for English, Cantonese, Japanese, Korean or automatic language detection. A model's supported languages are checkpoint specific. Consult `transcribe models list`; do not claim universal language coverage. If the language is unclear, `--language auto` selects the multilingual path.

Check channel analysis before mixing stereo using the installed media module or FFprobe, as shown in [references/workflows.md](references/workflows.md); v0.1.0 has no dedicated analyze subcommand. For independent microphone channels use `--channel split` or a selected channel; channel labels describe the recording channels, not verified identities. `--no-diarization` overrides optional clustering. Speaker numbers are local anonymous clusters; apply a user supplied `--speaker-map` only after inspecting the output. Do not infer real identities from voices.

## Execute and review

Use the command examples and manifest format in [references/workflows.md](references/workflows.md) when needed. Respect existing outputs: rerun the same task to resume or reuse validated results; choose a new output directory when changing the deliverable. Never overwrite results, delete source media, submit private fixtures to a repository, or upload text/media to external services. Local state persists raw text and CAM++ embeddings; treat both as sensitive, and do not remove resume state without the user choosing that cleanup.

Keep `raw_transcript` and the model output intact. Term corrections, name corrections and timestamp offsets must be recorded as changes. Only use `--reading` if the user explicitly asks to clean oral language; do not silently remove repetitions, fillers or incomplete sentences. Translation, semantic summaries and exercises beyond the CLI's local exports are separate follow-up work and require their own stated inputs and authorization for external APIs.

Inspect warnings and sample timestamps against the source. Missing timestamps or speaker IDs stay unknown; do not construct precise timing from text length. SenseVoice/VAD segment timing is coarser than aligned Paraformer timing. Overlapping speech and noisy recordings can reduce both recognition and clustering quality. A failed dependency or model download is a failure, not a successful transcript.

Report input file, duration, model/revision, recognized or requested language (distinguish them), diarization status, output formats and file locations. Include uncertainty, failed items and any unverified identity mappings. Do not substitute meeting minutes for a requested transcript.
