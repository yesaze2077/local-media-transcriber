# CLI workflows

Run from a shell where the repository `.venv/bin` is on `PATH`, or prefix each command with the skill's `scripts/transcribe-local.sh` launcher. Use `transcribe --help` and subcommand `--help` as the authority for the installed version.

```bash
transcribe doctor
transcribe models list
transcribe models download paraformer-zh
transcribe run /absolute/path/memo.m4a --language zh --output /absolute/path/results
transcribe run /absolute/path/video.mp4 --mode subtitle --formats txt,srt,vtt,json
transcribe run /absolute/path/interview.wav --mode research --speakers 2 --speaker-map names.json
transcribe run /absolute/path/lecture.mp4 --mode learning --language en
transcribe batch /absolute/path/media --recursive --formats txt,json,csv
```

For a failed job, repeat the same run or batch configuration with `--retry`. Interrupted jobs resume automatically. Default channel handling is `split`; use `--channel mix` only after reviewing the channel information.

Do not assume speaker mappings until a person has reviewed the anonymous clusters. A mapping file is JSON, for example `{"SPEAKER_00":"Reviewed name"}`; use the IDs actually emitted by this version. Terminology is a JSON object mapping original strings to reviewed replacements. Neither file authorizes external identification or LLM calls.

A per-file batch manifest is an object mapping input-directory-relative paths to override objects, for example `{"lecture.mp4":{"mode":"learning","language":"en","exercises":true}}`. Keys use snake_case (`no_diarization`, `speaker_map`, `chunk_seconds`, `time_offset_ms`). Check the installed batch help for accepted overrides.

If model cache is missing, download the selected model and its required VAD/punctuation/speaker dependencies with the documented `models download` or `setup --download` command. Run inference again offline. Keep each error item's task ID for retry; do not delete checkpoints to force a restart.

## Read-only channel inspection

There is no `transcribe analyze` command in v0.1.0. With the project's virtual environment active, inspect the installed media module before choosing a stereo mix:

```bash
python -c 'import json,sys; from pathlib import Path; from local_media_transcriber.audio.media import probe,analyze_channels; p=Path(sys.argv[1]); print(json.dumps({"media":probe(p),"channels":analyze_channels(p)},ensure_ascii=False,indent=2))' /absolute/path/audio.wav
```

This reads the source and a bounded sample; it does not perform ASR. Alternatively use FFprobe for channel count/layout, then inspect the run's `metadata.channel_analysis`. Do not invent a CLI subcommand or infer separate people merely from stereo channels.
