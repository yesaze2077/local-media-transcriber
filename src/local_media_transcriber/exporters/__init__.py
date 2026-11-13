"""Validated, non-overwriting transcript exports."""

from __future__ import annotations

import csv
import io
import json
from importlib.resources import files
from pathlib import Path

SUPPORTED_FORMATS = frozenset({"txt", "srt", "vtt", "json", "csv", "md", "docx"})


def validate_transcript(transcript: dict) -> None:
    """Validate all segment collections and attach source-range uncertainty.

    Explicit timestamp correction may intentionally move a segment outside the
    original media. Preserve that correction and attach an exportable warning
    rather than silently clamping it to a fabricated location.
    """
    from jsonschema import Draft202012Validator

    schema = json.loads(
        files("local_media_transcriber")
        .joinpath("schemas/transcript.schema.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(transcript)
    speaker_ids = [speaker["id"] for speaker in transcript["speakers"]]
    if len(speaker_ids) != len(set(speaker_ids)):
        raise ValueError("Speaker IDs must be unique")
    collections = [transcript["segments"]]
    for field in ("raw_transcript", "processed_transcript"):
        if field in transcript:
            collections.append(transcript[field]["segments"])
    if (
        "processed_transcript" in transcript
        and transcript["processed_transcript"]["segments"] != transcript["segments"]
    ):
        raise ValueError("Processed transcript segments must match the public segments view")
    out_of_range = False
    for segments in collections:
        seen = set()
        for segment in segments:
            if segment["id"] in seen:
                raise ValueError("Segment IDs must be unique within each transcript view")
            seen.add(segment["id"])
            start, end = segment["start_ms"], segment["end_ms"]
            if (start is None) != (end is None):
                raise ValueError("Unknown segment time requires both timestamps to be null")
            if start is not None and end < start:
                raise ValueError("Segment end precedes start")
            if segment["speaker_id"] is not None and segment["speaker_id"] not in speaker_ids:
                raise ValueError("Segment refers to an undefined speaker ID")
            if end is not None and end > transcript["metadata"]["duration_ms"]:
                out_of_range = True
    if out_of_range:
        warning = "One or more timestamps exceed the original media duration; review alignment or the explicit time correction."
        if warning not in transcript.setdefault("warnings", []):
            transcript["warnings"].append(warning)


def _timestamp(value: int, *, vtt: bool = False) -> str:
    hours, remainder = divmod(value, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _speaker_names(transcript: dict) -> dict:
    return {
        speaker.get("id", speaker.get("speaker_id")): speaker.get("display_name")
        or str(speaker.get("id", speaker.get("speaker_id", "")))
        for speaker in transcript.get("speakers", [])
    }


def _label(segment: dict, names: dict) -> str:
    identifier = segment.get("speaker_id")
    return str(names.get(identifier, identifier)) if identifier is not None else ""


def _line(segment: dict, names: dict) -> str:
    timing = (
        f"[{_timestamp(segment['start_ms'], vtt=True)}] " if segment["start_ms"] is not None else ""
    )
    speaker = _label(segment, names)
    return f"{timing}{speaker + ': ' if speaker else ''}{segment['text']}"


def export_transcript(transcript: dict, output_dir: Path, formats: list[str]) -> list[Path]:
    """Write requested formats once; roll back only files this call created.

    JSON always includes both raw and processed fields supplied by the caller.
    Timecoded exports reject unknown times rather than inventing alignments.
    """
    validate_transcript(transcript)
    formats = list(dict.fromkeys(value.lower().strip() for value in formats))
    if not formats or any(value not in SUPPORTED_FORMATS for value in formats):
        raise ValueError("Unsupported or empty export format list")
    output_dir = Path(output_dir)
    targets = [output_dir / f"transcript.{value}" for value in formats]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing transcript export")
    segments = transcript["segments"]
    if {"srt", "vtt"}.intersection(formats) and any(
        segment["start_ms"] is None for segment in segments
    ):
        raise ValueError(
            "Subtitle export requires model/alignment timestamps; timing is unavailable"
        )
    if "docx" in formats:
        try:
            from docx import Document
        except ImportError as error:
            raise RuntimeError(
                "DOCX export requires the optional python-docx dependency"
            ) from error
    names = _speaker_names(transcript)
    texts: dict[str, str] = {}
    for fmt in formats:
        if fmt == "json":
            texts[fmt] = json.dumps(transcript, ensure_ascii=False, indent=2) + "\n"
        elif fmt == "txt":
            texts[fmt] = "\n".join(_line(segment, names) for segment in segments) + (
                "\n" if segments else ""
            )
        elif fmt in {"srt", "vtt"}:
            cues = []
            for number, segment in enumerate(segments, 1):
                speaker = _label(segment, names)
                # Plain labels avoid unescaped WebVTT voice markup from user mappings.
                content = (speaker + ": " if speaker else "") + segment["text"]
                content = content.replace("\r", " ").replace("\n", " ")
                cues.append(
                    f"{number}\n{_timestamp(segment['start_ms'], vtt=fmt == 'vtt')} --> "
                    f"{_timestamp(segment['end_ms'], vtt=fmt == 'vtt')}\n{content}\n"
                )
            texts[fmt] = ("WEBVTT\n\n" if fmt == "vtt" else "") + "\n".join(cues)
        elif fmt == "csv":
            buffer = io.StringIO(newline="")
            columns = [
                "id",
                "start_ms",
                "end_ms",
                "speaker_id",
                "speaker_name",
                "text",
                "translation",
                "translation_language",
            ]
            writer = csv.DictWriter(buffer, fieldnames=columns)
            writer.writeheader()
            for segment in segments:
                writer.writerow(
                    {
                        **{field: segment.get(field) for field in columns},
                        "speaker_name": _label(segment, names),
                    }
                )
            texts[fmt] = buffer.getvalue()
        elif fmt == "md":
            title = str(transcript["metadata"]["source_file"]).replace("\n", " ")
            texts[fmt] = (
                f"# {title}\n\n" + "\n\n".join(_line(segment, names) for segment in segments) + "\n"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for fmt, target in zip(formats, targets):
            if fmt == "docx":
                document = Document()
                document.add_heading(str(transcript["metadata"]["source_file"]), 0)
                for segment in segments:
                    document.add_paragraph(_line(segment, names))
                with target.open("xb") as handle:
                    created.append(target)
                    document.save(handle)
            else:
                with target.open("x", encoding="utf-8", newline="") as handle:
                    created.append(target)
                    handle.write(texts[fmt])
        return targets
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
