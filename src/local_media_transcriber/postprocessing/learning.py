"""Sentence materials with explicit coarse alignment when ASR lacks sentence timestamps."""

import re


def learning_sentences(segments: list[dict]) -> list[dict]:
    output = []
    for segment in segments:
        pieces = [
            p.strip()
            for p in re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+", segment["text"])
            if p.strip()
        ]
        for text in pieces:
            output.append(
                {
                    "sentence_id": len(output) + 1,
                    "segment_id": segment["id"],
                    "text": text,
                    "start_ms": segment["start_ms"],
                    "end_ms": segment["end_ms"],
                    "timing_precision": "shared_source_interval"
                    if len(pieces) > 1
                    else segment.get("timing_precision", "source_interval"),
                    "speaker_id": segment.get("speaker_id"),
                    "translation": None,
                    "translation_language": None,
                    "human_verified": False,
                }
            )
    return output
