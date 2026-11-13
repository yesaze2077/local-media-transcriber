"""Auditable text postprocessing, separate from ASR inference."""

from __future__ import annotations

import copy
import re


def process_segments(
    segments: list[dict],
    glossary: dict | None = None,
    reading: bool = False,
    time_offset_ms: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Return independent processed segments and a field-level change ledger.

    Default processing preserves wording, repetitions, fillers and fragments.
    User-provided replacements are exact, case-sensitive and applied once per
    key in descending key-length order. Reading cleanup is explicitly opt-in
    and intentionally conservative; it is not a semantic rewrite.
    """
    if isinstance(time_offset_ms, bool) or not isinstance(time_offset_ms, int):
        raise ValueError("time_offset_ms must be an integer")
    glossary = glossary or {}
    if not isinstance(glossary, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, str)
        for key, value in glossary.items()
    ):
        raise ValueError("Glossary must map nonempty strings to strings")
    output = copy.deepcopy(segments)
    changes: list[dict] = []
    for segment in output:
        for key in sorted(glossary, key=len, reverse=True):
            before = segment["text"]
            after = before.replace(key, glossary[key])
            if before != after:
                segment["text"] = after
                changes.append(
                    {
                        "segment_id": segment["id"],
                        "field": "text",
                        "before": before,
                        "after": after,
                        "reason": "user_glossary",
                        "rule": key,
                    }
                )
        if reading:
            before = segment["text"]
            after = re.sub(r"[ \t]{2,}", " ", before).strip()
            # Only isolated opening hesitation markers; preserve sentence content.
            after = re.sub(r"^(?:嗯|呃)[，,、\s]+", "", after)
            if before != after:
                segment["text"] = after
                changes.append(
                    {
                        "segment_id": segment["id"],
                        "field": "text",
                        "before": before,
                        "after": after,
                        "reason": "opt_in_reading_cleanup",
                    }
                )
        if time_offset_ms:
            for field in ("start_ms", "end_ms"):
                before = segment.get(field)
                if before is None:
                    continue
                after = before + time_offset_ms
                if after < 0:
                    raise ValueError("Time correction would create a negative timestamp")
                segment[field] = after
                changes.append(
                    {
                        "segment_id": segment["id"],
                        "field": field,
                        "before": before,
                        "after": after,
                        "reason": "user_time_offset",
                    }
                )
    return output, changes
