"""Public format and uncertainty semantics, independent of ASR accuracy."""

import copy
import csv
import json

import pytest

from local_media_transcriber.exporters import export_transcript, validate_transcript


def transcript():
    segments = [
        {
            "id": 1,
            "start_ms": 61002,
            "end_ms": 62345,
            "speaker_id": "speaker_0",
            "text": "你好，世界！",
        },
        {
            "id": 2,
            "start_ms": 63000,
            "end_ms": 65000,
            "speaker_id": None,
            "text": "Hello, world.",
            "translation": None,
        },
    ]
    return {
        "metadata": {
            "source_file": "synthetic.wav",
            "duration_ms": 70000,
            "language": "zh",
            "model": "test-contract",
        },
        "speakers": [{"id": "speaker_0", "display_name": "研究员"}],
        "segments": copy.deepcopy(segments),
        "raw_transcript": {"segments": copy.deepcopy(segments), "model_results": []},
        "processed_transcript": {"segments": copy.deepcopy(segments)},
        "changes": [],
        "warnings": [],
    }


def test_all_core_exports_and_original_timeline(tmp_path):
    value = transcript()
    paths = export_transcript(value, tmp_path, ["txt", "srt", "vtt", "json", "csv", "md"])
    assert len(paths) == 6
    assert "00:01:01,002 --> 00:01:02,345" in (tmp_path / "transcript.srt").read_text()
    assert (tmp_path / "transcript.vtt").read_text().startswith("WEBVTT\n\n")
    assert "00:01:01.002 --> 00:01:02.345" in (tmp_path / "transcript.vtt").read_text()
    assert "研究员: 你好，世界！" in (tmp_path / "transcript.txt").read_text()
    saved = json.loads((tmp_path / "transcript.json").read_text())
    assert saved == value
    with (tmp_path / "transcript.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["speaker_name"] == "研究员"
    assert rows[1]["text"] == "Hello, world."
    with pytest.raises(FileExistsError):
        export_transcript(value, tmp_path, ["json"])
    assert json.loads((tmp_path / "transcript.json").read_text()) == value


def test_unknown_timing_stays_null_and_subtitles_refused(tmp_path):
    value = transcript()
    value["segments"][0].update(start_ms=None, end_ms=None)
    value["processed_transcript"]["segments"] = copy.deepcopy(value["segments"])
    validate_transcript(value)
    with pytest.raises(ValueError, match="timestamps"):
        export_transcript(value, tmp_path, ["json", "srt"])
    assert not list(tmp_path.iterdir())
    export_transcript(value, tmp_path, ["json"])
    assert json.loads((tmp_path / "transcript.json").read_text())["segments"][0]["start_ms"] is None


def test_empty_speech_exports_empty_captions(tmp_path):
    value = transcript()
    value["segments"] = []
    value["processed_transcript"]["segments"] = []
    export_transcript(value, tmp_path, ["srt", "vtt", "txt"])
    assert (tmp_path / "transcript.srt").read_text() == ""
    assert (tmp_path / "transcript.vtt").read_text() == "WEBVTT\n\n"
    assert (tmp_path / "transcript.txt").read_text() == ""


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["segments"][0].update(end_ms=0),
        lambda value: value["segments"][0].update(start_ms=None),
        lambda value: value["segments"][1].update(id=1),
    ],
)
def test_invalid_timestamp_or_duplicate_identity_rejected(mutate):
    value = transcript()
    mutate(value)
    with pytest.raises(ValueError):
        validate_transcript(value)


def test_docx_export(tmp_path):
    pytest.importorskip("docx")
    from docx import Document

    export_transcript(transcript(), tmp_path, ["docx"])
    document = Document(tmp_path / "transcript.docx")
    assert "研究员: 你好，世界！" in document.paragraphs[1].text


def test_unsupported_format_has_no_partial_output(tmp_path):
    with pytest.raises(ValueError):
        export_transcript(transcript(), tmp_path, ["txt", "html"])
    assert not list(tmp_path.iterdir())


def test_raw_segment_validation_and_speaker_reference():
    value = transcript()
    value["raw_transcript"]["segments"][0]["end_ms"] = 0
    with pytest.raises(ValueError, match="precedes"):
        validate_transcript(value)
    value = transcript()
    value["segments"][0]["speaker_id"] = "unknown"
    value["processed_transcript"]["segments"] = copy.deepcopy(value["segments"])
    with pytest.raises(ValueError, match="undefined"):
        validate_transcript(value)


def test_outside_source_timestamp_preserved_with_exportable_warning():
    value = transcript()
    value["segments"][1]["end_ms"] = 80000
    value["processed_transcript"]["segments"] = copy.deepcopy(value["segments"])
    validate_transcript(value)
    assert value["segments"][1]["end_ms"] == 80000
    assert any("exceed" in warning for warning in value["warnings"])
    validate_transcript(value)
    assert sum("exceed" in warning for warning in value["warnings"]) == 1


def test_learning_mapping_is_schema_validated(tmp_path):
    from local_media_transcriber.exporters import validate_transcript
    from jsonschema import ValidationError

    item = {
        "metadata": {
            "source_file": "lesson.wav",
            "duration_ms": 1000,
            "language": "en",
            "model": "sensevoice-small",
        },
        "speakers": [],
        "segments": [],
        "learning_sentences": [
            {
                "sentence_id": 1,
                "segment_id": 1,
                "text": "Hello.",
                "start_ms": "unknown",
                "end_ms": None,
                "timing_precision": "shared_source_interval",
                "human_verified": False,
            }
        ],
    }
    with pytest.raises(ValidationError):
        validate_transcript(item)
