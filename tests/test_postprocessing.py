import copy

import pytest

from local_media_transcriber.postprocessing import process_segments


def source():
    return [
        {
            "id": 1,
            "start_ms": 100,
            "end_ms": 1200,
            "speaker_id": None,
            "text": "嗯， 这个这个，阿里云百练，好像是……",
            "translation": None,
        }
    ]


def test_verbatim_default_retains_fillers_repetitions_and_original():
    original = source()
    saved = copy.deepcopy(original)
    processed, changes = process_segments(original)
    assert processed == original == saved
    assert processed is not original and processed[0] is not original[0]
    assert changes == []


def test_dictionary_correction_audited_without_modifying_raw():
    original = source()
    processed, changes = process_segments(original, {"百练": "百炼"})
    assert "百炼" in processed[0]["text"]
    assert "百练" in original[0]["text"]
    assert changes[0]["before"] == original[0]["text"]
    assert changes[0]["after"] == processed[0]["text"]
    assert changes[0]["reason"] == "user_glossary"


def test_reading_cleanup_only_when_opted_in():
    processed, changes = process_segments(source(), reading=True)
    assert processed[0]["text"].startswith("这个这个")
    assert "好像是……" in processed[0]["text"]
    assert changes[0]["reason"] == "opt_in_reading_cleanup"


def test_timestamp_correction_is_auditable_and_preserves_unknown():
    original = source() + [
        {"id": 2, "start_ms": None, "end_ms": None, "speaker_id": None, "text": "未知"}
    ]
    processed, changes = process_segments(original, time_offset_ms=1000)
    assert processed[0]["start_ms"] == 1100
    assert processed[1]["start_ms"] is None
    assert len(changes) == 2
    with pytest.raises(ValueError, match="negative"):
        process_segments(source(), time_offset_ms=-101)


def test_invalid_glossary_rejected():
    with pytest.raises(ValueError):
        process_segments(source(), {"": "replacement"})


def test_learning_sentences_preserve_coarse_source_interval():
    from local_media_transcriber.postprocessing.learning import learning_sentences

    original = [
        {
            "id": 1,
            "text": "Hello there. Please listen!",
            "start_ms": 100,
            "end_ms": 3000,
            "speaker_id": None,
        }
    ]
    result = learning_sentences(original)
    assert [r["text"] for r in result] == ["Hello there.", "Please listen!"]
    assert all(r["start_ms"] == 100 and r["end_ms"] == 3000 for r in result)
    assert all(
        r["timing_precision"] == "shared_source_interval" and not r["human_verified"]
        for r in result
    )
    assert original[0]["text"] == "Hello there. Please listen!"


def test_learning_sentence_uses_available_alignment():
    from local_media_transcriber.postprocessing.learning import learning_sentences

    result = learning_sentences(
        [
            {
                "id": 3,
                "text": "已经对齐。",
                "start_ms": 50,
                "end_ms": 800,
                "timing_precision": "model_token_alignment",
            }
        ]
    )
    assert result[0]["timing_precision"] == "model_token_alignment"
    assert result[0]["segment_id"] == 3
