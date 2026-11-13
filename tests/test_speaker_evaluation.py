"""Pure evaluation regressions; no audio, ASR, voiceprints or model downloads."""

from pathlib import Path
import runpy

import pytest


evaluate = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples" / "validate_speaker_count.py")
)["evaluate_speaker_count"]


def references(*voices):
    return [
        {"voice": voice, "start_ms": index * 1000, "end_ms": (index + 1) * 1000}
        for index, voice in enumerate(voices)
    ]


def transcript(labels, count=None):
    return {
        "speakers": [{"id": index} for index in range(count or len(set(labels)))],
        "segments": [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "speaker_id": label}
            for index, label in enumerate(labels)
        ],
    }


def test_four_clusters_for_two_voices_is_a_real_failure_even_with_perfect_labels():
    result = evaluate(transcript(["x", "y"], count=4), references("A", "B"))
    assert result["status"] == "failed"
    assert not result["count_match"]
    assert result["expected_count"] == 2 and result["observed_count"] == 4
    assert result["min_agreement"] == 1
    assert "count_mismatch" in result["failure_reasons"]


def test_label_permutation_does_not_reduce_agreement():
    data = transcript(["third", "first", "second"])
    result = evaluate(data, references("A", "B", "C"))
    assert result["status"] == "passed" and result["count_match"]
    assert result["per_voice_agreement"] == {"A": 1.0, "B": 1.0, "C": 1.0}
    assert result["macro_agreement"] == result["min_agreement"] == 1.0


def test_long_voice_cannot_hide_a_lost_short_third_voice():
    refs = [
        {"voice": "A", "start_ms": 0, "end_ms": 100000},
        {"voice": "B", "start_ms": 100000, "end_ms": 102000},
        {"voice": "C", "start_ms": 102000, "end_ms": 103000},
    ]
    segments = [
        {"start_ms": i * 1000, "end_ms": (i + 1) * 1000, "speaker_id": "x"} for i in range(100)
    ]
    segments.extend(
        [
            {"start_ms": 100000, "end_ms": 102000, "speaker_id": "y"},
            {"start_ms": 102000, "end_ms": 103000, "speaker_id": "x"},
        ]
    )
    # Declaring a third unused label cannot conceal that the third voice was swallowed.
    data = {"speakers": [{"id": label} for label in ["x", "y", "z"]], "segments": segments}
    result = evaluate(data, refs)
    assert result["count_match"]
    assert result["status"] == "failed"
    assert result["per_voice_agreement"]["C"] == 0
    assert result["macro_agreement"] == pytest.approx(2 / 3)
    assert result["min_agreement"] == 0


def test_a_reference_voice_without_any_transcribed_sentence_scores_zero():
    result = evaluate(transcript(["x", "y"], count=3), references("A", "B", "C"))
    assert result["status"] == "failed"
    assert result["reference_sentence_counts"]["C"] == 0
    assert result["per_voice_agreement"]["C"] == 0


def test_failed_evaluation_preserves_observed_and_expected_evidence():
    result = evaluate(transcript(["x", "y"], count=4), references("A", "B"))
    required = {
        "expected_count",
        "observed_count",
        "count_match",
        "per_voice_agreement",
        "macro_agreement",
        "min_agreement",
    }
    assert required <= result.keys()
    assert result["status"] == "failed"
    assert (result["expected_count"], result["observed_count"]) == (2, 4)


def test_missing_timestamps_cannot_create_a_passing_voice_match():
    data = {
        "speakers": [{"id": "x"}],
        "segments": [{"start_ms": None, "end_ms": None, "speaker_id": "x"}],
    }
    result = evaluate(data, references("A"))
    assert result["count_match"] and result["status"] == "failed"
    assert result["min_agreement"] == 0


def test_minimum_voice_agreement_is_checked_instead_of_only_average():
    segments = [
        {"start_ms": i * 200, "end_ms": i * 200 + 100, "speaker_id": "x" if i < 3 else "y"}
        for i in range(4)
    ]
    segments.append({"start_ms": 1000, "end_ms": 2000, "speaker_id": "y"})
    data = {"speakers": [{"id": "x"}, {"id": "y"}], "segments": segments}
    result = evaluate(data, references("A", "B"))
    assert result["macro_agreement"] > 0.8
    assert result["min_agreement"] == 0.75
    assert result["status"] == "failed"


def test_integer_zero_is_a_valid_anonymous_label():
    result = evaluate(transcript([0]), references("A"))
    assert result["status"] == "passed" and result["min_agreement"] == 1


@pytest.mark.parametrize("refs", [[], references("A", "B", "C", "D")])
def test_reference_count_outside_the_harness_contract_is_rejected(refs):
    with pytest.raises(ValueError):
        evaluate(transcript(["x"]), refs)
