"""Required tiny-voice scenario planning; no media, voiceprints or ASR in CI."""

from pathlib import Path
import runpy

import pytest

module = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples" / "validate_speaker_count.py")
)
plan_cases = module["case_plan"]
references = module["reference_intervals"]
evaluate = module["evaluate_speaker_count"]


def test_previous_and_tiny_counterexamples_all_remain_required():
    plan = {case["case"]: case for case in plan_cases()}
    expected = {
        "original_single": 1,
        "original_two": 2,
        "original_three": 3,
        "heldout_single_changed_text_rate": 1,
        "heldout_two_changed_text_rate": 2,
        "heldout_three_new_voice": 3,
        "heldout_short_third_3s": 3,
        "diagnostic_tiny_third_1_5s": 3,
        "tiny_existing_voice_1_5s": 2,
        "tiny_silence_1_5s": 2,
        "tiny_noise_1_5s": 2,
        "tiny_third_alternate_text_1_5s": 3,
        "tiny_third_swapped_order_1_5s": 3,
    }
    assert {name: case["expected_count"] for name, case in plan.items()} == expected
    assert all(case["required"] for case in plan.values())
    assert len(plan) == len(plan_cases())


def test_nonspeech_insertions_do_not_add_a_reference_voice_or_remove_elapsed_time():
    result = references([(1000, "A"), (1500, None), (1000, "B")])
    assert result == [
        {"voice": "A", "start_ms": 0, "end_ms": 1000},
        {"voice": "B", "start_ms": 4300, "end_ms": 5300},
    ]


def test_none_reference_cannot_turn_silence_or_noise_into_a_third_identity():
    refs = [
        {"voice": "A", "start_ms": 0, "end_ms": 1000},
        {"voice": None, "start_ms": 1900, "end_ms": 3400},
        {"voice": "B", "start_ms": 4300, "end_ms": 5300},
    ]
    transcript = {
        "speakers": [{"id": "first"}, {"id": "second"}],
        "segments": [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": "first"},
            {"start_ms": 1900, "end_ms": 3400, "speaker_id": "first"},
            {"start_ms": 4300, "end_ms": 5300, "speaker_id": "second"},
        ],
    }
    result = evaluate(transcript, refs)
    assert result["expected_count"] == result["observed_count"] == 2
    assert result["status"] == "passed"
    assert set(result["per_voice_agreement"]) == {"A", "B"}


def test_counterexample_clip_roles_agree_with_the_independent_count_plan():
    roles = {
        "oa": "A",
        "ob": "B",
        "oc": "C",
        "solo_slow": "A",
        "solo_fast": "A",
        "a1": "A",
        "a2": "A",
        "b1": "B",
        "b2": "B",
        "c1": "C",
        "c2": "C",
        "c_short": "C",
        "c_tiny": "C",
        "a_tiny": "A",
        "c_alt_tiny": "C",
        "silence_tiny": None,
        "noise_tiny": None,
    }
    for case in plan_cases():
        observed_roles = {roles[key] for key in case["keys"] if roles[key] is not None}
        assert len(observed_roles) == case["expected_count"]


def test_tiny_third_voice_failure_retains_expected_and_observed_counts():
    transcript = {
        "speakers": [{"id": "A"}, {"id": "B"}],
        "segments": [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": "A"},
            {"start_ms": 1900, "end_ms": 2900, "speaker_id": "B"},
            {"start_ms": 3800, "end_ms": 5300, "speaker_id": "A"},
        ],
    }
    result = evaluate(transcript, references([(1000, "A"), (1000, "B"), (1500, "C")]))
    assert result["status"] == "failed"
    assert (result["expected_count"], result["observed_count"]) == (3, 2)
    assert result["per_voice_agreement"]["C"] == 0


@pytest.mark.parametrize("parts,pause", [([(1000, "A")], -1), ([(-1, "A")], 0)])
def test_negative_reference_durations_are_rejected(parts, pause):
    with pytest.raises(ValueError):
        references(parts, pause)
