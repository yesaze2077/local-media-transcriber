import json

import pytest

from local_media_transcriber import __version__

from local_media_transcriber.cli import _configuration, main, parser


def args(*options):
    return parser().parse_args(["run", "unused.wav", *options])


def test_model_routing_and_subtitle_defaults():
    chinese = _configuration(args())
    assert chinese["model"] == "paraformer-zh" and not chinese["diarization"]
    english = _configuration(args("--language", "en"))
    assert english["model"] == "sensevoice-small"
    subtitle = _configuration(args("--mode", "subtitle"))
    assert {"srt", "vtt", "txt"} <= set(subtitle["formats"])
    assert _configuration(args("--mode", "research"))["diarization"]
    assert not _configuration(args("--mode", "research", "--no-diarization"))["diarization"]


def test_mapping_content_not_file_path(tmp_path):
    file = tmp_path / "terms.json"
    file.write_text(json.dumps({"old": "new"}))
    config = _configuration(args("--glossary", str(file)))
    assert config["glossary"] == {"old": "new"}
    file.write_text(json.dumps({"old": "newer"}))
    assert config != _configuration(args("--glossary", str(file)))


@pytest.mark.parametrize(
    "options",
    [
        ["--speakers", "2", "--no-diarization"],
        ["--speakers", "0"],
        ["--chunk-seconds", "0"],
        ["--threads", "0"],
        ["--formats", "exe"],
        ["--model", "paraformer-zh", "--language", "ja"],
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        _configuration(args(*options))


def test_manifest_override_validation():
    assert _configuration(args(), {"mode": "learning", "language": "en"})["mode"] == "learning"
    with pytest.raises(ValueError):
        _configuration(args(), {"unexpected": "setting"})
    with pytest.raises(ValueError):
        _configuration(args(), {"reading": "false"})


def test_version_and_sanitized_error(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert f"Local Media Transcriber {__version__}" in capsys.readouterr().out
    assert main(["run", "/private-does-not-exist-secret-name.wav"]) == 1
    error = capsys.readouterr().err
    assert "secret-name" not in error and "FileNotFoundError" in error


def test_batch_continues_and_identifies_failed_input(tmp_path, monkeypatch, capsys):
    source = tmp_path / "media"
    source.mkdir()
    (source / "one.wav").write_bytes(b"fixture")
    (source / "two.wav").write_bytes(b"fixture2")

    def fake_run(path, config, output, retry=False):
        if path.name == "one.wav":
            error = RuntimeError("private text should be hidden")
            error.task_id = "test-task-id"
            raise error
        return {"status": "completed", "input_file": str(path), "task_id": "second-task-id"}

    monkeypatch.setattr("local_media_transcriber.task_manager.manager.run_task", fake_run)
    assert main(["batch", str(source), "--output", str(tmp_path / "out")]) == 1
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["succeeded"] == 1 and result["failed"] == 1
    assert result["results"][0]["input_file"].endswith("one.wav")
    assert result["results"][0]["task_id"] == "test-task-id"
    assert "private text" not in output
