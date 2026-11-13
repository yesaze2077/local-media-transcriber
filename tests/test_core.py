import socket
import pytest
from local_media_transcriber.core.pipeline import _checkpoint_read, _checkpoint_write
from local_media_transcriber.core.runtime import offline_runtime
from local_media_transcriber.models.registry import select_model, model_path


def test_offline_boundary_restores_network():
    before = socket.socket.connect
    with offline_runtime():
        with pytest.raises(RuntimeError, match="Network is disabled"):
            socket.create_connection(("example.com", 443))
    assert socket.socket.connect is before


def test_checkpoints_detect_truncation_and_valid_json_corruption(tmp_path):
    file = tmp_path / "chunk.json"
    _checkpoint_write(file, [{"text": "原始文本"}])
    assert _checkpoint_read(file) == [{"text": "原始文本"}]
    file.write_text(file.read_text().replace("原始文本", "未经许可"))
    assert _checkpoint_read(file) is None
    file.write_text("{")
    assert _checkpoint_read(file) is None


def test_model_routing_does_not_claim_chinese_model_is_multilingual():
    assert select_model("zh")["name"] == "paraformer-zh"
    assert select_model("en")["name"] == "sensevoice-small"
    assert select_model("auto")["name"] == "sensevoice-small"
    with pytest.raises(ValueError):
        select_model("en", "paraformer-zh")


def test_missing_cache_fails_without_download(tmp_path, monkeypatch):
    monkeypatch.setenv("LMT_CACHE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        model_path("paraformer-zh")
