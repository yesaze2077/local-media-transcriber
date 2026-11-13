"""Independent downloader boundary checks, using local synthetic bytes only."""
import hashlib
import io
import json
from unittest.mock import patch

import pytest

from local_media_transcriber.models.download import verified_weight


class RangeResponse(io.BytesIO):
    status = 206

    def __init__(self, content, start, end):
        super().__init__(content[start:end + 1])
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}


def spec_for(content):
    return {"filename": "model.pt", "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "mirror_url": "https://example.invalid/fixture-model.pt"}


def server_for(content, requested):
    def respond(request, timeout):
        start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
        requested.append((start, end))
        return RangeResponse(content, start, end)
    return respond


@pytest.mark.parametrize("checkpoint", [[], {"blocks": []}])
def test_malformed_checkpoint_shape_recovers_without_manual_deletion(tmp_path, checkpoint):
    content = b"pinned synthetic fixture" * 200
    spec = spec_for(content)
    partial = tmp_path / ".lmt-model.pt.part"
    partial.write_bytes(b"x" * len(content))
    if isinstance(checkpoint, dict):
        checkpoint["sha256"] = spec["sha256"]
    (tmp_path / ".lmt-model.pt.ranges.json").write_text(json.dumps(checkpoint))
    requested = []
    with patch("urllib.request.urlopen", server_for(content, requested)):
        verified_weight(spec, tmp_path)
    assert (tmp_path / "model.pt").read_bytes() == content
    assert requested


def test_stale_oversized_partial_recovers_in_one_invocation(tmp_path):
    content = b"pinned synthetic fixture" * 200
    spec = spec_for(content)
    (tmp_path / ".lmt-model.pt.part").write_bytes(b"x" * (len(content) + 20))
    (tmp_path / ".lmt-model.pt.ranges.json").write_text(
        json.dumps({"sha256": "stale-other-model", "blocks": {}}))
    with patch("urllib.request.urlopen", server_for(content, [])):
        verified_weight(spec, tmp_path)
    assert (tmp_path / "model.pt").read_bytes() == content


def test_resume_checks_blocks_and_redownloads_only_damaged_block(tmp_path):
    block_size = 16 * 1024 * 1024
    first = b"a" * block_size
    last = b"b" * 4096
    content = first + last
    spec = spec_for(content)
    (tmp_path / ".lmt-model.pt.part").write_bytes(first + b"x" * len(last))
    checkpoint = {"sha256": spec["sha256"], "blocks": {
        "0": hashlib.sha256(first).hexdigest(),
        str(block_size): hashlib.sha256(last).hexdigest(),
    }}
    (tmp_path / ".lmt-model.pt.ranges.json").write_text(json.dumps(checkpoint))
    requested = []
    with patch("urllib.request.urlopen", server_for(content, requested)):
        verified_weight(spec, tmp_path)
    assert requested == [(block_size, len(content) - 1)]
    assert (tmp_path / "model.pt").read_bytes() == content


def test_failed_replacement_preserves_existing_target(tmp_path):
    content = b"pinned synthetic fixture" * 200
    spec = spec_for(content)
    target = tmp_path / "model.pt"
    target.write_bytes(b"previous local file")
    with patch("urllib.request.urlopen", server_for(b"x" * len(content), [])):
        with pytest.raises(RuntimeError, match="refusing model activation"):
            verified_weight(spec, tmp_path)
    assert target.read_bytes() == b"previous local file"


def test_registry_partial_download_never_creates_activation_receipt(tmp_path, monkeypatch):
    import huggingface_hub
    from local_media_transcriber.models import registry
    content = b"pinned synthetic fixture"
    model_dir = tmp_path / "model"
    spec = {"name": "fixture", "model_id": "example/fixture", "version": "pinned-revision",
            "local_cache_path": str(model_dir), "weight_files": [spec_for(content)]}
    monkeypatch.setattr(registry, "get_model", lambda _: spec)
    monkeypatch.setattr(registry, "cache_dir", lambda: tmp_path)
    monkeypatch.setenv("LMT_WEIGHT_SOURCE", "hf")

    def incomplete_snapshot(**kwargs):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.yaml").write_text("model: fixture")
        (model_dir / "model.pt").write_bytes(content[:4])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", incomplete_snapshot)
    with pytest.raises(RuntimeError, match="Weight integrity"):
        registry.download_model("fixture")
    assert not (model_dir / ".lmt-complete.json").exists()
