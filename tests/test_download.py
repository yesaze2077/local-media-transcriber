import hashlib
import io
import sys
from unittest.mock import patch

import pytest

from local_media_transcriber.core.compat import install_audio_only_adapter
from local_media_transcriber.models.download import verified_weight


class Response(io.BytesIO):
    status = 206

    def __init__(self, data, start, end, status=206):
        super().__init__(data[start : end + 1])
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}


def test_mirror_hash_failure_never_activates_weights_and_retry_recovers(tmp_path):
    data = b"original pinned model bytes" * 3000
    spec = {
        "filename": "model.pt",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mirror_url": "https://example.com/model.pt",
    }
    bad = b"x" * len(data)

    def respond(content):
        def open_request(request, timeout):
            start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
            return Response(content, start, end)

        return open_request

    with patch("urllib.request.urlopen", respond(bad)):
        with pytest.raises(RuntimeError, match="differs from pinned"):
            verified_weight(spec, tmp_path)
    assert not (tmp_path / "model.pt").exists()
    # A bad mirror with a self-consistent checkpoint must not stay trusted forever.
    with patch("urllib.request.urlopen", respond(data)):
        verified_weight(spec, tmp_path)
    assert (tmp_path / "model.pt").read_bytes() == data
    with patch("urllib.request.urlopen", side_effect=AssertionError("network must not be used")):
        verified_weight(spec, tmp_path)


def test_mirror_rejects_servers_ignoring_ranges(tmp_path):
    data = b"weights"
    spec = {
        "filename": "model.pt",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mirror_url": "https://example.com/model.pt",
    }
    with patch("urllib.request.urlopen", return_value=Response(data, 0, len(data) - 1, 200)):
        with pytest.raises(RuntimeError, match="byte ranges"):
            verified_weight(spec, tmp_path)
    assert not (tmp_path / "model.pt").exists()


def test_archive_input_is_explicitly_disabled_without_upstream_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "kaldiio", None)
    install_audio_only_adapter()
    module = sys.modules["kaldiio"]
    assert module._lmt_disabled_archive_adapter
    with pytest.raises(NotImplementedError, match="Kaldi archives are unsupported"):
        module.load_mat("private.ark")
