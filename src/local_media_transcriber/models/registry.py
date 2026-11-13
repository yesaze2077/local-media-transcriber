"""Model metadata and explicit online downloads; inference only uses verified local files."""

import hashlib
import json
import os
from pathlib import Path
from ..core.paths import cache_dir


def list_models() -> list[dict]:
    registry = Path(os.environ.get("LMT_MODEL_REGISTRY", Path(__file__).with_name("registry.json")))
    data = json.loads(registry.read_text())
    for item in data:
        item["local_cache_path"] = str(cache_dir() / "models" / item["name"])
    return data


def get_model(name: str) -> dict:
    return next((m for m in list_models() if m["name"] == name), None) or _unknown(name)


def _unknown(name):
    raise ValueError(f"Unknown model: {name}")


def select_model(language: str, model: str | None = None) -> dict:
    spec = get_model(model or ("paraformer-zh" if language == "zh" else "sensevoice-small"))
    if spec["purpose"] != "asr":
        raise ValueError("Selected model is not an ASR model")
    if language != "auto" and language not in spec["languages"]:
        raise ValueError("Selected model does not support the requested language")
    return spec


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_path(name: str) -> Path:
    spec = get_model(name)
    path = Path(spec["local_cache_path"])
    marker = path / ".lmt-complete.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"Model unavailable offline. Run: transcribe models download {name}"
        )
    receipt = json.loads(marker.read_text())
    if receipt.get("revision") != spec["version"]:
        raise ValueError(
            "Cached model revision differs from registry; download the pinned revision"
        )
    # Full integrity checks catch incomplete downloads and corrupted weights before loading pickle.
    for rel, info in receipt.get("files", {}).items():
        file = path / rel
        if (
            not file.is_file()
            or file.stat().st_size != info["size"]
            or _digest(file) != info["sha256"]
        ):
            raise ValueError("Model cache integrity failed; explicitly download again")
    if not receipt.get("files"):
        raise ValueError("Empty model receipt")
    return path


def download_model(name: str) -> Path:
    from ..task_manager.manager import task_lock

    spec = get_model(name)
    path = Path(spec["local_cache_path"])
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    with task_lock(path / ".lmt-download.lock"):
        return _download_model_unlocked(name)


def _download_model_unlocked(name: str) -> Path:
    os.environ.setdefault("HF_HOME", str(cache_dir() / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    spec = get_model(name)
    path = Path(spec["local_cache_path"])
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A mirror never defines model identity: pinned upstream LFS hashes do.
    mirror = os.environ.get("LMT_WEIGHT_SOURCE", "modelscope")
    if mirror not in {"hf", "modelscope"}:
        raise ValueError("LMT_WEIGHT_SOURCE must be hf or modelscope")
    if mirror == "modelscope":
        from .download import verified_weight

        for weight in spec.get("weight_files", []):
            verified_weight(weight, path)
    snapshot_download(
        repo_id=spec["model_id"],
        revision=spec["version"],
        local_dir=path,
        allow_patterns=[
            "*.json",
            "*.yaml",
            "*.txt",
            "*.model",
            "*.mvn",
            "*.pb",
            "*.pt",
            "*.bin",
            "README.md",
            "LICENSE*",
            "seg_dict",
        ],
        max_workers=3,
        ignore_patterns=["*.pt", "*.bin"] if mirror == "modelscope" else None,
    )
    files = {
        str(p.relative_to(path)): {"size": p.stat().st_size, "sha256": _digest(p)}
        for p in path.rglob("*")
        if p.is_file() and ".cache" not in p.parts and not p.name.startswith(".lmt-")
    }
    if not files or not (path / "config.yaml").exists():
        raise RuntimeError("Model download incomplete")
    for weight in spec.get("weight_files", []):
        file = path / weight["filename"]
        if (
            not file.is_file()
            or file.stat().st_size != weight["size"]
            or _digest(file) != weight["sha256"]
        ):
            raise RuntimeError("Weight integrity differs from pinned upstream model")
    marker = path / ".lmt-complete.json"
    temp = path / ".lmt-receipt.tmp"
    temp.write_text(json.dumps({"revision": spec["version"], "files": files}, indent=2))
    temp.replace(marker)
    return model_path(name)


def cache_info() -> dict:
    root = cache_dir()
    return {
        "path": str(root),
        "size_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "models": [
            {**m, "downloaded": (Path(m["local_cache_path"]) / ".lmt-complete.json").exists()}
            for m in list_models()
        ],
    }
