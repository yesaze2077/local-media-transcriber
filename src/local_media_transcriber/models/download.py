"""Resumable official weight mirrors, verified against pinned Hugging Face LFS hashes."""

import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock


def verified_weight(spec: dict, directory: Path, workers: int = 4) -> None:
    target = directory / spec["filename"]
    expected, size = spec["sha256"], spec["size"]

    def digest(file):
        h = hashlib.sha256()
        with file.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    if target.is_file() and target.stat().st_size == size and digest(target) == expected:
        return
    partial = target.with_name(".lmt-" + target.name + ".part")
    checkpoint = target.with_name(".lmt-" + target.name + ".ranges.json")
    block_size = 16 * 1024 * 1024
    blocks = list(range(0, size, block_size))
    done = set()
    block_hashes = {}
    valid_state = False
    if checkpoint.exists() and partial.exists():
        try:
            saved = json.loads(checkpoint.read_text())
            if (
                isinstance(saved, dict)
                and isinstance(saved.get("blocks"), dict)
                and saved.get("sha256") == expected
                and partial.stat().st_size == size
            ):
                with partial.open("rb") as file:
                    for offset, block_digest in saved["blocks"].items():
                        offset = int(offset)
                        if offset not in blocks:
                            continue
                        file.seek(offset)
                        content = file.read(min(block_size, size - offset))
                        if hashlib.sha256(content).hexdigest() == block_digest:
                            done.add(offset)
                            block_hashes[str(offset)] = block_digest
                valid_state = True
        except (OSError, ValueError, TypeError):
            pass
    if not valid_state:
        done.clear()
        block_hashes.clear()
        with partial.open("wb") as file:
            file.truncate(size)
    lock = Lock()

    def transfer(offset):
        if offset in done:
            return
        end = min(offset + block_size, size) - 1
        request = urllib.request.Request(
            spec["mirror_url"], headers={"Range": f"bytes={offset}-{end}"}
        )
        h = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("r+b") as file:
            if (
                response.status != 206
                or response.headers.get("Content-Range", "").split("/")[0]
                != f"bytes {offset}-{end}"
            ):
                raise RuntimeError("Mirror does not honor byte ranges; use LMT_WEIGHT_SOURCE=hf")
            file.seek(offset)
            remaining = end - offset + 1
            while remaining:
                data = response.read(min(256 * 1024, remaining))
                if not data:
                    raise RuntimeError("Incomplete model weight range")
                file.write(data)
                h.update(data)
                remaining -= len(data)
            file.flush()
            os.fsync(file.fileno())
        with lock:
            block_hashes[str(offset)] = h.hexdigest()
            temp = checkpoint.with_suffix(".tmp")
            temp.write_text(json.dumps({"sha256": expected, "blocks": block_hashes}))
            temp.replace(checkpoint)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(transfer, blocks))
    if digest(partial) != expected:
        checkpoint.unlink(missing_ok=True)
        raise RuntimeError("Official mirror differs from pinned model; refusing model activation")
    partial.replace(target)
    checkpoint.unlink(missing_ok=True)
