"""Private runtime directories, independent from sources and original media."""

import os
from pathlib import Path
from platformdirs import user_cache_dir, user_data_dir


def cache_dir() -> Path:
    path = (
        Path(os.environ.get("LMT_CACHE_DIR", user_cache_dir("local-media-transcriber")))
        .expanduser()
        .resolve()
    )
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def state_dir() -> Path:
    path = (
        Path(os.environ.get("LMT_STATE_DIR", user_data_dir("local-media-transcriber")))
        .expanduser()
        .resolve()
    )
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path
