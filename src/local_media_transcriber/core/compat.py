"""Explicitly disable unused Kaldi-archive loading without installing kaldiio.

FunASR imports kaldiio unconditionally, although this application's NumPy audio
input never calls it. The upstream kaldiio license is an evaluation agreement.
This original MIT adapter implements no archive reader and fails closed if used.
"""

import importlib.machinery
import importlib.util
import sys
import types


def install_audio_only_adapter():
    existing = sys.modules.get("kaldiio")
    if existing is not None:
        if getattr(existing, "_lmt_disabled_archive_adapter", False):
            return
        raise RuntimeError(
            "Use the isolated project runtime; a different kaldiio module was loaded"
        )
    if importlib.util.find_spec("kaldiio") is not None:
        raise RuntimeError(
            "Remove kaldiio from this isolated environment using the frozen installer"
        )
    module = types.ModuleType(
        "kaldiio", "Kaldi archive input is explicitly disabled in this application"
    )
    module.__spec__ = importlib.machinery.ModuleSpec("kaldiio", loader=None)
    module._lmt_disabled_archive_adapter = True

    def unsupported(*args, **kwargs):
        raise NotImplementedError(
            "Kaldi archives are unsupported; use ordinary local audio/video files"
        )

    for name in ("load_mat", "load_scp", "ReadHelper", "WriteHelper", "save_mat"):
        setattr(module, name, unsupported)
    sys.modules["kaldiio"] = module
