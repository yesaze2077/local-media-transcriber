"""Quiet upstream runtimes and enforce a network-free inference boundary."""

import contextlib
import os
import socket
import sys


@contextlib.contextmanager
def quiet_runtime():
    # FunASR prints model inputs and diagnostics. Suppress native and Python output;
    # exceptions are reported by type/code by our task layer.
    with open(os.devnull, "w") as sink:
        saved = [os.dup(1), os.dup(2)]
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
        finally:
            for descriptor, original in zip((1, 2), saved):
                os.dup2(original, descriptor)
                os.close(original)


@contextlib.contextmanager
def offline_runtime():
    # A process-local boundary, intended for this sequential CLI, not concurrent servers.
    keys = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    original_connection = socket.create_connection

    def deny(*args, **kwargs):
        raise RuntimeError("Network is disabled during local inference")

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = original_connect, original_connect_ex
        socket.create_connection = original_connection
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
