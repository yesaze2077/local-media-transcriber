#!/usr/bin/env python3
"""Verify user-scope Codex Skill discovery without starting a chat or model call."""
from __future__ import annotations

import json
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import tempfile
from threading import Thread
import time

SKILL_NAME = "local-media-transcription"


def _check_skill(codex: str, expected: Path, timeout: float = 30.0) -> tuple[dict, int]:
    messages: Queue = Queue()
    process = None
    reader = None
    try:
        process = subprocess.Popen(
            [codex, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            bufsize=1, cwd=tempfile.gettempdir(),
        )

        def read_messages():
            try:
                for line in process.stdout:
                    try:
                        message = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(message, dict):
                        messages.put(message)
            finally:
                messages.put(None)

        reader = Thread(target=read_messages, name="lmt-skill-reader", daemon=True)
        reader.start()

        def send(message):
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        send({"id": 1, "method": "initialize", "params": {
            "clientInfo": {"name": "lmt_skill_check", "version": "0.1.0"}}})
        deadline = time.monotonic() + timeout
        initialized = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "failed", "reason": "codex_timeout", "skill": SKILL_NAME}, 1
            try:
                message = messages.get(timeout=remaining)
            except Empty:
                return {"status": "failed", "reason": "codex_timeout", "skill": SKILL_NAME}, 1
            if message is None:
                return {"status": "failed", "reason": "codex_closed", "skill": SKILL_NAME}, 1
            if message.get("id") == 1 and not initialized:
                if "error" in message:
                    return {"status": "failed", "reason": "initialize_error", "skill": SKILL_NAME}, 1
                initialized = True
                send({"method": "initialized", "params": {}})
                send({"id": 2, "method": "skills/list", "params": {
                    "cwds": [tempfile.gettempdir()], "forceReload": True}})
            elif message.get("id") == 2 and initialized:
                if "error" in message:
                    return {"status": "failed", "reason": "skills_list_error", "skill": SKILL_NAME}, 1
                result = message.get("result", {})
                data = result.get("data", []) if isinstance(result, dict) else []
                for group in data if isinstance(data, list) else []:
                    skills = group.get("skills", []) if isinstance(group, dict) else []
                    for skill in skills if isinstance(skills, list) else []:
                        if not isinstance(skill, dict) or skill.get("name") != SKILL_NAME:
                            continue
                        path = skill.get("path")
                        if (skill.get("enabled") is True and skill.get("scope") == "user"
                                and isinstance(path, str) and path
                                and Path(path).expanduser().resolve() == expected.resolve()):
                            # Do not disclose unrelated skills or app-server configuration.
                            return {"status": "verified", "name": SKILL_NAME, "enabled": True,
                                    "scope": "user", "path": str(expected.resolve())}, 0
                return {"status": "failed", "reason": "enabled_user_skill_not_found",
                        "skill": SKILL_NAME}, 1
    except (OSError, ValueError, TypeError):
        return {"status": "failed", "reason": "codex_protocol_error", "skill": SKILL_NAME}, 1
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            finally:
                if reader is not None:
                    reader.join(timeout=1)
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()


def main() -> int:
    codex = shutil.which("codex")
    if codex is None:
        print(json.dumps({"status": "unverified", "reason": "codex_not_found",
                          "skill": SKILL_NAME}))
        return 0
    expected = Path(__file__).resolve().parents[1] / "SKILL.md"
    result, code = _check_skill(codex, expected)
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
