"""Inventory installed distribution metadata without recording local install paths.

Run with the project virtual environment after frozen dependency installation:
    .venv/bin/python docs/generate_dependency_licenses.py --checked-date 2026-11-13

This records publisher declarations and license-file checksums, not legal clearance.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path


def inventory() -> list[dict]:
    # An empty bootstrap environment must never appear as a completed audit.
    metadata.version("funasr")
    result = []
    for distribution in metadata.distributions():
        info = distribution.metadata
        declared = (info.get("License") or "").strip().splitlines()
        expression = info.get("License-Expression")
        classifiers = [value for value in info.get_all("Classifier", [])
                       if value.startswith("License ::")]
        files = []
        for item in distribution.files or []:
            if item.name.lower().startswith(("license", "copying", "notice")):
                source = distribution.locate_file(item)
                if source.is_file():
                    files.append({"distribution_relative_path": str(item),
                                  "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        declaration = declared[0][:160] if declared else None
        unknown = declaration is None or declaration.lower() in {"unknown", "n/a", "unspecified"}
        needs_review = not expression and unknown and not classifiers
        result.append({"name": info.get("Name"), "version": distribution.version,
                       "license_expression": expression,
                       "license_metadata_first_line": declaration,
                       "license_classifiers": classifiers,
                       "publisher_project_urls": info.get_all("Project-URL", []),
                       "license_file_checksums": files,
                       "needs_manual_review": needs_review})
    return sorted(result, key=lambda item: (item["name"] or "").lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checked-date", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("dependency-licenses.json"))
    args = parser.parse_args()
    rows = inventory()
    report = {"checked_date": args.checked_date, "scope": "installed Python distributions",
              "method": "importlib.metadata declarations and distribution-relative license checksums",
              "limitation": "Publisher metadata is not legal clearance. Model weights and external FFmpeg have separate terms; review distribution license files for redistribution.",
              "distribution_count": len(rows),
              "manual_review_count": sum(item["needs_manual_review"] for item in rows),
              "distributions": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"distribution_count": report["distribution_count"],
                      "manual_review_count": report["manual_review_count"]}))


if __name__ == "__main__":
    main()
