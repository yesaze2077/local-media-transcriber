.PHONY: install test lint build doctor smoke

install:
	bash install.sh

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check src tests examples

build:
	.venv/bin/python -m build

doctor:
	.venv/bin/transcribe doctor

smoke:
	@smoke_dir=$$(mktemp -d /tmp/lmt-smoke.XXXXXXXX); trap 'rm -rf -- "$$smoke_dir"' EXIT; .venv/bin/python examples/smoke.py --work-dir "$$smoke_dir"
