.PHONY: help install lint typecheck test check preflight smoke core full plan analyze plots report dryrun clean

UV      ?= uv
RUN     ?= $(UV) run
PROFILE ?= core
RESULTS ?= results
FIGURES ?= figures

help:
	@echo "quant-bench-5070ti"
	@echo "  make install     - create the venv and install dev extras"
	@echo "  make check       - lint + typecheck + tests (no GPU required)"
	@echo "  make preflight   - environment probe, writes results/environment.json"
	@echo "  make dryrun      - plan + execute the smoke profile with the mock engine (no GPU)"
	@echo "  make smoke|core|full - plan -> run -> analyze -> plot -> report (GPU required)"
	@echo "  make plots       - regenerate every figure from committed results (no GPU)"

install:
	$(UV) sync --extra dev

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest

check: lint typecheck test

preflight:
	$(RUN) quantbench preflight --out $(RESULTS)/environment.json

dryrun:
	$(RUN) quantbench plan --profile smoke --out plan.smoke.json
	$(RUN) quantbench run --plan plan.smoke.json --dry-run

smoke: PROFILE=smoke
smoke: bench

core: PROFILE=core
core: bench

full: PROFILE=full
full: bench

bench:
	$(RUN) quantbench plan --profile $(PROFILE) --out plan.$(PROFILE).json
	$(RUN) quantbench run --plan plan.$(PROFILE).json --resume
	$(MAKE) analyze plots report

plan:
	$(RUN) quantbench plan --profile $(PROFILE) --out plan.$(PROFILE).json

analyze:
	$(RUN) quantbench analyze --results $(RESULTS) --out $(RESULTS)/results.parquet

plots:
	$(RUN) quantbench plot --results $(RESULTS)/results.parquet --out $(FIGURES)

report:
	$(RUN) quantbench report --results $(RESULTS)/results.parquet

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache plan.*.json
