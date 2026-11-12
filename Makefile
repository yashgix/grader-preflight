OLD ?= .tau2/pre-v1.0.1/venv/bin/python
NEW ?= .tau2/v1.0.1/venv/bin/python
export TAU2_OLD_PYTHON := $(abspath $(OLD))
export TAU2_NEW_PYTHON := $(abspath $(NEW))
export LITELLM_LOCAL_MODEL_COST_MAP := True

setup:
	./scripts/setup-tau2.sh

lint:
	ruff check . && ruff format --check .

test:
	$(OLD) -m unittest discover -s tests -t .
	$(NEW) -m unittest discover -s tests -t .

check: lint test

example:
	$(NEW) examples/report_blind_spot.py

phase1:
	$(OLD) experiment.py --phase 1 --out runs/phase1/metrics.json

phase3:
	$(OLD) experiment.py --phase 3 --out runs/phase3/metrics.json

site:
	python3 scripts/build_site.py

.PHONY: setup lint test check example phase1 phase3 site
