.PHONY: install test smoke validate-paper-data plot-paper-data docs-check compile

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

compile:
	$(PYTHON) -m compileall -q src tests scripts examples

test: compile
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/run_smoke.py

validate-paper-data:
	$(PYTHON) scripts/validate_reference_data.py

plot-paper-data:
	$(PYTHON) scripts/plot_all_paper_data.py

docs-check:
	$(PYTHON) scripts/check_tutorials.py
