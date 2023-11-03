.PHONY: run clean paper

FILENAME=main
PAPER_DIR=paper
AUX_DIR=.aux

VENV = .venv
# VENV = .direnv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run: $(VENV)/bin/activate
	$(PYTHON) src/train.py

$(VENV)/bin/activate: setup.py
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install laplace-torch==0.1a2
	$(PIP) install -e ".[experiments, dev]"
