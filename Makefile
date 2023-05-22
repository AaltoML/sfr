.PHONY: run clean paper

FILENAME=main
PAPER_DIR=paper
AUX_DIR=.aux

VENV = .venv
# VENV = .direnv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

${FILENAME}.pdf: media
	cd paper && latexmk
	cd paper && mv ${AUX_DIR}/${FILENAME}.pdf ${FILENAME}.pdf

paper: ${FILENAME}.pdf

media: figures tables

all: run paper

run: $(VENV)/bin/activate
	$(PYTHON) src/train.py

$(VENV)/bin/activate: setup.py
	python3 -m venv $(VENV)
	python3 -m pip install --upgrade pip
	$(PIP) install laplace-torch==0.1a2
	$(PIP) install -e ".[experiments, dev]"

figures: $(VENV)/bin/activate
	$(PYTHON) src/figures.py --save_dir="./${PAPER_DIR}/figs"

tables: $(VENV)/bin/activate
	$(PYTHON) src/tables.py --save_dir="./${PAPER_DIR}/tables"

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm ${PAPER_DIR}/${AUX_DIR}

submission:
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=1 -dLastPage=13 -sOutputFile=submission.pdf ${FILENAME}.pdf

appendix: ${FILENAME}.pdf
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=14 -sOutputFile=supplement.pdf ${FILENAME}.pdf

supplement:
	rm -rf supplement
	mkdir supplement

	# make appendix
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=14 -sOutputFile=supplement/supplement.pdf paper/main.pdf

	# make code directory
	mkdir supplement/code
	cp Makefile supplement/code
	cp README.md supplement/code
	cp LICENSE supplement/code
	cp requirements.txt supplement/code
	cp setup.py supplement/code

	# SFR stuff in src
	mkdir supplement/code/src
	cp src/*.py supplement/code/src/

	# Experiments
	mkdir supplement/code/experiments
	cp experiments/*.py supplement/code/experiments/
	cp experiments/README.md supplement/code/experiments/


	# SL stuff
	mkdir supplement/code/experiments/sl
	cp experiments/sl/*.py supplement/code/experiments/sl/
	cp -r experiments/sl/configs supplement/code/experiments/sl/configs/
	rm -rf supplement/code/experiments/sl/configs/hydra/launcher/lumi.yaml
	cp -r experiments/sl/configs supplement/code/experiments/sl/configs
	cp -r experiments/sl/bnn_predictive supplement/code/experiments/sl/bnn_predictive
	cp experiments/sl/README.md supplement/code/experiments/sl/


	# RL stuff
	mkdir supplement/code/experiments/rl
	mkdir supplement/code/experiments/rl/utils
	mkdir supplement/code/experiments/rl/agents
	mkdir supplement/code/experiments/rl/models
	mkdir supplement/code/experiments/rl/models/transitions
	mkdir supplement/code/experiments/rl/models/rewards
	cp experiments/rl/*.py supplement/code/experiments/rl/
	cp experiments/rl/README.md supplement/code/experiments/rl/
	cp experiments/rl/utils/*.py supplement/code/experiments/rl/utils/
	cp experiments/rl/agents/*.py supplement/code/experiments/rl/agents/
	cp experiments/rl/models/*.py supplement/code/experiments/rl/models
	cp experiments/rl/models/transitions/*.py supplement/code/experiments/rl/models/transitions
	cp experiments/rl/models/rewards/*.py supplement/code/experiments/rl/models/rewards
	cp -r experiments/rl/configs supplement/code/experiments/rl/configs/
	rm -rf supplement/code/experiments/rl/configs/hydra/launcher/lumi.yaml # contains lumi project code
	rm supplement/code/experiments/rl/plot_rl_figure.py # contains aalto-ml in wandb name

	# CL stuff
	mkdir supplement/code/experiments/cl

	# Notebooks
	mkdir supplement/code/notebooks
	cp -r notebooks/1D-regression-and-dual-updates.ipynb supplement/code/notebooks/
