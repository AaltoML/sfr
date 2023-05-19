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
	$(PIP) install -e ".[experiments, dev]"
	$(PIP) install laplace-torch==0.1a2
	$(PIP) install torch==2.0.0 torchvision==0.15.1

# $(VENV)/bin/activate: requirements.txt
# 	python3 -m venv $(VENV)
# 	$(PIP) install -r requirements.txt

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
