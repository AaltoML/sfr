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
	$(PYTHON) -m pip install --upgrade pip
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
	cp Makefile_nosupplement supplement/code/Makefile
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
	# cp -r experiments/sl/configs supplement/code/experiments/sl/configs
	mkdir supplement/code/experiments/sl/bnn_predictive
	cp -r experiments/sl/bnn_predictive/preds supplement/code/experiments/sl/bnn_predictive/preds
	mkdir supplement/code/experiments/sl/uci
	mkdir supplement/code/experiments/sl/uci/results
	cp -r experiments/sl/uci/data supplement/code/experiments/sl/uci/data
	cp experiments/sl/uci/README.md supplement/code/experiments/sl/uci
	cp experiments/sl/uci/classification.py supplement/code/experiments/sl/uci
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
	rm -rf supplement/code/experiments/rl/configs/hydra/launcher/lumi.yaml
	rm supplement/code/experiments/rl/plot_rl_figure.py

	# CL stuff
	mkdir supplement/code/experiments/cl
	mkdir supplement/code/experiments/cl/backbone
	mkdir supplement/code/experiments/cl/backbone/utils
	mkdir supplement/code/experiments/cl/datasets/
	mkdir supplement/code/experiments/cl/datasets/transforms
	mkdir supplement/code/experiments/cl/datasets/utils
	mkdir supplement/code/experiments/cl/models/
	mkdir supplement/code/experiments/cl/models/utils
	mkdir supplement/code/experiments/cl/utils

	cp experiments/cl/gem_license supplement/code/experiments/cl/gem_license
	cp experiments/cl/LICENSE supplement/code/experiments/cl/LICENSE
	cp experiments/cl/README.md supplement/code/experiments/cl/README.md
	cp experiments/cl/backbone/*.py supplement/code/experiments/cl/backbone
	cp experiments/cl/backbone/utils/*.py supplement/code/experiments/cl/backbone/utils
	cp experiments/cl/datasets/*.py supplement/code/experiments/cl/datasets/
	cp experiments/cl/datasets/transforms/*.py supplement/code/experiments/cl/datasets/transforms
	cp experiments/cl/datasets/utils/*.py supplement/code/experiments/cl/datasets/utils
	cp experiments/cl/models/*.py supplement/code/experiments/cl/models/	
	cp experiments/cl/models/utils/*.py supplement/code/experiments/cl/models/utils
	cp experiments/cl/utils/*.py supplement/code/experiments/cl/utils

	mkdir supplement/code/experiments/cl/baselines/
	mkdir supplement/code/experiments/cl/baselines/fromp
	mkdir supplement/code/experiments/cl/baselines/fromp/data
	cp experiments/cl/baselines/fromp/*.py supplement/code/experiments/cl/baselines/fromp
	cp experiments/cl/baselines/fromp/data/mnist.pkl.gz supplement/code/experiments/cl/baselines/fromp/data

	mkdir supplement/code/experiments/cl/baselines/S-FSVI
	cp experiments/cl/baselines/S-FSVI/*.py supplement/code/experiments/cl/baselines/S-FSVI
	cp experiments/cl/baselines/S-FSVI/environment.yml experiments/cl/baselines/S-FSVI/LICENSE experiments/cl/baselines/S-FSVI/README.md supplement/code/experiments/cl/baselines/S-FSVI

	mkdir supplement/code/experiments/cl/baselines/S-FSVI/baselines
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/baselines/vcl
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/baselines/vcl/alg
	cp experiments/cl/baselines/S-FSVI/baselines/vcl/*.py supplement/code/experiments/cl/baselines/S-FSVI/baselines/vcl
	cp experiments/cl/baselines/S-FSVI/baselines/vcl/README.md supplement/code/experiments/cl/baselines/S-FSVI/baselines/vcl
	cp experiments/cl/baselines/S-FSVI/baselines/vcl/alg/*.py supplement/code/experiments/cl/baselines/S-FSVI/baselines/vcl/alg

	mkdir supplement/code/experiments/cl/baselines/S-FSVI/benchmarking
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/benchmarking/data_loaders
	cp experiments/cl/baselines/S-FSVI/benchmarking/*.py supplement/code/experiments/cl/baselines/S-FSVI/benchmarking/
	cp experiments/cl/baselines/S-FSVI/benchmarking/data_loaders/*.py supplement/code/experiments/cl/baselines/S-FSVI/benchmarking/data_loaders/

	mkdir supplement/code/experiments/cl/baselines/S-FSVI/sfsvi
	cp experiments/cl/baselines/S-FSVI/sfsvi/*.py supplement/code/experiments/cl/baselines/S-FSVI/sfsvi
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/models
	cp experiments/cl/baselines/S-FSVI/sfsvi/models/*.py supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/models
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/general_utils
	cp experiments/cl/baselines/S-FSVI/sfsvi/general_utils/*.py supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/general_utils
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/fsvi_utils
	cp experiments/cl/baselines/S-FSVI/sfsvi/fsvi_utils/*.py supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/fsvi_utils/
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/fsvi_utils/coreset
	cp experiments/cl/baselines/S-FSVI/sfsvi/fsvi_utils/coreset/*.py supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/fsvi_utils/coreset
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/exps
	cp experiments/cl/baselines/S-FSVI/sfsvi/exps/*.py supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/exps
	mkdir supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/exps/utils
	cp experiments/cl/baselines/S-FSVI/sfsvi/exps/utils/*.py supplement/code/experiments/cl/baselines/S-FSVI/sfsvi/exps/utils


	# Notebooks
	mkdir supplement/code/notebooks
	mkdir supplement/code/notebooks/data
	cp -r notebooks/1D-regression-and-dual-updates.ipynb supplement/code/notebooks/
	cp -r notebooks/classification-banana-data-set.ipynb supplement/code/notebooks/
	cp notebooks/data/banana_X_train notebooks/data/banana_Y_train supplement/code/notebooks/data
	
	rm -f supplement.zip
	zip supplement.zip 'supplement/supplement.pdf'
	zip -r supplement.zip supplement/code -i '*.py'
	zip -r supplement.zip supplement/code -i '*.txt'
	zip -r supplement.zip supplement/code -i '*.md'
	zip -r supplement.zip supplement/code -i '*.ipynb'
	zip -r supplement.zip supplement/code -i '*.yaml'
	zip -r supplement.zip supplement/code -i '*.yml'
	zip -r supplement.zip supplement/code -i '*.csv'
	zip -r supplement.zip supplement/code -i '*.data'
	zip -r supplement.zip supplement/code -i '*.names'
	zip -r supplement.zip supplement/code -i '*.spy'
	zip -r supplement.zip supplement/code -i '*.gz'
	zip supplement.zip supplement/code/notebooks/data/banana_X_train
	zip supplement.zip supplement/code/notebooks/data/banana_Y_train
	# zip -r supplement.zip supplement/code -i 'LICENSE'


