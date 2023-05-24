# UCI classification experiments
Instructions for reproducing the UCI classification benchmark results in the paper.


# Running experiments
Go the the UCI experiment folder
``` sh
cd experiments/sl/uci
```
Run the UCI classification experiment for the australian dataset and an array of prior precisions:
``` sh
python classification.py -d australian --root_dir . --seed 711 --name sparse_64 --n_inducing 64 --double
```
The script writes a result pickle to experiments/sl/uci/results/test. The pickle object is a list of length 10 (for the number of prior precisions), and each of the 10 list items is a dictionary. For any of the dictionaries (called results), query
``` sh
val_nll = results['results']['valid_nll_svgp_ntk']
```
for the validation negative log-likelihood. 