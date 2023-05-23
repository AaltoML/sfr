# UCI classification experiments
Instructions for reproducing the UCI classification benchmark results in the paper.


# Running experiments
Go the the UCI experiment folder
``` sh
cd experiments/sl/uci
```

Run the UCI classification experiment for the australian dataset and a single value of prior precision:
``` sh
python classification.py -d australian --root_dir . --seed 711 --name sparse_64 --n_inducing 64 --double
```
Where dataset is selected from 