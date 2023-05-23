# UCI classification experiments
Instructions for reproducing the UCI classification benchmark results in the paper.


# Setting up the environment
To be written
# Running experiments
Run the UCI classification experiment for a single dataset:
``` sh
python sl/uci/classification.py -d ${dataset} --root_dir . --seed ${SEED} --n_layers 2 --activation tanh --logd_min ${logmin} --refine 0 --name ${name} --n_inducing ${n_inducing} --double --res_folder ${res_folder}
```
