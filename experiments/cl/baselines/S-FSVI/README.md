# Continual Learning via Sequential Function-Space Variational Inference (S-FSVI)

Contains the adapted implementation for "**_Continual Learning via Sequential Function-Space Variational Inference_**"; Tim G. J. Rudner, Freddie Bickford Smith, Qixuan Feng, Yee Whye Teh, Yarin Gal. **ICML 2022**, to obtain the results with Split-FashionMNIST (SH). This was used to run both models using S-FSVI and VCL.

## Installation

To install requirements:

```bash
conda env update -f environment.yml
conda activate fsvi
pip install -e .
```
This environment includes all necessary dependencies.

## Run Experiments
### S-MNIST 40pts./task
#### S-FSVI
``` bash
```
#### VCL (with random coresets)
``` bash
python baselines/vcl/run_vcl.py  --dataset smnist_sh --n_epochs 100 --batch_size 256 --hidden_size 256 --n_layers 2  --select_method random_choice --n_permuted_tasks 10 --logroot ablation --subdir vcl_smnist200 --n_coreset_inputs_per_task 40 --seed=<SEED>
```


### S-MNIST 200pts./task
#### S-FSVI
``` bash
```
#### VCL (with random coresets)
``` bash
python ./baselines/vcl/run_vcl.py --dataset smnist_sh --n_epochs 100 --batch_size 256 --hidden_size 256 --n_layers 2 \
  --select_method random_choice --n_permuted_tasks 10 --logroot ablation --subdir vcl_smnist200 --n_coreset_inputs_per_task 200 \
  --seed=<SEED>
```

### S-FMNIST 200pts./task
#### S-FSVI
``` bash
```
#### VCL (with random coresets)
``` bash
  python baselines/vcl/run_vcl.py  --dataset sfashionmnist_sh --n_epochs 100 --batch_size 256 --hidden_size 256 --n_layers 2  --select_method random_choice --n_permuted_tasks 10 --logroot ablation --subdir vcl_sfmnist --n_coreset_inputs_per_task 200 --seed=<SEED>
```


### P-MNIST 200pts./task
#### S-FSVI
``` bash
```

#### VCL (with random coresets)
``` bash
python baselines/vcl/run_vcl.py  --dataset pmnist --n_epochs 100 --batch_size 256 --hidden_size 100 --n_layers 2  --select_method random_choice --n_permuted_tasks 10 --n_coreset_inputs_per_task 200 --seed=<SEED>
```


## Citation

```
@InProceedings{rudner2022continual,
      author={Tim G. J. Rudner and Freddie Bickford Smith and Qixuan Feng and Yee Whye Teh and Yarin Gal},
      title = {{C}ontinual {L}earning via {S}equential {F}unction-{S}pace {V}ariational {I}nference},
      booktitle ={Proceedings of the 39th International Conference on Machine Learning},
      year = {2022},
      series ={Proceedings of Machine Learning Research},
      publisher ={PMLR},
}
```
