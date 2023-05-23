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
python cli.py cl_v2 --data_training continual_learning_smnist_sh_40 --model_type fsvi_mlp --optimizer adam --momentum 0.0 --momentum_var 0.0 \
    --architecture fc_256_256 --activation relu --prior_mean 0.0 --prior_cov 0.001 --prior_covs 0.0 --prior_type bnn_induced --epochs 10 \
    --start_var_opt 0 --batch_size 128 --learning_rate 0.0005 --learning_rate_var 0.001 --dropout_rate 0.0 --regularization 0.0 --context_points 0 \
    --n_marginals 1 --n_condition 0 --context_point_type uniform_rand --kl_scale equal --td_prior_scale 0.0 --feature_update 1 --n_samples 5 \
    --n_samples_eval 5 --tau 1.0 --noise_std 1.0 --ind_lim ind_-1_1 --name --init_logvar 0.0 0.0 --init_logvar_lin 0.0 0.0 --init_logvar_conv 0.0 0.0 \
    --perturbation_param 0.01 --logroot sfsvi --subdir smnist_40 --n_context_points 40 --context_points_bound 0.0 1.0 --context_points_add_mode 0 --logging 1 \
    --coreset random --coreset_entropy_mode soft_highest --coreset_entropy_offset 0.0 --coreset_kl_heuristic lowest --coreset_kl_offset 0.0 --coreset_elbo_heuristic lowest \
    --coreset_elbo_offset 0.0 --coreset_entropy_n_mixed 1 --augment_mode constant --loss_type 1 --seed=<SEED>
```
#### VCL (with random coresets)
``` bash
python baselines/vcl/run_vcl.py --dataset smnist_sh_40 --n_epochs 100 --batch_size 256 --hidden_size 256 --n_layers 2 \ 
--select_method random_choice --logroot vcl --subdir vcl_smnist200 --n_coreset_inputs_per_task 40 \
 --seed=<SEED>
```


### S-MNIST 200pts./task
#### S-FSVI
``` bash
python cl_v2 --data_training continual_learning_smnist_sh --model_type fsvi_mlp --optimizer adam --momentum 0.0 --momentum_var 0.0 \
--architecture fc_256_256 --activation relu --prior_mean 0.0 --prior_cov 0.001 --prior_covs 0.0 --prior_type bnn_induced --epochs 80 \
--start_var_opt 0 --batch_size 128 --learning_rate 0.0005 --learning_rate_var 0.001 --dropout_rate 0.0 --regularization 0.0 --context_points 0 \
--n_marginals 1 --n_condition 0 --context_point_type uniform_rand --kl_scale equal --td_prior_scale 0.0 --feature_update 1 --n_samples 5 \
--n_samples_eval 5 --tau 1.0 --noise_std 1.0 --ind_lim ind_-1_1 --name --init_logvar 0.0 0.0 --init_logvar_lin 0.0 0.0 --init_logvar_conv 0.0 0.0 \
--perturbation_param 0.01 --logroot sfsvi --subdir smnist_200 --n_context_points 40 --context_points_bound 0.0 1.0 --context_points_add_mode 0 --logging 1 \
--coreset random --coreset_entropy_mode soft_highest --coreset_entropy_offset 0.0 --coreset_kl_heuristic lowest --coreset_kl_offset 0.0 --coreset_elbo_heuristic lowest \
--coreset_elbo_offset 0.0 --coreset_entropy_n_mixed 1 --augment_mode constant --loss_type 1 --seed=<SEED>
```

#### VCL (with random coresets)
``` bash
python ./baselines/vcl/run_vcl.py --dataset smnist_sh_40 --n_epochs 100 --batch_size 256 --hidden_size 256 --n_layers 2 \
  --select_method random_choice --logroot vcl --subdir vcl_smnist200 --n_coreset_inputs_per_task 200 \
  --seed=<SEED>
```

### S-FMNIST 200pts./task
#### S-FSVI
``` bash
python cli.py cl_v2 --data_training continual_learning_sfashionmnist_sh --model_type fsvi_mlp --optimizer adam --momentum 0.0 --momentum_var 0.0 \
--architecture fc_256_256 --activation relu --prior_mean 0.0 --prior_cov 0.001 --prior_covs 0.0 --prior_type bnn_induced --epochs 60 \
--start_var_opt 0 --batch_size 128 --learning_rate 0.0005 --learning_rate_var 0.001 --dropout_rate 0.0 --regularization 0.0 --context_points 0 \
--n_marginals 1 --n_condition 0 --context_point_type uniform_rand --kl_scale equal --td_prior_scale 0.0 --feature_update 1 --n_samples 5 \
--n_samples_eval 5 --tau 1.0 --noise_std 1.0 --ind_lim ind_-1_1 --init_logvar 0.0 0.0 --init_logvar_lin 0.0 0.0 --init_logvar_conv 0.0 0.0 \
--perturbation_param 0.01 --logroot sfsvi --subdir sfmnist_200 --n_context_points 40 --context_points_bound 0.0 1.0 --context_points_add_mode 0 --logging 1 \
--coreset random --coreset_entropy_mode soft_highest --coreset_entropy_offset 0.0 --coreset_kl_heuristic lowest --coreset_kl_offset 0.0 --coreset_elbo_heuristic lowest \
--coreset_elbo_offset 0.0 --coreset_entropy_n_mixed 1  --augment_mode constant  --loss_type 1 --seed <SEED> 
```
#### VCL (with random coresets)
``` bash
  python baselines/vcl/run_vcl.py  --dataset sfashionmnist_sh --n_epochs 100 --batch_size 256 --hidden_size 256 --n_layers 2 \
    --select_method random_choice --logroot vcl --subdir vcl_sfmnist --n_coreset_inputs_per_task 200 \
    --seed=<SEED>
```


### P-MNIST 200pts./task
#### S-FSVI
``` bash
python cl_v2 --data_training continual_learning_pmnist  --model_type fsvi_mlp --optimizer adam --momentum 0.0 --momentum_var 0.0 \
 --architecture fc_100_100 --activation relu --prior_mean 0.0 --prior_cov 0.001 --prior_covs 0.0 --prior_type bnn_induced --epochs 10 \
 --start_var_opt 0 --batch_size 128 --learning_rate 0.0005 --learning_rate_var 0.001 --dropout_rate 0.0 --regularization 0.0 --context_points 0 \
 --n_marginals 1 --n_condition 0 --context_point_type uniform_rand --kl_scale equal --td_prior_scale 0.0 --feature_update 1 --n_samples 5 \
 --n_samples_eval 5 --tau 1.0 --noise_std 1.0 --ind_lim ind_-1_1 --logging_frequency 10 --figsize 10 4 --init_logvar 0.0 0.0 --init_logvar_lin 0.0 0.0 \
 --init_logvar_conv 0.0 0.0 --perturbation_param 0.01 --logroot sfsvi --subdir pmnist --n_context_points 40 --context_points_bound 0.0 1.0 --context_points_add_mode 0 --logging 1 \
 --coreset random --coreset_entropy_mode soft_highest --coreset_entropy_offset 0.0 --coreset_kl_heuristic lowest --coreset_kl_offset 0.0 --coreset_elbo_heuristic lowest \
 --coreset_elbo_offset 0.0 --coreset_entropy_n_mixed 1 --n_permuted_tasks 10 --augment_mode constant --loss_type 1 --seed=<SEED>

```

#### VCL (with random coresets)
``` bash
python baselines/vcl/run_vcl.py  --dataset pmnist --n_epochs 100 --batch_size 256 --hidden_size 100 --n_layers 2  \
--select_method random_choice --n_permuted_tasks 10 --n_coreset_inputs_per_task 200 --seed=<SEED>
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
