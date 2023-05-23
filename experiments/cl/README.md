# SFR for Continual Learning
This part of the repository contains the code to replicate the Continual learning experiments reported in the paper. This part of the repo is based on the [Mammoth library](https://github.com/aimagelab/mammoth) and allows to reproduce the results for  SFR, DER, Online-EWC, and SI. For running the experiments with the other baselines check [FROMP README](./baselines/fromp/README.md) and [S-SFVI/VCL README](./baselines/S-FSVI/README.md).


### S-MNIST 40pts./task
Commands to replicate the results with Split-MNIST using 40 points per task (i.e., a total of 200 points).
#### SFR
```bash
python ./utils/main.py --model=sfr --dataset=seq-mnist --lr=0.0003 --dual_batchsize=1000 \
	--batch_size=32 --buffer_size=200 --tau=1. --delta=0.0001 --optimizer=adam --n_epochs=1 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### DER
```bash
python ./utils/main.py --model=der --dataset=seq-mnist --lr=0.03 --batch_size=10 \
	--minibatch_size=10 --optimizer=sgd --n_epochs=1 --alpha=0.2 --buffer_size=200 \	
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT
```

#### Online-EWC
```bash
python ./utils/main.py --model=ewc_on --dataset=seq-mnist --lr=0.03 --batch_size=10 \
	--optimizer=sgd --n_epochs=1 --e_lambda=90 --gamma=1.0 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### SI
```bash
python ./utils/main.py --model=si --dataset=seq-mnist --lr=0.1 --batch_size=10 \
	--optimizer=sgd --n_epochs=1 --c=1.0 --xi=0.9 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

### S-MNIST 200pts./task
Commands to replicate the results with Split-MNIST using 200 points per task (i.e., a total of 1000 points).
#### SFR
```bash
python ./utils/main.py --model=sfr --dataset=seq-mnist --lr=0.0001 --dual_batchsize=1000 \
	--batch_size=32 --buffer_size=1000 --tau=0.5 --delta=1e-05 --optimizer=adam --n_epochs=5 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### DER
```bash
python ./utils/main.py --model=der --dataset=seq-mnist --lr=0.03 --batch_size=10 \
	--minibatch_size=10 --optimizer=sgd --n_epochs=1 --alpha=0.3 --buffer_size=1000 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### Online-EWC
```bash
python ./utils/main.py --model=ewc_on --dataset=seq-mnist --lr=0.03 --batch_size=10 \
	--optimizer=sgd --n_epochs=1 --e_lambda=90 --gamma=1.0 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### SI
```bash
python ./utils/main.py --model=si --dataset=seq-mnist --lr=0.1 --batch_size=10 \
	--optimizer=sgd --n_epochs=1 --c=1.0 --xi=0.9 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```


### S-FMNIST 200pts./task
Commands to replicate the results with Split-FashionMNIST using 200 points per task (i.e., a total of 1000 points).

#### SFR
```bash
python ./utils/main.py --model=sfr --dataset=seq-fmnist --lr=0.0003 --dual_batchsize=1000 \
	--batch_size=32 --buffer_size=1000 --tau=1.0 --delta=0.0001 --optimizer=adam --n_epochs=5 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### DER
```bash 
python ./utils/main.py --model=der --dataset=seq-fmnist --lr=0.03 --batch_size=10 \
	--minibatch_size=10 --optimizer=sgd --n_epochs=5 --alpha=0.3 --buffer_size=1000 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### Online-EWC
```bash
python ./utils/main.py --model=ewc_on --dataset=seq-fmnist --lr=0.03 --batch_size=10 \
	--optimizer=sgd --n_epochs=1 --e_lambda=90 --gamma=1.0 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### SI
```bash
python ./utils/main.py --model=si --dataset=seq-fmnist --lr=0.1 --batch_size=10 \
	--optimizer=sgd --n_epochs=1 --c=1.0 --xi=0.9 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

### P-MNIST 200pts./task
Commands to replicate the results with Permuted-MNIST using 200 points per task (i.e., a total of 2000 points).

#### SFR
```bash 
python ./utils/main.py  --model=sfr --dataset=perm-mnist --lr=0.0003 --dual_batchsize=1000 \
	--batch_size=64 --buffer_size=2000 --tau=1.0 --delta=0.0001 --optimizer=adam --n_epochs=10 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### DER
```bash
python ./utils/main.py --model=der --dataset=perm-mnist --lr=0.2 --batch_size=128 \
	--minibatch_size=128 --optimizer=sgd --n_epochs=1 --alpha=0.3 --buffer_size=2000 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### Online-EWC
```bash
python ./utils/main.py --model=ewc_on --dataset=perm-mnist --lr=0.1 --batch_size=128 \
	--optimizer=sgd --n_epochs=10 --e_lambda=0.7 --gamma=1.0 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```

#### SI
```bash
python ./utils/main.py --model=si --dataset=perm-mnist --lr=0.1 --batch_size=128 \
	--optimizer=sgd --n_epochs=10 --c=0.5 --xi=1.0 \
	--seed=<SEED> --wandb_entity=<WANDB_ENTITY> --wandb_project=<WANDB_PROJECT>
```
