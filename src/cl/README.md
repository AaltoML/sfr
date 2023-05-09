# Sparse Functional Regularizer

## Install the environment
```bash
conda create -n <env_name> python=3.9
conda activate <env_name>
pip install -r requirements.txt
```

## Launch SFR experiments
```bash
# Sequential FashionMNIST
python ./utils/main.py --model=sfr --dataset=seq-fmnist --lr=1e-4 --A_batchsize=1000 --batch_size=32 --buffer_size=200 --tau=5e-2 --delta=1e-5 --seed=66 --optimizer=adam 
```

```bash
python ./utils/main.py --model=sfr --dataset=seq-mnist --lr=3e-4 --A_batchsize=1000 --batch_size=32 --buffer_size=1000 --tau=5e-2 --delta=1e-4 --seed=66 --optimizer=adam

python ./utils/main.py --model=sfr --dataset=seq-mnist --lr=3e-4 --A_batchsize=1000 --batch_size=32 --buffer_size=200 --tau=5e-2 --delta=1e-4 --seed=66 --optimizer=adam
```

TODO: The following commands are not up to date
```bash
# seq-mnist -> 92% 
python ./utils/main.py --model=sfr --dataset=seq-mnist --lr=3e-4 --A_batchsize=1000 --batch_size=32 --buffer_size=1000 --tau=5e-2 --delta=1e-4 --seed=66

# seq-mnist -> 89% 
python ./utils/main.py --model=sfr --dataset=seq-mnist --lr=3e-4 --A_batchsize=200 --batch_size=32 --buffer_size=1000 --tau=5e-2 --delta=1e-4 --seed=66
```

```bash
# seq-cifar10
python ./utils/main.py --model=sfr --dataset=seq-cifar10 --lr=3e-4 --buffer_size=200 --tau=1e-2 --delta=1e-5 --seed=36
```


