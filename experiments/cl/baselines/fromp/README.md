# FROMP
Contains the adapted code for the NeurIPS 2020 paper by Pan et al., "[Continual Deep Learning by Functional Regularisation of Memorable Past](https://arxiv.org/abs/2004.14070)", to run Single Head (SH) experiments on S-MNIST and S-FMNIST

## Install the FROMP environment
``` sh
conda create env -n "fromp_env" python=3.7
conda activate fromp_env
pip install -r fromp_requirements.txt
```

## Run experiments
### S-MNIST 40pts./task
``` bash
python main_splitsmnist_SH.py --batch_size=32 --num_epochs=10 --select_method=random --num_points=40 --seed=<SEED>
```

### S-MNIST 200pts./task
``` bash
python main_splitsmnist_SH.py --batch_size=32 --num_epochs=10 --select_method=random --num_points=200 --seed=<SEED>
```

### S-FMNIST 200pts./task
``` bash
python main_splitfmnist.py --batch_size=32 --num_epochs=10 --select_method=random --num_points=200 --seed=<SEED>
```

### P-MNIST 200pts./task
``` bash
python main_permutedmnist.py --batch_size=128 --num_epochs=10 --select_method=lambda_descend --num_points=200 --seed=<SEED>
```

## Citation

```
@article{pan2020continual,
  title = {Continual Deep Learning by Functional Regularisation of Memorable Past},
  author = {Pan, Pingbo and Swaroop, Siddharth and Immer, Alexander and Eschenhagen, Runa and Turner, Richard E and Khan, Mohammad Emtiyaz},
  journal = {Advances in neural information processing systems},
  year = {2020}
}
```
