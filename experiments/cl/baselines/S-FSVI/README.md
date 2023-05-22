# Continual Learning via Sequential Function-Space Variational Inference (S-FSVI)

Contains the adapted implementation for "**_Continual Learning via Sequential Function-Space Variational Inference_**"; Tim G. J. Rudner, Freddie Bickford Smith, Qixuan Feng, Yee Whye Teh, Yarin Gal. **ICML 2022**, to obtain the results with Split-FashionMNIST (SH).



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
```


### S-MNIST 200pts./task
#### S-FSVI
``` bash
```
#### VCL (with random coresets)
``` bash
```

### S-FMNIST 200pts./task
#### S-FSVI
``` bash
```
#### VCL (with random coresets)
``` bash
```


### P-MNIST 200pts./task
#### S-FSVI
``` bash
```
#### VCL (with random coresets)
``` bash
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

Please cite our paper if you use this code in your own work.
