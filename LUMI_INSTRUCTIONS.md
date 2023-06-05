# Instuctions for setting up on Lumi

Create container with `torch==2.0.0+rocm5.4.2` using
```sh
SFR_CONTAINER_DIR=/scratch/project_462000217/sfr
mkdir  $SFR_CONTAINER_DIR
module load LUMI/22.08
module load lumi-container-wrapper
conda-containerize new --mamba --prefix $SFR_CONTAINER_DIR sfr-env.yaml
```
Add container to path
```sh
SFR_CONTAINER_DIR=/scratch/project_462000217/sfr
export PATH="$SFR_CONTAINER_DIR/bin:$PATH"
```
Comment out `torch` and `torchvision` in `setup.py`. Then install our package with
```sh
conda-containerize update $SFR_CONTAINER_DIR --post-install post_install.txt
```


