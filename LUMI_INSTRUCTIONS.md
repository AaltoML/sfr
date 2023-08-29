# Instuctions for setting up on Lumi

Create container with `torch==2.0.0+rocm5.4.2` using
```sh
SFR_CONTAINER_DIR=/scratch/project_462000217/sfr
mkdir  $SFR_CONTAINER_DIR
module load LUMI/22.08
module load lumi-container-wrapper
conda-containerize new --mamba --prefix $SFR_CONTAINER_DIR sfr-env-amd.yaml
conda-containerize update $SFR_CONTAINER_DIR --post-install post-install-amd.txt
```
Add container to path
```sh
SFR_CONTAINER_DIR=/scratch/project_462000217/sfr
export PATH="$SFR_CONTAINER_DIR/bin:$PATH"
```


``` sh 
mkdir  venv
module load LUMI/22.08
module load lumi-container-wrapper
conda-containerize new --mamba --prefix venv sfr-env-amd.yaml
conda-containerize update venv --post-install post-install-amd.txt
```