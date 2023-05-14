# Instuctions for setting up on Lumi

Create container with `torch==2.0.0+rocm5.4.2` using
```sh
NN2SVGP_CONTAINER_DIR=/scratch/project_462000217/nn2svgp
mkdir  $NN2SVGP_CONTAINER_DIR
module load LUMI/22.08
module load lumi-container-wrapper
conda-containerize new --mamba --prefix $NN2SVGP_CONTAINER_DIR nn2svgp-env.yaml
```
Add container to path
```sh
NN2SVGP_CONTAINER_DIR=/scratch/project_462000217/nn2svgp
export PATH="$NN2SVGP_CONTAINER_DIR/bin:$PATH"
```
Comment out `torch` and `torchvision` in `setup.py`. Then install our package with
```sh
conda-containerize update $NN2SVGP_CONTAINER_DIR --post-install post_install.txt
```


