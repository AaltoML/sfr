# Document Title


Create container
```sh
NN2SVGP_CONTAINER_DIR=/scratch/project_462000217/nn2svgp
mkdir  $NN2SVGP_CONTAINER_DIR
module load LUMI/22.08
module load lumi-container-wrapper
conda-containerize new --prefix $NN2SVGP_CONTAINER_DIR nn2svgp-env.yml
```

Add container to path
```sh
NN2SVGP_CONTAINER_DIR=/scratch/project_462000217/nn2svgp
export PATH="$NN2SVGP_CONTAINER_DIR/bin:$PATH"
```

```sh
conda-containerize update $NN2SVGP_CONTAINER_DIR --post-install post_install.txt
'pip install -e ".[experiments, dev]"'
```
