# Image classification experiments
Instructions for reproducing the image classification results in the paper.

# Running experiments
Run the MNIST image classification experiment with:
``` sh
python sl/train.py +experiment=fmnist
```
or run both of the image classification experiments with:
``` sh
python sl/train.py --multirun +experiment=mnist,fmnist
```
You can display the base config using:
``` shell
python train.py --cfg=job
```

# Reproducing tables
