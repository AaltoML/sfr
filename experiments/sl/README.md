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
Make the relevant rows with:
``` shell
python tables/MNIST_table.py
```
``` shell
python tables/FMNIST_table.py
```
<!-- TODO make this produce all rows not just ours -->
