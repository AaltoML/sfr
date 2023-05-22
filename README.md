# Sparse Function-space Representation of Neural Networks
Code accompanying NeurIPS 2023 submission Sparse Function-space Representation of Neural Networks.

## Install

### Option 1: using make
Make a virtual environment and install the dependencies with:
```sh
make .venv/bin/activate
```
Activate the environment with:
``` sh
source .venv/bin/activate
```
The [Makefile](Makefile) just runs:
``` sh
python -m venv $(VENV)
python -m pip install --upgrade pip
pip install laplace-torch==0.1a2
pip install -e ".[experiments, dev]"
```

### Option 2: pip install
Alternatively, manually install the dependencies with:
``` sh
pip install laplace-torch==0.1a2
pip install -r requirements.txt
```
We install `laplace-torch` separately due to version conflicts with `backpacpk-for-pytorch`.

## Useage
See the [notebooks/](notebooks/) for using our code for both regression and classification.

### Reproducing experiments
See [experiments/README.md](experiments/README.md) for details of how to reproduce the experiments in the paper.

### Example
See the [notebooks](./src/notebooks) for how to use our code.
Here's a short example:
```python
import src
import torch

torch.set_default_dtype(torch.float64)

def func(x, noise=True):
    return torch.sin(x * 5) / x + torch.cos(x * 10)

# Toy data set
X_train = torch.rand((100, 1)) * 2
Y_train = func(X_train, noise=True)
data = (X_train, Y_train)

# Training config
width = 64
num_epochs = 1000
batch_size = 16
learning_rate = 1e-3
delta = 0.00005  # prior precision
data_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*data), batch_size=batch_size
)

# Create a neural network
network = torch.nn.Sequential(
    torch.nn.Linear(1, width),
    torch.nn.Tanh(),
    torch.nn.Linear(width, width),
    torch.nn.Tanh(),
    torch.nn.Linear(width, 1),
)

# Instantiate SFR (handles NN training/prediction as they're coupled via the prior/likelihood)
sfr = src.sfr.SFR(
    network=network,
    prior=src.priors.Gaussian(params=network.parameters, delta=delta),
    likelihood=src.likelihoods.Gaussian(sigma_noise=2),
    output_dim=1,
    num_inducing=32,
    dual_batch_size=None, # this reduces the memory required for computing dual parameters
    jitter=1e-4,
)

sfr.train()
optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=learning_rate)
for epoch_idx in range(num_epochs):
    for batch_idx, batch in enumerate(data_loader):
        x, y = batch
        loss = sfr.loss(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

sfr.set_data(data) # This builds the dual parameters

# Make predictions in function space
X_test = torch.linspace(-0.7, 3.5, 300, dtype=torch.float64).reshape(-1, 1)
f_mean, f_var = sfr.predict_f(X_test)

# Make predictions in output space
y_mean, y_var = sfr.predict(X_test)
```

## Citation
```bibtex
@article{XXX,
    title={Sparse Function-space Representation of Neural Networks,
    author={},
    journal={},
    year={2023}
}
```
