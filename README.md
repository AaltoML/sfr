# Sparse Function-space Representation of Neural Networks
PyTorch implementation of Sparse Function-space Representation of Neural Networks.


<table>
    <tr>
        <td>
            <strong>Sparse Function-space Representation of Neural Networks</strong><br>
            Aidan Scannell*, Riccardo Mereu*, Paul Chang, Ella Tamir, Joni Pajarinen, Arno Solin<br>
            <strong>ICML 2023 Workshop on Duality Principles for Modern Machine Learning</strong><br>
            <a href="https://arxiv.org/abs/2309.02195"><img alt="Paper" src="https://img.shields.io/badge/-Paper-gray"></a>
            <a href="https://github.com/aidanscannell/sfr"><img alt="Code" src="https://img.shields.io/badge/-Code-gray" ></a>
            <a href="https://aaltoml.github.io/sfr"><img alt="Website" src="https://img.shields.io/badge/-Website-gray" ></a>
        </td>
</table>

## Install

Make a virtual environment and install the dependencies with:
``` sh
python -m venv sfr-venv
source sfr-venv/bin/activate
pip install -e ".[experiments, dev]"
```

## Useage
See the [notebooks](./notebooks) for how to use our code for both regression and classification.

### Example
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
sfr = src.SFR(
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
@inproceedings{scannellSparse2023,
  title           = {Sparse Function-space Representation of Neural Networks},
  maintitle       = {ICML 2023 Workshop on Duality Principles for Modern Machine Learning},
  author          = {Aidan Scannell and Riccardo Mereu and Paul Chang and Ella Tami and Joni Pajarinen and Arno Solin},
  year            = {2023},
  month           = {7},
}
```
