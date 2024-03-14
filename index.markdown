---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

# layout: home
layout: post
# title:  "Function-space Prameterization of Neural Networks for Sequential Learning"
title:  "SFR: Sparse Function-space Representation of Neural Networks"
date:   2023-11-04 14:36:41 +0200
categories: jekyll update
author: "<a href='https://www.aidanscannell.com/'>Aidan Scannell</a><sup>*</sup>, <a href='https://github.com/rm-wu'>Riccardo Mereu</a><sup>*</sup>, <a href='https://edchangy11.github.io/'>Paul Chang</a>, Ella Tamir, <a href='https://rl.aalto.fi/'>Joni Pajarinen</a>, <a href='https://users.aalto.fi/~asolin/'>Arno Solin</a>"
---
<a href="https://openreview.net/forum?id=2dhxxIKhqz&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)"><img alt="Conference Paper" src="https://img.shields.io/badge/Conference-paper-gray?logo=arxiv"></a>
<a href="https://arxiv.org/abs/2309.02195"><img alt="Workshop Paper" src="https://img.shields.io/badge/Workshop-paper-gray?logo=arxiv"></a>
<a href="https://github.com/AaltoML/sfr"><img alt="Code" src="https://img.shields.io/badge/-Code-gray?logo=github" ></a>
<a href="https://github.com/AaltoML/sfr-experiments"><img alt="Experiments" src="https://img.shields.io/badge/-Experiments-gray?logo=github" ></a>
<!-- <a href="https://scholar.google.fi/citations?view_op=view_citation&hl=en&user=piA0zS4AAAAJ&citation_for_view=piA0zS4AAAAJ:zYLM7Y9cAGgC"><img alt="Google Scholar" src="https://img.shields.io/badge/-Scholar-gray?logo=googlescholar" ></a> -->
<table>
    <tr>
        <td>
            <a href="https://openreview.net/forum?id=2dhxxIKhqz&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)">
              <strong >Function-space Parameterization of Neural Networks for Sequential Learning</strong><br>
            </a>
            Aidan Scannell*, Riccardo Mereu*, Paul Chang, Ella Tamir, Joni Pajarinen, Arno Solin<br>
            <strong>International Conference on Learning Representations (ICLR 2024)</strong><br>
            <!-- <a href="https://arxiv.org/abs/2309.02195"><img alt="Paper" src="https://img.shields.io/badge/-Paper-gray"></a> -->
            <!-- <a href="https://github.com/aidanscannell/sfr"><img alt="Code" src="https://img.shields.io/badge/-Code-gray" ></a> -->
        </td>
    </tr>
    <tr>
        <td>
            <a href="https://arxiv.org/abs/2309.02195">
              <strong>Sparse Function-space Representation of Neural Networks</strong><br>
            </a>
            Aidan Scannell*, Riccardo Mereu*, Paul Chang, Ella Tamir, Joni Pajarinen, Arno Solin<br>
            <strong>ICML 2023 Workshop on Duality Principles for Modern Machine Learning</strong><br>
            <!-- <a href="https://arxiv.org/abs/2309.02195"><img alt="Paper" src="https://img.shields.io/badge/-Paper-gray"></a> -->
            <!-- <a href="https://github.com/aidanscannell/sfr"><img alt="Code" src="https://img.shields.io/badge/-Code-gray" ></a> -->
        </td>
    </tr>
</table>


<!-- <sup>*</sup> == Equal Contribution  -->

![SFR](regression.png "SFR")
<!-- PyTorch implementation of Sparse Function-space Representation of Neural Networks. -->

## Abstract
> Sequential learning paradigms pose challenges for gradient-based deep learning due to difficulties incorporating new data and retaining prior knowledge. While Gaussian processes elegantly tackle these problems, they struggle with scalability and handling rich inputs, such as images. To address these issues, we introduce a technique that converts neural networks from weight space to function space, through a dual parameterization. Our parameterization offers: (i) a way to scale function-space methods to large data sets via sparsification, (ii) retention of prior knowledge when access to past data is limited, and (iii) a mechanism to incorporate new data without retraining. Our experiments demonstrate that we can retain knowledge in continual learning and incorporate new data efficiently. We further show its strengths in uncertainty quantification and guiding exploration in model-based RL.
<!-- > Deep neural networks (NNs) are known to lack uncertainty estimates and struggle to incorporate new data. We present a method that mitigates these issues by converting NNs from weight space to function space, via a dual parameterization. Importantly, the dual parameterization enables us to formulate a sparse representation that captures information from the entire data set. This offers a compact and principled way of capturing uncertainty and enables us to incorporate new data without retraining whilst retaining predictive performance. We provide proof-of-concept demonstrations with the proposed approach for quantifying uncertainty in supervised learning on UCI benchmark tasks. -->
<!-- > Sequential learning paradigms pose challenges for gradient-based deep learning due to difficulties incorporating new data and retaining prior knowledge. While Gaussian processes elegantly tackle these problems, they struggle with scalability and handling rich inputs, such as images. To address these issues, we introduce a technique that converts neural networks from weight space to function space, through a dual parameterization. Our parameterization offers: (i) a way to scale function-space methods to large data sets via sparsification, (ii) retention of prior knowledge when access to past data is limited, and (iii) a mechanism to incorporate new data without retraining. Our experiments demonstrate that we can retain knowledge in continual learning and incorporate new data efficiently. We further show its strengths in uncertainty quantification and guiding exploration in model-based RL. -->

## TL;DR
- `SFR` is a "posthoc" Bayesian deep learning method
    - equip any trained NN with uncertainty estimates
 <!-- which equips trained neural networks (NNs) with uncertainty estimates. -->
<!-- - `SFR` converts trained NNs to sparse GPs. -->
    <!-- - Unlike GPs, `SFR` scales to complex inputs (e.g. images)  -->
    <!--     - as it learns expressive covariance structures from data -->
- `SFR` can be viewed as a function-space Laplace approximation for NNs
- `SFR` has several benefits over [weight-space Laplace approximation for NNs](https://arxiv.org/abs/2106.14806):
    - Its function-space representation is effective for regularization in continual learning (CL)
    - It has good uncertainty estimates
        - We use them to guide exploration in model-based reinforcement learning (RL)
    - It can incorporate new data without retraining the NN
    <!-- - It learns expressive covariance structures from data -->
        <!-- - Not limited by stat -->

|                               | **SFR** | **GP** | **Laplace BNN**            |
|-------------------------------|:-------:|:------:|:--------------------------:|
| **Function-space**            | ✅      | ✅     | ❌ (*weight space*)        |
| **Image inputs**              | ✅      | ❌     | ✅                         |
| **Large data**                | ✅      | ❌     | ✅                         |
| **Incorporate new data fast** | ✅/❌   | ✅     | ❌ (*requires retraining*) |

## Useage
See the [notebooks](https://github.com/AaltoML/sfr/tree/main/notebooks) for how to use our code for both regression and classification.

### Minimal example
Here's a short example:
{% highlight python %}
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
{% endhighlight %}


## Citation
Please consider citing our conference paper:
{% highlight bibtex %}
@inproceedings{scannellFunction2024,
  title           = {Function-space Prameterization of Neural Networks for Sequential Learning},
  booktitle       = {Proceedings of The Twelth International Conference on Learning Representations (ICLR 2024)},
  author          = {Aidan Scannell and Riccardo Mereu and Paul Chang and Ella Tami and Joni Pajarinen and Arno Solin},
  year            = {2024},
  month           = {5},
}
{% endhighlight %}
Or our workshop paper:
{% highlight bibtex %}
@inproceedings{scannellSparse2023,
  title           = {Sparse Function-space Representation of Neural Networks},
  booktitle       = {ICML 2023 Workshop on Duality Principles for Modern Machine Learning},
  author          = {Aidan Scannell and Riccardo Mereu and Paul Chang and Ella Tami and Joni Pajarinen and Arno Solin},
  year            = {2023},
  month           = {7},
}
{% endhighlight %}
