#!/usr/bin/env python3
import numpy as np


## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    """Takes in a module and initializes all linear layers with weight
    values taken from a normal distribution."""

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find("Linear") != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)
