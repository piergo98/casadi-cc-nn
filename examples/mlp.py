"""This example shows how to build a simple multilayer perceptron (MLP)."""

import numpy as np

import ccnn

np_random = np.random.default_rng(69)

# define the features
n_in = 10
n_hidden = [32, 16, 8]
n_out = 4
features = [n_in, *n_hidden, n_out]

# define the activation functions - can also be a list including Nones
acts = [ccnn.ReLU, None, ccnn.ELU, ccnn.Tanh]  # one per hidden layer + output
mdl = ccnn.feedforward.Mlp(features, acts, biases=True)

# take a look at the model
print(mdl)
