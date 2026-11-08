import torch.nn as nn


ACTIVATIONS = {
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}
