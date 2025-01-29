import torch
import torch.nn as nn


def actmodule(activation: str):
    if activation == 'softplus':
        return nn.Softplus()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError('unknown activation function specified')


def draw_normal(mean: torch.Tensor, lnvar: torch.Tensor):

    std = lnvar
    eps = torch.randn_like(std)  # re-parametrization trick
    return mean + eps * std
