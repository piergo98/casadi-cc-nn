__version__ = "1.0.6.post3"

__all__ = [
    "BatchNorm1d",
    "Dropout",
    "Dropout1d",
    "Module",
    "Sequential",
    "Linear",
    "ReLU",
    "RNN",
    "RNNCell",
    "Sigmoid",
    "Softplus",
    "Step",
    "Tanh",
    "convex",
    "feedforward",
    "get_sym_type",
    "init_parameters",
    "set_sym_type",
]


from typing import Literal, Union

import casadi as ca

from .activation import ReLU, Sigmoid, Softplus, Step, Tanh
from .containers import Sequential
from .dropout import Dropout, Dropout1d
from .linear import Linear
from .module import Module
from .norm import BatchNorm1d
from .recurrent import RNN, RNNCell


def get_sym_type() -> Union[type[ca.SX], type[ca.MX]]:
    """Gets the casadi's symbolic type used to build the networks.

    Returns
    -------
    type[ca.SX] or type[ca.MX]]
        The current symbolic type, either `casadi.SX` or `MX`.
    """
    return Module.sym_type


def set_sym_type(type: Literal["SX", "MX"]) -> None:
    """Sets the casadi's symbolic type to be used in building the networks.

    Parameters
    ----------
    type : "SX" or "MX"
        The name of the symbolic type to set.
    """
    Module.sym_type = getattr(ca, type)


# import these guys for last

import ccnn.convex as convex
import ccnn.feedforward as feedforward

from .init import init_parameters
