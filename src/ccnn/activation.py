from typing import Literal

import ccnn.functional as F
from ccnn.module import Module, SymType


class ReLU(Module[SymType]):
    """Applies the rectified linear unit function element-wise as
    `ReLU(x) = (x)^+ = max(0, x)`."""

    def __init__(self, complementarity: bool = False, tau: float = 1.0) -> None:
        super().__init__()
        self.complementarity = bool(complementarity)
        self.tau = float(tau)

    def forward(self, input: SymType) -> SymType:
        return F.relu(input, complementarity=self.complementarity, tau=self.tau)

    def extra_repr(self) -> str:
        return f"complementarity={self.complementarity}, tau={self.tau}"


class Softplus(Module[SymType]):
    """Applies the softplus function element-wise as
    `Softplus(x) = 1 / beta * log(1 + exp(beta * x))`.

    Parameters
    ----------
    beta : float, optional
        The beta parameter of the softplus function, by default `1.0`.
    threshold : float, optional
        The threshold parameter of the softplus function, by default `20.0`.
    """

    def __init__(
        self,
        beta: float = 1.0,
        threshold: float = 20.0,
        complementarity: bool = False,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.complementarity = bool(complementarity)

    def forward(self, input: SymType) -> SymType:
        return F.softplus(
            input,
            self.beta,
            self.threshold,
            complementarity=self.complementarity,
        )

    def extra_repr(self) -> str:
        return (
            f"beta={self.beta}, threshold={self.threshold}, "
            f"complementarity={self.complementarity}"
        )


class Sigmoid(Module[SymType]):
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`."""

    def __init__(self, complementarity: bool = False) -> None:
        super().__init__()
        self.complementarity = bool(complementarity)

    def forward(self, input: SymType) -> SymType:
        return F.sigmoid(input, complementarity=self.complementarity)

    def extra_repr(self) -> str:
        return f"complementarity={self.complementarity}"


class Tanh(Module[SymType]):
    """Applies the element-wise function
    `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`."""

    def __init__(self, complementarity: bool = False) -> None:
        super().__init__()
        self.complementarity = bool(complementarity)

    def forward(self, input: SymType) -> SymType:
        return F.tanh(input, complementarity=self.complementarity)

    def extra_repr(self) -> str:
        return f"complementarity={self.complementarity}"
    
class Step(Module[SymType]):
    """Applies the element-wise function `Step(x) = 1 if x > 0 else 0`."""

    def __init__(self, complementarity: bool = False) -> None:
        super().__init__()
        self.complementarity = bool(complementarity)

    def forward(self, input: SymType) -> SymType:
        return F.step(input, complementarity=self.complementarity)

    def extra_repr(self) -> str:
        return f"complementarity={self.complementarity}"
