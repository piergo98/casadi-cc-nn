from itertools import product
from random import random
from typing import Literal, Optional, TypeVar

import casadi as ca

SymType = TypeVar("SymType", ca.SX, ca.MX)


x = ca.MX.sym("x_mx")
y = ca.logsumexp(ca.vertcat(x, 0))
softplus_elementwise_mx = ca.Function("softplus_mx", [x], [y])
del x, y


def linear(input: SymType, weight: SymType, bias: Optional[SymType] = None) -> SymType:
    """Applies a linear transformation to the incoming data: `y = xA^T + b`."""
    output = input @ weight.T
    if bias is not None:
        output = (output.T + bias.T).T  # transpose trick is required
    return output


def _sym_like(input: SymType, name: str) -> SymType:
    sym = ca.MX.sym if isinstance(input, ca.MX) else ca.SX.sym
    nrow, ncol = input.shape
    return sym(name, nrow, ncol)


def _activation_result(
    output: SymType,
    vars_list: list[SymType],
    lbw: list[float],
    ubw: list[float],
    g: list[SymType],
    lbg: list[float],
    ubg: list[float],
) -> dict[str, object]:
    return {
        "output": output,
        "vars": vars_list,
        "lbw": lbw,
        "ubw": ubw,
        "g": g,
        "lbg": lbg,
        "ubg": ubg,
    }


def relu(
    input: SymType, complementarity: bool = False, tau: float = 1.0
) -> SymType | dict[str, object]:
    """Applies the rectified linear unit function element-wise as
    `ReLU(x) = (x)^+ = max(0, x)`.

    If `complementarity` is True, returns a dict with the output variable and
    complementarity constraints.
    """
    if not complementarity:
        return ca.fmax(0, input)

    output = _sym_like(input, "relu")
    n_out = int(output.numel())
    vars_list = [output]
    lbw = [0.0] * n_out
    ubw = [ca.inf] * n_out
    g = [output - input, output * (output - input)]
    lbg = [0.0] * n_out + [-ca.inf] * n_out
    ubg = [ca.inf] * n_out + [tau] * n_out
    return _activation_result(output, vars_list, lbw, ubw, g, lbg, ubg)


def softplus(
    input: SymType,
    beta: float = 1.0,
    threshold: float = 20.0,
    complementarity: bool = False,
) -> SymType | dict[str, object]:
    """Applies the softplus function element-wise as
    `Softplus(x) = 1/beta * log(1 + exp(beta * x))`.

    If `complementarity` is True, returns a dict with the output variable and
    equality constraints tying it to the softplus expression.
    """
    bi = beta * input
    if isinstance(input, ca.SX):
        expr = ca.if_else(input > threshold, bi, ca.log1p(ca.exp(bi)) / beta)
    else:
        expr = softplus_elementwise_mx(bi) / beta

    if not complementarity:
        return expr

    output = _sym_like(input, "softplus")
    n_out = int(output.numel())
    vars_list = [output]
    lbw = [0.0] * n_out
    ubw = [ca.inf] * n_out
    g = [output - expr]
    lbg = [0.0] * n_out
    ubg = [0.0] * n_out
    return _activation_result(output, vars_list, lbw, ubw, g, lbg, ubg)


def sigmoid(input: SymType, complementarity: bool = False) -> SymType | dict[str, object]:
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`.

    If `complementarity` is True, returns a dict with the output variable and
    equality constraints tying it to the sigmoid expression.
    """
    expr = 1 / (1 + ca.exp(-input))
    if not complementarity:
        return expr

    output = _sym_like(input, "sigmoid")
    n_out = int(output.numel())
    vars_list = [output]
    lbw = [0.0] * n_out
    ubw = [1.0] * n_out
    g = [output - expr]
    lbg = [0.0] * n_out
    ubg = [0.0] * n_out
    return _activation_result(output, vars_list, lbw, ubw, g, lbg, ubg)


def tanh(input: SymType, complementarity: bool = False) -> SymType | dict[str, object]:
    """Applies the element-wise function
    `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`.

    If `complementarity` is True, returns a dict with the output variable and
    equality constraints tying it to the tanh expression.
    """
    expr = ca.tanh(input)
    if not complementarity:
        return expr

    output = _sym_like(input, "tanh")
    n_out = int(output.numel())
    vars_list = [output]
    lbw = [-1.0] * n_out
    ubw = [1.0] * n_out
    g = [output - expr]
    lbg = [0.0] * n_out
    ubg = [0.0] * n_out
    return _activation_result(output, vars_list, lbw, ubw, g, lbg, ubg)

def step(input: SymType, complementarity: bool = False, tau: float = 1.0) -> SymType | dict[str, object]:
    """Applies the element-wise function `Step(x) = 1 if x > 0 else 0`.

    If `complementarity` is True, returns a dict with the output variable and
    complementarity constraints.
    """
    if not complementarity:
        return ca.if_else(input > 0, 1.0, 0.0)

    output = _sym_like(input, "step")
    n_out = int(output.numel())
    # 0 <= output <= 1
    vars_list = [output]
    lbw = [0.0] * n_out
    ubw = [1.0] * n_out
    # output * (-input) <= tau and (1 - output) * input <= tau
    g = [output * (-input), (1 - output) * input]
    lbg = [-ca.inf] * n_out + [-ca.inf] * n_out
    ubg = [tau] * n_out + [tau] * n_out
    return _activation_result(output, vars_list, lbw, ubw, g, lbg, ubg)


def rnn_cell(
    input: SymType,
    hidden: SymType,
    weight_ih: SymType,
    weight_hh: SymType,
    bias_ih: Optional[SymType] = None,
    bias_hh: Optional[SymType] = None,
    nonlinearity: Literal["tanh", "relu"] = "tanh",
    complementarity: bool = False,
    tau: float = 1.0,
) -> SymType:
    """Computes the output of a single Elman RNN cell."""
    out = linear(input, weight_ih, bias_ih) + linear(hidden, weight_hh, bias_hh)
    if nonlinearity == "tanh":
        return tanh(out)
    return relu(out, complementarity=complementarity, tau=tau)


def rnn(
    input: SymType,
    hidden: SymType,
    weights_ih: list[SymType],
    weights_hh: list[SymType],
    biases_ih: Optional[list[SymType]] = None,
    biases_hh: Optional[list[SymType]] = None,
    nonlinearity: Literal["tanh", "relu"] = "tanh",
    complementarity: bool = False,
    tau: float = 1.0,
) -> tuple[SymType, SymType]:
    """Applies a multi-layer Elman RNN cell with tanh or ReLU nonlinearity."""
    num_layers, h_size = hidden.shape
    seq_len, in_size = input.shape
    has_biases = biases_ih is not None

    # transform the evaluation of all layers into a single function call - but if the
    # sequence length is 1, we can just return at the end of the loop
    if seq_len == 1:
        input_ = input[0, :]
        hidden_ = hidden
    else:
        sym = weights_ih[0].sym
        input_ = input_loop = sym("in", in_size, 1).T
        hidden_ = sym("hidd", *hidden.shape)
    output_ = []
    for layer in range(num_layers):
        input_loop = rnn_cell(
            input_loop,
            hidden_[layer, :],
            weights_ih[layer],
            weights_hh[layer],
            biases_ih[layer] if has_biases else None,
            biases_hh[layer] if has_biases else None,
            nonlinearity,
            complementarity,
            tau
        )
        if complementarity:
            # if complementarity is True, input_loop is a dict with the output and constraints
            output_.append(input_loop["output"].T)
        else:
            output_.append(input_loop)
    output = ca.vcat(output_)
    if seq_len == 1:
        return output_[-1], output

    weights = weights_ih + weights_hh
    if has_biases:
        weights += biases_ih + biases_hh
    layers = ca.Function(
        "L", [hidden_, input_.T] + weights, (output, output[-1, :].T), {"cse": True}
    )

    # process each layer
    # OLD FOR-LOOP IMPLEMENTATION
    # output_.clear()
    # for t in range(seq_len):
    #     hidden, output = layers(hidden, input[t, :].T, *weights)
    #     output_.append(output.T)
    # return ca.vcat(output_), hidden
    mapaccum = layers.mapaccum(seq_len)
    all_hiddens, output = mapaccum(hidden, input.T, *weights)
    last_hidden = all_hiddens[:, -h_size:]
    return output.T, last_hidden