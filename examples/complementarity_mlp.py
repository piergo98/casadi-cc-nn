"""Example: complementarity constraints for a ReLU hidden layer in a small MLP.

The complementarity form encodes ReLU as:
- y >= 0
- y >= z
- y * (y - z) <= 0
"""

import casadi as cs
import numpy as np

import ccnn.functional as F

rng = np.random.default_rng(69)

# Define a small MLP: y = W2 * relu(W1 * x + b1) + b2
n_in = 2
n_hidden = 3
n_out = 1

W1 = rng.normal(size=(n_hidden, n_in))
b1 = rng.normal(size=(1, n_hidden))
W2 = rng.normal(size=(n_out, n_hidden))
b2 = rng.normal(size=(1, n_out))

# Symbolic input (row vector to match ccnn Linear conventions)
x = cs.MX.sym("x", 1, n_in)

z1 = x @ W1.T + b1
relu_res = F.relu(z1, complementarity=True, tau=0.0)

h = relu_res["output"]
y = h @ W2.T + b2

# Build an NLP that solves for the complementarity output h
h_vec = cs.vec(h)
g_list = [cs.vec(g) for g in relu_res["g"]]
g_all = cs.vertcat(*g_list) if g_list else cs.MX.zeros(0, 1)

nlp = {
    "x": h_vec,
    "p": cs.vec(x),
    "f": cs.sum1(h_vec),
    "g": g_all,
}
solver = cs.nlpsol("solver", "ipopt", nlp)

x_val = cs.DM(rng.normal(size=(1, n_in)))
sol = solver(
    x0=cs.DM.zeros(h_vec.shape),
    lbx=relu_res["lbw"],
    ubx=relu_res["ubw"],
    lbg=relu_res["lbg"],
    ubg=relu_res["ubg"],
    p=cs.vec(x_val),
)

h_opt = cs.reshape(sol["x"], *h.shape)

# Compare with the standard ReLU output
relu_direct = cs.fmax(0, z1)
y_direct = relu_direct @ W2.T + b2

y_fun = cs.Function("y_fun", [x, h], [y])
y_direct_fun = cs.Function("y_direct_fun", [x], [y_direct])

y_val = y_fun(x_val, h_opt).full().item()
y_direct_val = y_direct_fun(x_val).full().item()

print("Input x:", np.array(x_val))
print("Hidden (complementarity) h:", np.array(h_opt))
print("Output y (complementarity):", y_val)
print("Output y (direct ReLU):", y_direct_val)
