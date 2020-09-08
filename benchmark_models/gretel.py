# GRETEL aka fancy markov model

# input x_t: probability distribution of agent across all nodes at time t
# trajectory phi = subseq of (x_1, x_2, ...x_t)
# pseudo-coordinate vectors c_i = [c[i,1],...c[i, |phi|]]
#   c[i,T] = GCN(x_T) with k layers
#    ??   GCN layer X_k = relu(sum(w(e[j,i]) * X_k-1[j]) over all nodes j
#   w(e[j,i]) = MLP(f[i], f[j], f[i,j])
#   final weight = softmax(MLP(c[i], c[j], f[i], f[j], f[i,j]))

import jax.numpy as np

def softmax(z, j):
    """
    z: vector of initialized weight for edges i->l for all l
    j: index of target destination neighbor
    """
    return z[j] / np.sum(z)

def relu(x):
    return np.maximum(x, 0)

def gcn_layer(W, x): # ?? correct? (use same W for all layers)
    return relu(np.dot(W, x))

def mlp_layer(W, b, x): # 3 inputs; concatenate them?
    return np.dot(x, W) + b

# feed each vector x[t] in trajectory through k-layer gcn; output is "pseudo-coordinate vector" c[t]
# Each layer in GCN has same weights, initialized through a ?-layer MLP w/node i, node j, and edge i,j feature inputs
# For each edge, initialize each edge weight z[i,j] with ?-layer MLP w/c[i], c[j], f[i], f[j], f[i,j] inputs
# Final weight = softmax(z[i,j]) over its neighbors

# Build path by choosing biggest edge weight at each node