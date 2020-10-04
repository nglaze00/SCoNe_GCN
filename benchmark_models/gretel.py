# GRETEL aka fancy markov model

# input x_t: probability distribution of agent across all nodes at time t
# trajectory phi = subseq of (x_1, x_2, ...x_t)
# pseudo-coordinate vectors c_i = [c[i,1],...c[i, |phi|]]
#   c[i,T] = GCN(x_T) with k layers
#    ??   GCN layer X_k = relu(sum(w(e[j,i]) * X_k-1[j]) over all nodes j
#   w(e[j,i]) = MLP(f[i], f[j], f[i,j])
#   final weight = softmax(MLP(c[i], c[j], f[i], f[j], f[i,j]))


# data gen
# no features & only node features

# questions:
# unique weights for each path, right?


import jax.numpy as np
from jax import vmap, jit, grad
from jax.ops import index_update, index
import numpy as onp


def softmax(z):
    """
    z: vector of initialized weight for edges i->l for all l
    """
    return z / np.sum(z, axis=1, keepdims=True)

def relu(x):
    return np.maximum(x, 0)

def gcn_layer(W, X): # same W for all layers
    return relu(np.dot(X, W))

def mlp_layer(W, b, X):
    return relu(np.dot(W, X) + b)

def gen_params(c_mlp_shapes, z_mlp_shapes, seed=5):
    """
    Generates initial parameters for the C-matrix MLPs and Z-matrix MLPs
    Returns c_mlp_Ws, c_mlp_bs, z_mlp_Ws, z_mlp_bs
    """
    onp.random.seed(seed)
    return [onp.random.random_sample(shape) for shape in c_mlp_shapes], \
            [onp.random.random_sample((shape[0], 1)) for shape in c_mlp_shapes], \
            [onp.random.random_sample(shape) for shape in z_mlp_shapes], \
            [onp.random.random_sample((shape[0], 1)) for shape in z_mlp_shapes]

# feed each vector x[t] in trajectory through k-layer gcn; output is "pseudo-coordinate vector" c[t]
# Each layer in GCN has same weights, initialized through a ?-layer MLP w/node i, node j, and edge i,j feature inputs
# For each edge, initialize each edge weight z[i,j] with ?-layer MLP w/c[i], c[j], f[i], f[j], f[i,j] inputs
# Final weight = softmax(z[i,j]) over its neighbors

def gretel(X, F, E, A, weights, k=3):
    """
    X:          matrix of vectors x_t
    F:          node features
    E:          edge features
    c_mlp_Ws:   W matrices for each layer of GCN weights MLP
    todo
    """
    c_mlp_Ws, c_mlp_bs, z_mlp_Ws, z_mlp_bs = weights[:3], weights[3:6], weights[6:9], weights[9:]

    # GCN weight MLPs (build matrix W_gcn)
    W_gcn = np.zeros(E.shape)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if A[i][j] == 0: # skip non-connected edges
                continue

            cur = np.concatenate((F[i], F[j], E[i][j]))
            cur = cur.reshape((cur.shape[0], 1))

            for W, b in zip(c_mlp_Ws, c_mlp_bs):
                cur = mlp_layer(W, b, cur)

            W_gcn = index_update(W_gcn, index[i, j], cur[0][0])
    W_gcn = W_gcn.reshape(W_gcn.shape[:2])

    # GCN over each x_t (build matrix C)
    C = np.zeros(X.shape)
    for t in range(X.shape[1]):
        cur = X[:, t]
        for _ in range(k):
            cur = gcn_layer(W_gcn, cur)
        C = index_update(C, index[:, t], cur)

    # edge weight MLP (build matrix Z)
    Z = np.zeros(E.shape)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if A[i][j] == 0: # skip non-connected edges
                continue
            cur = np.concatenate((C[i], C[j], F[i], F[j], E[i][j]))
            cur = cur.reshape((cur.shape[0], 1))

            for W, b in zip(z_mlp_Ws, z_mlp_bs):
                cur = mlp_layer(W, b, cur)

            Z = index_update(Z, index[i, j], cur[0][0])

    # softmax over outgoing nbrs of each node
    return softmax(Z)

gretel_batch = vmap(gretel, in_axes=(0, None, None, None, None))

def target_likelihood(X_all, F, E, A, idx_to_edge, weights, suffixes, ):
    W_all = gretel_batch(X_all, F, E, A, weights)

    P_all = np.zeros((X_all.shape[0], len(idx_to_edge), len(idx_to_edge)))
    for s in range(W_all.shape[0]):
        for a, (i, j) in idx_to_edge.items():
            for b, (k, l) in idx_to_edge.items():
                if j != k or i == l:
                    continue
                P_all = index_update(P_all, index[s,a,b], (W_all[s][k][l] / (1 - W_all[s][k][i]))[0])

    B_all = np.zeros((X_all.shape[0], len(idx_to_edge), A.shape[0]))
    for s in range(B_all.shape[0]):
        for m, edge in idx_to_edge.items():
            B_all = index_update(B_all, index[s, m, edge[0]], W_all[s][edge[0]][edge[1]][0])

    preds = np.linalg.pinv(B_all[0]) @ P_all[0] @ B_all[0] @ X_all[0][:, -1] # todo vectorize
    raise Exception
def suffix_likelihood(X, F, E, A, weights, suffix):
    """
    Returns the negative log-likelihood of the given suffix
    """

    W = gretel(X, F, E, A, weights)

    likelihood = 1
    prev_node = None
    cur_node = np.argmax(X[:, -1])
    for t in range(len(suffix)):
        if t == 0:
            likelihood *= W[cur_node][suffix[t]]
        elif cur_node != suffix[t]:
            likelihood *= W[cur_node][suffix[t]] / (1 - W[cur_node][prev_node])  # exclude backtracking
        else:
            return 0
        prev_node = cur_node
        cur_node = suffix[t]

    return -np.log(likelihood)

def loss(weights, X_all, F, E, A, suffixes):
    """
    Returns the negative log-likelihood of each sample
    """
    return np.sum(np.array([suffix_likelihood(X, F, E, A, weights, suffix) for X, suffix in zip(X_all, suffixes)]))


def sample_dataset():
    # sample dataset
    A = np.array([
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0]])

    edges = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == 1:
                edges.append((i, j))

    idx_to_edge = {i: edge for i, edge in enumerate(edges)}

    X = np.array([  # 2 paths of length 2
        [
            [1.0, 0.0],  # 0 -> 1
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ],
        [
            [1.0, 0.0],  # 0 -> 2
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ],
    ])
    # todo think of better sample data (tree structure?)
    # todo right format?
    y = np.array([  # suffix nodes
        [
            3
            # [0.0],  # 0 -> 1 -> (3)
            # [0.0],
            # [0.0],
            # [1.0],
            # [0.0],
            # [0.0],
            # [0.0]
        ],
        [
            5
            # [0.0],  # 0 -> 2 -> (5)
            # [0.0],
            # [0.0],
            # [0.0],
            # [0.0],
            # [1.0],
            # [0.0]
        ],
    ])
    F = np.array([
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0]
    ])
    E = np.array([
        [
            [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
        ],
        [
            [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
        ],
        [
            [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
        ],
        [
            [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
        ],
        [
            [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
        ],
        [
            [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
        ],
        [
            [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
        ],

    ])
    return X, F, E, A, y, idx_to_edge

def train_gretel(X, F, E, A, y, idx_to_edge, k=3, lr=1):
    # initialize weights
    c_mlp_shapes = [[2 * F.shape[-1] + E.shape[-1]] * 2, [2 * F.shape[-1] + E.shape[-1]] * 2, [1, 2 * F.shape[-1] + E.shape[-1]]]
    z_mlp_shapes = [[2 * X.shape[-1] + 2 * F.shape[-1] + E.shape[-1]] * 2, [2 * X.shape[-1] + 2 * F.shape[-1] + E.shape[-1]] * 2, [1, 2 * X.shape[-1] + 2 * F.shape[-1] + E.shape[-1]]]
    c_mlp_W, c_mlp_b, z_mlp_W, z_mlp_b = gen_params(c_mlp_shapes, z_mlp_shapes)

    weights = [*c_mlp_W, *c_mlp_b, *z_mlp_W, *z_mlp_b]

    # test GRETEL on first sample
    gretel(X[1], F, E, A, weights)
    # print(target_likelihood(X, F, E, A, idx_to_edge, weights, y))
    print(loss(weights, X, F, E, A, y))
    # train GRETEL


    # @jit
    def gradient_step(X, F, E, A, weights, y):
        grads = grad(loss)(weights, X, F, E, A, y)
        # print(grads[0])
        # print([a.shape for a in grads])
        # print([a.shape for a in weights])
        for i in range(len(weights)):
            weights[i] -= lr * grads[i]

        print(np.average(np.abs(grads[-1])))

        return weights
    for i in range(10):
        weights = gradient_step(X, F, E, A, weights, y)
        print(loss(weights, X, F, E, A, y))
    raise Exception


if __name__ == '__main__':
    train_gretel(*sample_dataset())

# Build path by choosing biggest edge weight at each node
# at node i-> next is max(W[i])