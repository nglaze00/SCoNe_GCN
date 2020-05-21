import jax.numpy as np
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
import numpy as onp

from synthetic_analysis.synthetic_sc_walk import G, G_undir, E, E_lookup, B1, paths

# get max degree for padding
D = onp.max(list(dict(G_undir.degree()).values()))

# hyperparams
step_size = 0.0005
batch_size = 100

# build graph matrices
L1_lower = B1.T @ B1
#L1_upper = B2 @ B2.T
L1_upper = onp.zeros([G.size(), G.size()])
L1 = L1_lower + L1_upper

# given a node, return an array of neighbors
def neighborhood(G, v):
    '''
    G: networkx undirected graph
    v: node label
    '''
    return onp.array(G[v])

# given an array of nodes and a correct node, return an indicator vector
def neighborhood_to_onehot(Nv, w, D=D):
    '''
    Nv: numpy array
    w: integer, presumably present in Nv
    D: max degree, for zero padding
    '''
    onehot = (Nv==w).astype(onp.float)
    onehot_final = onp.zeros(D)
    onehot_final[:onehot.shape[0]] = onehot
    return np.array([onehot_final]).T

# given an array of neighbors, return the subincidence matrix
def conditional_incidence_matrix(B1, Nv, D=D):
    '''
    B1: numpy array
    Nv: row indices of B1 to extract
    D: max degree, for zero padding
    '''
    Bcond = onp.zeros([D,B1.shape[1]])
    Bcond[:len(Nv),:] = B1[Nv]
    return Bcond

# given a path, return the flow vector
# assumes edges (a,b) obey a<b
def path_to_flow(path, E_lookup, m):
    '''
    path: list of nodes
    E_lookup: dictionary mapping edge tuples to indices
    m: number of edges
    '''
    l = len(path)
    f = onp.zeros([m,1])
    for j in range(l-1):
        v0 = path[j]
        v1 = path[j+1]
        if v0 < v1:
            k = E_lookup[(v0,v1)]
            f[k] += 1
        else:
            k = E_lookup[(v1,v0)]
            f[k] -= 1
    return f

# model definitions
def relu(x):
    return np.maximum(x, 0)

def predict(params, S0, S1, f, Bcond):
    [[W00, W10, W20], [W01, W11, W21], [Wf]] = params
    
    g0 = np.dot(f,W00) + np.dot(np.dot(S0,f),W10) + np.dot(np.dot(S1,f),W20)
    g1 = relu(g0)

    h0 = np.dot(g1,W01) + np.dot(np.dot(S0,g1),W11) + np.dot(np.dot(S1,g1),W21)
    h1 = relu(h0)
    
    logits = np.dot(np.dot(Bcond,h1),Wf)
    
    return logits - logsumexp(logits)

batched_predict = vmap(predict, in_axes=(None, None, None, 0, 0))

def loss(params, fs, S0, S1, Bconds, targets):
    preds = batched_predict(params, S0, S1, fs, Bconds)
    return -np.sum(preds * targets)

def accuracy(params, fs, S0, S1, Bconds, targets):
    target_choice = np.argmax(targets, axis=1)
    preds = batched_predict(params, S0, S1, fs, Bconds)
    pred_choice = np.argmax(preds, axis=1)
    return np.mean(pred_choice == target_choice)

@jit
def update(params, fs, S0, S1, Bconds, targets):
    grads = grad(loss)(params, fs, S0, S1, Bconds, targets)
    
    [[W00, W10, W20], [W01, W11, W21], [Wf]] = params
    [[dW00, dW10, dW20], [dW01, dW11, dW21], [dWf]] = grads
    
    return [[W00-step_size*dW00, W10-step_size*dW10, W20-step_size*dW20],
            [W01-step_size*dW01, W11-step_size*dW11, W21-step_size*dW21],
            [Wf-step_size*dWf]]

# paths
# sample, truncate
paths_truncated = [p[0:4+onp.random.choice(len(p)-4)] for p in paths]
paths_truncated_in = [p[0:-1] for p in paths_truncated]
paths_truncated_endpoints = [p[-1] for p in paths_truncated]
# convert to flow
flows_in = [path_to_flow(p, E_lookup, len(E)) for p in paths_truncated_in]
# get conditional incidence matrices
paths_truncated_neighbors = [neighborhood(G_undir, p[-1]) for p in paths_truncated_in]
print('Mean number of choices: {}'.format(onp.mean([len(Nv) for Nv in paths_truncated_neighbors])))
Bconds = [conditional_incidence_matrix(B1, Nv) for Nv in paths_truncated_neighbors]
# create one-hot target vectors
targets = [neighborhood_to_onehot(Nv, w) for Nv,w in zip(paths_truncated_neighbors, paths_truncated_endpoints)]

flows_in = onp.array(flows_in)
Bconds = onp.array(Bconds)
targets = onp.array(targets)

params = [
    [0.01*onp.random.randn(1,8), 0.01*onp.random.randn(1,8), 0.01*onp.random.randn(1,8)],
    [0.01*onp.random.randn(8,8), 0.01*onp.random.randn(8,8), 0.01*onp.random.randn(8,8)],
    [0.01*onp.random.randn(8,1)]
]

for i in range(1000*flows_in.shape[0]//batch_size):
    batch_indices = onp.random.choice(flows_in.shape[0],batch_size,replace=False)
    batch_flows_in = flows_in[batch_indices]
    batch_Bconds = Bconds[batch_indices]
    batch_targets = targets[batch_indices]
    params = update(params, batch_flows_in, L1_lower, L1_upper, batch_Bconds, batch_targets)
    if i%(flows_in.shape[0]//batch_size)==0:
        cost = loss(params, flows_in, L1_lower, L1_upper, Bconds, targets)
        acc = accuracy(params, flows_in, L1_lower, L1_upper, Bconds, targets)
        print('{} {:.3f} {:.3f}'.format(i//100,cost,acc))

def preview_output():
    j = onp.random.choice(50)
    print(paths_truncated[j])
    print(paths_truncated_neighbors[j])
    preds = predict(params, L1_lower, L1_upper, flows_in[j], Bconds[j])
    print(onp.exp(preds.T[0]))
    print(onp.argmax(preds.T[0]))

# plot network flow
# node_size = onp.zeros(len(V))
# node_size[path_neighbors] = final_preds
# nx.draw_networkx(G_undir, with_labels=False,
#                  width=0.3,
#                  node_size=100*node_size, pos=dict(zip(V, points)))
# plt.plot(points[p,0],points[p,1], color='g')
# plt.plot(points[path_in,0],points[path_in,1], color='r')
# plt.savefig('flow-gnn.pdf')
# plt.gcf().clear()
