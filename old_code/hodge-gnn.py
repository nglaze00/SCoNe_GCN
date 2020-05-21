import jax.numpy as np
from jax import grad, jit
from jax.scipy.special import logsumexp
import numpy as onp

import networkx as nx
import matplotlib.pyplot as plt

from synthetic_analysis.synthetic_sc_walk import G, G_undir, V, E, E_lookup, B1, paths, points

# hyperparams
step_size = 0.005

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
def neighborhood_to_onehot(Nv, w):
    '''
    Nv: numpy array
    w: integer, presumably present in Nv
    '''
    return np.array([(Nv==w).astype(onp.float)]).T

# given an array of neighbors, return the subincidence matrix
def conditional_incidence_matrix(B1, Nv):
    '''
    B1: numpy array
    Nv: row indices of B1 to extract
    '''
    return B1[Nv]

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

def predict(params, f, S0, S1, Bcond):
    [[W00, W10, W20], [W01, W11, W21], [Wf]] = params
    
    g0 = np.dot(f,W00) + np.dot(np.dot(S0,f),W10) + np.dot(np.dot(S1,f),W20)
    g1 = relu(g0)

    h0 = np.dot(g1,W01) + np.dot(np.dot(S0,g1),W11) + np.dot(np.dot(S1,g1),W21)
    h1 = relu(h0)
    
    logits = np.dot(np.dot(Bcond,h1),Wf)
    
    return logits - logsumexp(logits)

def loss(params, f, S0, S1, Bcond, targets):
    preds = predict(params, f, S0, S1, Bcond)
    return -np.sum(preds * targets)

def choice(params, f, S0, S1, Bcond, targets):
    target_choice = np.argmax(targets)
    preds = predict(params, f, S0, S1, Bcond)
    pred_choice = np.argmax(preds)
    return pred_choice == target_choice

@jit
def update(params, f, S0, S1, Bcond, targets):
    grads = grad(loss)(params, f, S0, S1, Bcond, targets)
    [[W00, W10, W20], [W01, W11, W21], [Wf]] = params
    [[dW00, dW10, dW20], [dW01, dW11, dW21], [dWf]] = grads
    return [[W00-step_size*dW00, W10-step_size*dW10, W20-step_size*dW20],
            [W01-step_size*dW01, W11-step_size*dW11, W21-step_size*dW21],
            [Wf-step_size*dWf]]

# start out with a single path to train on
p = paths[onp.random.choice(len(paths))]
path_in = p[0:-1]
flow_in = path_to_flow(path_in, E_lookup, len(E))
path_neighbors = neighborhood(G_undir, p[-2])
Bcond = conditional_incidence_matrix(B1, path_neighbors)
path_next_step = neighborhood_to_onehot(path_neighbors, p[-1])

print(p)
print(path_in)

params = [
    [onp.random.randn(1,4), onp.random.randn(1,4), onp.random.randn(1,4)],
    [onp.random.randn(4,4), onp.random.randn(4,4), onp.random.randn(4,4)],
    [onp.random.randn(4,1)]
]

right_buffer = onp.array([])

for i in range(1000):
    params = update(params, flow_in, L1_lower, L1_upper, Bcond, path_next_step)
    cost = loss(params, flow_in, L1_lower, L1_upper, Bcond, path_next_step)
    right = choice(params, flow_in, L1_lower, L1_upper, Bcond, path_next_step)
    right_buffer = onp.append(right_buffer, right)
    print('{:d} {:d} {:.4f}        \r'.format(i,right,cost),end='')
    if (right_buffer[-5:]).all():
        # terrible early stopping lol
        break
    
print('\n')
final_preds = predict(params, flow_in, L1_lower, L1_upper, Bcond)
final_preds = onp.exp(final_preds.T[0])
print(path_neighbors)
print(final_preds)

# plot network flow
node_size = onp.zeros(len(V))
node_size[path_neighbors] = final_preds
nx.draw_networkx(G_undir, with_labels=False,
                 width=0.3,
                 node_size=100*node_size, pos=dict(zip(V, points)))
plt.plot(points[p,0],points[p,1], color='g')
plt.plot(points[path_in,0],points[path_in,1], color='r')
plt.savefig('flow-gnn.pdf')
plt.gcf().clear()
