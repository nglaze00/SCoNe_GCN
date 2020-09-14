# normalized hodge laplacian (or not normalized)
# project flow onto null space of Laplacian
# set of e-vectors s.t. L*v = 0
# scipy.linalg.nullspace = V
# projection - inner product?
# proj = VV^T * flow
# softmax to get probs
import networkx as nx
import numpy as np
from scipy.linalg import null_space

def build_flow(G, path, edge_to_idx):
    """
    Builds a flow from this path on the given graph
    """
    flow = [0] * len(G.edges)
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        try:
            flow[edge_to_idx[edge]] = 1
        except:
            flow[edge_to_idx[edge[::-1]]] = -1
    return flow

def embed(G):
    """
    Embeds a networkx graph in the nullspace of its L1 matrix; returns embedding
    """
    B1 = np.array(nx.incidence_matrix(G, oriented=True).todense())
    L1 = B1.T @ B1
    V = null_space(L1)
    return V,B1

def predict_next(V, B1, flow):
    """
    Projects a flow onto embedding V and uses embedding to predict next node
    """
    proj = V @ V.T @ flow
    proj[B1[path[-1]] == 0] = -float('inf') # restrict to nbrs only
    return idx_to_edge[np.argmax(proj)][1] # return next node


path = [2, 3, 0]
suffix = [1, 2]

# test data
A = np.array([
     [0, 1, 1, 1],
     [1, 0, 1, 0],
     [1, 1, 0, 1],
     [1, 0, 1, 0]])
G = nx.from_numpy_matrix(A)

# build flow
edge_to_idx = {edge: i for i, edge in enumerate(G.edges)}
idx_to_edge = {i: edge for i, edge in enumerate(G.edges)}
flow = build_flow(G, path, edge_to_idx)

# embed
V, B1 = embed(G)

# pick next node
print(predict_next(V, B1, flow))
