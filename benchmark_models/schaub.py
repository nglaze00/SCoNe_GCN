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
from sklearn.metrics import log_loss
from synthetic_analysis.synthetic_data_gen import load_dataset, incidience_matrices

def softmax(z, axis=None):
    """
    z: vector of initialized weight for edges i->l for all l
    """
    return np.exp(z) / np.sum(np.exp(z), axis=axis)

def build_flow(G, path, edge_to_idx):
    """
    Builds a flow from this path on the given graph
    """
    flow = np.zeros(len(G.edges))
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        try:
            flow[edge_to_idx[edge]] = 1
        except:
            flow[edge_to_idx[edge[::-1]]] = -1
    return flow

def get_faces(G):
    """
    Returns a list of the faces in an undirected graph
    """
    edges = list(G.edges)
    faces = []
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            if e1[0] == e2[0]:
                shared = e1[0]
                e3 = (e1[1], e2[1])
            elif e1[1] == e2[0]:
                shared = e1[1]
                e3 = (e1[0], e2[1])
            elif e1[0] == e2[1]:
                shared = e1[0]
                e3 = (e1[1], e2[0])
            elif e1[1] == e2[1]:
                shared = e1[1]
                e3 = (e1[0], e2[0])
            else: # edges don't connect
                continue

            if e3[0] in G[e3[1]]: # if 3rd edge is in graph
                faces.append(tuple(sorted((shared, *e3))))
    return list(sorted(set(faces)))


def embed(G, B2=None):
    """
    Embeds a networkx graph in the nullspace of its L1 matrix; returns the embedding

    """
    # embed with L1
    B1 = np.array(nx.incidence_matrix(G, oriented=True).todense())
    # L1 = B1.T @ B1
    # V = null_space(L1)

    # embed with L2
    if type(B2) != np.ndarray:
        _, B2 = incidience_matrices(G, G.nodes, G.edges, get_faces(G))
    L2 = B2 @ B2.T
    V = null_space(L2)
    return V, B1

def neighborhood(G, v):
    '''
    G: networkx undirected graph
    v: node label
    '''
    return np.array(sorted(G[v]))

def project_flows(V, B1, flows, last_nodes, nbrhoods, max_deg):
    # print(B1.shape)
    # print(V.shape, V.T.shape, flows.shape)
    projs = V @ V.T @ flows
    # for i in range(projs.shape[1]):
    #     projs[:, i][B1[last_nodes[i]] == 0] = -float('inf')

    projs_nbrs = np.zeros((max_deg, flows.shape[1]))
    for i in range(len(last_nodes)):
        projs_nbrs[:len(nbrhoods[i]), i] = nbrhoods[i]
    return softmax(projs_nbrs, axis=0)

def loss(y, y_hat):
    """
    Evaluates cross-entropy loss for the given next-node distributions
        and predicted distributions # todo fix
    """
    y_hat_log = np.log(y_hat)
    y_hat_log[y_hat_log == -np.inf] = 0
    return -np.sum(y_hat_log * y) / y.shape[1]

def accuracy(y, y_hat):
    return np.average(np.argmax(y, axis=0) == np.argmax(y_hat, axis=0))


def sample_dataset():
    A = np.array([
         [0, 1, 1, 1],
         [1, 0, 1, 0],
         [1, 1, 0, 1],
         [1, 0, 1, 0]])
    G = nx.from_numpy_matrix(A)
    paths = np.array([[2,3,0], [1,0,3]])
    suffixes = [1, 2]
    y = np.array([
        [1, 0],
        [0, 1],
        [0, 0]]
    )
    last_nodes = [0, 3]

    edge_to_idx = {edge: i for i, edge in enumerate(G.edges)}
    idx_to_edge = {i: edge for i, edge in enumerate(G.edges)}

    target_nodes = [1, 2]

    return G, paths, last_nodes, y, edge_to_idx, idx_to_edge, target_nodes

def synthetic_dataset(folder="trajectory_data_1hop_schaub"):
    X, B_matrices, y, train_mask, test_mask, G_undir, last_nodes, target_nodes = load_dataset("../synthetic_analysis/" + folder)

    edge_to_idx = {edge: i for i, edge in enumerate(G_undir.edges)}
    idx_to_edge = {i: edge for i, edge in enumerate(G_undir.edges)}

    return G_undir, B_matrices, X.reshape(X.shape[:-1]).T, last_nodes, target_nodes, y, edge_to_idx, idx_to_edge

def eval_dataset(G, paths, last_nodes, y, edge_to_idx, idx_to_edge, target_nodes, flows=None, B1=None, B2=None):
    # embed
    V, b1 = embed(G, B2=B2)
    if type(B1) != np.ndarray:
        B1 = b1
    # build flows
    if type(flows) != np.ndarray:
        flows = np.array([build_flow(G, path, edge_to_idx) for path in paths]).T
    # print(flows.shape)

    nbrhoods = [neighborhood(G, n) for n in last_nodes]
    max_deg = max([len(nbrhd) for nbrhd in nbrhoods])
    # project flows
    preds = project_flows(V, B1, flows, last_nodes, nbrhoods, max_deg)
    print(y.shape)

    for i in range(preds.shape[1]):
        assert len(nbrhoods[i]) > np.argmax(y[:, i])
        print('preds:', preds[:, i], 'y:', y[:, i], 'best node:', np.argmax(y[:, i]), 'pred node:', np.argmax(preds[:, i]))
    # for i in range(preds.shape[1]):
    #     idxs = [j for j, x in enumerate(preds[:, i]) if x != 0]
    #     print(preds[idxs, i])
    #     print(y[idxs, i])

    # compute loss + acc
    ce = loss(y, preds)
    acc = accuracy(y, preds)
    return ce, acc

# test dataset
print(eval_dataset(*sample_dataset()))
# synthetic dataset
G, (B1, B2), flows, last_nodes, target_nodes, y, edge_to_idx, idx_to_edge = synthetic_dataset()
B1_mine = np.array(nx.incidence_matrix(G, oriented=True).todense())
# _, B2_mine = incidience_matrices(G, G.nodes, G.edges, get_faces(G))

print('Avg degree:', 2 * len(G.edges) / len(G.nodes))


# todo verify B2; why do mine & saved Bs give diff results?

print(eval_dataset(G, None, last_nodes, y.reshape(y.shape[:2]).T, edge_to_idx, idx_to_edge, target_nodes, flows=flows, B1=B1, B2=B2))
