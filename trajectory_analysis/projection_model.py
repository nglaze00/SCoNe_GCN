"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)

Projection model described in this paper: https://arxiv.org/pdf/1807.05044.pdf

To use, change :folder_suffix: to the suffix used by your dataset and run this file.
"""
folder_suffix = 'buoy'

import networkx as nx
import numpy as np
from scipy.linalg import null_space
from scipy.special import softmax
from synthetic_data_gen import load_dataset, incidence_matrices

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

def embed(B1, B2):
    """
    Embeds a networkx graph in the nullspace of its L1 matrix; returns the embedding
    """
    # embed with L1
    L1_lower = B1.T @ B1

    # embed with L2
    L1_upper = B2 @ B2.T

    V = null_space(L1_lower + L1_upper)
    # V = null_space(L1_lower) <- embed without face

    return V, B1

def neighborhood(G, v):
    '''
    G: networkx undirected graph
    v: node label
    '''
    return np.array(sorted(G[v]))

def project_flows(V, B1, flows, last_nodes, nbrhoods, max_deg):
    """
    Project flows onto null space; return probabilities of each suffix for each flow
    """
    projs = V @ V.T @ flows

    res = np.zeros((len(last_nodes), max_deg))
    # select nbr edges from B1 and projs

    for i in range(len(last_nodes)):
        n0, e0 = nbrhoods[i], np.where(B1[last_nodes[i]] != 0)[0]

        B1_cond = B1[n0][:, e0]
        projs_cond = projs[e0, i]
        res[i, :len(n0)] = B1_cond @ projs_cond

    return softmax(res, axis=1).T

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

def accuracy_2target(y, preds, n_nbrs):
    """
    Returns the ratio of the time that the true suffix has higher predicted probability than a random node
    """
    true_next = np.argmax(y, axis=0)
    n_true_greater = 0
    for i in range(len(true_next)):
        choices = np.arange(0, n_nbrs[i], 1)
        choices = np.delete(choices, np.where(choices == true_next[i]))
        choice = np.random.choice(choices)

        if preds[true_next[i], i] > preds[choice, i]:
            n_true_greater += 1
        elif preds[true_next[i], i] == preds[choice, i]:
            n_true_greater += 0.5

    return n_true_greater / len(true_next)

def test_dataset():
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

    B1, B2 = incidence_matrices(G, sorted(G.nodes), sorted(G.edges), get_faces(G), edge_to_idx)
    return G, paths, last_nodes, y, edge_to_idx, idx_to_edge, target_nodes, B1, B2

def synthetic_dataset(folder="trajectory_data_1hop_schaub2"):
    """
    Load synthetic dataset from folder
    """
    X, B_matrices, y, train_mask, test_mask, G_undir, last_nodes, target_nodes = load_dataset(folder)

    edge_to_idx = {edge: i for i, edge in enumerate(G_undir.edges)}
    idx_to_edge = {i: edge for i, edge in enumerate(G_undir.edges)}

    return G_undir, B_matrices, X.reshape(X.shape[:-1]), last_nodes, target_nodes, y.reshape(y.shape[:-1]), edge_to_idx, idx_to_edge, train_mask, test_mask

def eval_dataset(G, paths, last_nodes, y, edge_to_idx, idx_to_edge, target_nodes, B1, B2, max_deg, flows=None, two_target=False, target_nodes_2hop=None):
    """
    Runs experiment on given dataset
    """

    # embed
    V, b1 = embed(B1, B2)
    if type(B1) != np.ndarray or type(B2) != np.ndarray:
        raise Exception

    # build flows
    if type(flows) != np.ndarray:
        flows = np.array([build_flow(G, path, edge_to_idx) for path in paths]).T

    # find nbrhoods of each last node
    nbrhoods = [neighborhood(G, n) for n in last_nodes]

    n_nbrs = np.array([len(nbrhd) for nbrhd in nbrhoods])


    # project flows
    preds = project_flows(V, B1, flows, last_nodes, nbrhoods, max_deg)


    # compute loss + acc
    ce = loss(y, preds)
    if two_target:
        acc = accuracy_2target(y, preds, n_nbrs)
    else:
        acc = accuracy(y, preds)
    return ce, acc

# test dataset
# print(eval_dataset(*test_dataset()))


folder = 'trajectory_data_1hop_' + folder_suffix

# synthetic dataset
G, (B1, B2), flows, last_nodes, target_nodes, y, edge_to_idx, idx_to_edge, train_mask, test_mask = synthetic_dataset(folder='trajectory_data_1hop_' + folder_suffix)

max_deg = np.max([len(G[i]) for i in G.nodes])


print('Avg degree:', 2 * len(G.edges) / len(G.nodes))

# # Standard test set
last_nodes_test, y_test, target_nodes_test, flows_test = last_nodes[test_mask == 1], \
                                                         y[test_mask == 1].T, target_nodes[test_mask == 1], \
                                                         flows[test_mask == 1].T

print('Standard experiment loss / acc:', eval_dataset(G, None, last_nodes_test, y_test, edge_to_idx, idx_to_edge, target_nodes_test, B1, B2, max_deg, flows=flows_test))

# Reversed
flows_rev, last_nodes_rev, target_nodes_rev, targets_rev = tuple(map(np.load, (folder + '/rev_flows_in.npy', folder + '/rev_last_nodes.npy', folder + '/rev_target_nodes.npy', folder + '/rev_targets.npy')))
print('Reverse experiment loss / acc:', eval_dataset(G, None, last_nodes_rev[test_mask == 1], targets_rev.reshape(targets_rev.shape[:-1])[test_mask == 1].T, edge_to_idx, idx_to_edge, target_nodes_rev[test_mask == 1], B1, B2, max_deg, flows=flows_rev.reshape(flows_rev.shape[:-1])[test_mask == 1].T))


# 2-target
print('2-target acc:', eval_dataset(G, None, last_nodes_test, y_test, edge_to_idx, idx_to_edge, target_nodes_test, B1, B2, max_deg, flows=flows_test, two_target=True)[1])

# Transfer
regional_mask = np.array([1 if i % 3 == 2 else 0 for i in range(y.shape[0])])
print('Transfer experiment loss / acc:', eval_dataset(G, None, last_nodes[regional_mask == 1], y[regional_mask == 1].T, edge_to_idx, idx_to_edge, target_nodes[regional_mask == 1], B1, B2, max_deg, flows=flows[regional_mask == 1].T))


