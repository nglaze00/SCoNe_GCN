# normalized hodge laplacian (or not normalized)
# project flow onto null space of Laplacian
# set of e-vectors s.t. L*v = 0
# scipy.linalg.nullspace = V
# projection - inner product?
# proj = VV^T * flow
# softmax to get probs

# With faces:
# test data: 54.5% 0.09054440753101489
# reverse: 58%
# 2-target: 82%
# middle only: 51%

# Without faces:
# test data: 32% 0.125266608753494
# reverse: 21%
# 2-target: 73.5%
# Regional: 58%


# Hodge accs:
# Epoch 552 -- train loss: 1.033017 -- train acc 0.624 -- test loss 0.960188 -- test acc 0.685
# Reversed: Test loss: 1.121262, Test acc: 0.585
# 2-target: 0.84
# Regional (train top, test bottom): Epoch 165 -- train loss: 1.314668 -- train acc 0.619 -- test loss 1.281366 -- test acc 0.607
# train middle, test middle = Epoch 599 -- train loss: 1.227443 -- train acc 0.540 -- test loss 0.907206 -- test acc 0.696 <--- Schaub relies on structure more, but Hodge has some memory capability

import networkx as nx
import numpy as np
from scipy.linalg import null_space
from scipy.special import softmax
from sklearn.metrics import log_loss
from synthetic_analysis.synthetic_data_gen import load_dataset, incidence_matrices
import matplotlib.pyplot as plt
from treelib import Tree


def edge_to_node_y(y_edge):
    """
    Given a matrix of one-hot encoded edge labels, convert it to a matrix of one-hot encoded node labels
    """

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


def embed(G, B1, B2):
    """
    Embeds a networkx graph in the nullspace of its L1 matrix; returns the embedding

    """
    # embed with L1
    # B1 = np.array(nx.incidence_matrix(G, oriented=True).todense())
    L1_lower = B1.T @ B1
    # V = null_space(L1)

    # embed with L2
    # if type(B2) != np.ndarray:
    #     _, B2 = incidience_matrices(G, G.nodes, G.edges, get_faces(G))
    L1_upper = B2 @ B2.T


    # W = np.linalg.eigvalsh(L1_lower+L1_upper)
    # plt.plot(W)
    # plt.savefig('test.pdf')
    # plt.show
    # raise Exception
    V = null_space(L1_lower + L1_upper)
    # V = null_space(L1_lower)
    return V, B1

def neighborhood(G, v):
    '''
    G: networkx undirected graph
    v: node label
    '''
    return np.array(sorted(G[v]))

def project_flows(V, B1, flows, last_nodes, nbrhoods, max_deg):

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


def multi_hop_accuracy_dist(V, flows, target_nodes_2hop, nbrhoods_by_node, nbrhoods, edge_to_idx, last_nodes, hops):
    """
    Returns accuracy of the model in making multi-hop predictions, using distributions at each intermediate hop
        instead of binary choices
    """

    max_deg = max([len(nbrhood) for nbrhood in nbrhoods_by_node])

    path_trees = [Tree() for _ in range(flows.shape[0])]
    # initialize leaves
    for i in range(len(path_trees)):
        path_trees[i].create_node(tag=last_nodes[i], identifier=str(last_nodes[i]), data=[flows[i], 1])

    # build trees
    for h in range(hops):
        for i in range(len(path_trees)):
            for leaf in path_trees[i].leaves():
                flow, last_node = np.array([leaf.data[0]]), leaf.tag



                probs = project_flows(V, B1, flow.T, [last_node], nbrhoods, max_deg)


                nbrs = np.array(nbrhoods_by_node[leaf.tag])

                for j in range(len(nbrs)):
                    new_edge = (int(leaf.tag), nbrs[j])
                    new_flow = np.array(flow)[0]
                    if new_edge[0] < new_edge[1]:
                        flow_val = 1

                    else:
                        flow_val = -1
                    new_flow[edge_to_idx[tuple(sorted(new_edge))]] = flow_val

                    prob_so_far = leaf.data[1]
                    path_trees[i].create_node(tag=nbrs[j], identifier=leaf.identifier + str(nbrs[j]),
                                              data=[new_flow, prob_so_far * probs[j]], parent=leaf.identifier)



    # find prob that target node is reached for each flow
    target_probs = np.zeros(len(path_trees))
    for i in range(len(path_trees)):
        target_prob = 0
        valid_paths = 0

        for leaf in path_trees[i].leaves():
            print(leaf.tag, target_nodes_2hop[i])
            if leaf.tag == target_nodes_2hop[i]:
                valid_paths += 1
                target_prob += leaf.data[1]
        target_prob /= valid_paths
        target_probs[i] = target_prob



    return np.average(target_probs)

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

    B1, B2 = incidence_matrices(G, sorted(G.nodes), sorted(G.edges), get_faces(G), edge_to_idx)
    return G, paths, last_nodes, y, edge_to_idx, idx_to_edge, target_nodes, B1, B2

def synthetic_dataset(folder="trajectory_data_1hop_schaub"):
    X, B_matrices, y, train_mask, test_mask, G_undir, last_nodes, target_nodes = load_dataset("../synthetic_analysis/" + folder)

    edge_to_idx = {edge: i for i, edge in enumerate(G_undir.edges)}
    idx_to_edge = {i: edge for i, edge in enumerate(G_undir.edges)}

    return G_undir, B_matrices, X.reshape(X.shape[:-1]), last_nodes, target_nodes, y.reshape(y.shape[:-1]), edge_to_idx, idx_to_edge, train_mask, test_mask

def eval_dataset(G, paths, last_nodes, y, edge_to_idx, idx_to_edge, target_nodes, B1, B2, flows=None, two_target=False, two_hop=False, target_nodes_2hop=None, max_deg=None):
    # embed
    print(y.shape, flows.shape)

    V, b1 = embed(G, B1, B2)
    if type(B1) != np.ndarray or type(B2) != np.ndarray:
        raise Exception
    # build flows
    if type(flows) != np.ndarray:
        flows = np.array([build_flow(G, path, edge_to_idx) for path in paths]).T

    # find nbrhoods of each last node
    nbrhoods = [neighborhood(G, n) for n in last_nodes]

    n_nbrs = np.array([len(nbrhd) for nbrhd in nbrhoods])

    if not max_deg:
        max_deg = np.max(n_nbrs)

    # project flows
    preds = project_flows(V, B1, flows, last_nodes, nbrhoods, max_deg)


    # compute loss + acc
    ce = loss(y, preds)
    if two_target:
        acc = accuracy_2target(y, preds, n_nbrs)
    elif two_hop:
        # nbrhoods_by_node[node] = nbrhood of node
        nbrhoods_by_node = [list(sorted(G[node].keys())) for node in range(len(G.nodes))]

        acc = multi_hop_accuracy_dist(V, flows.T, target_nodes_2hop, nbrhoods_by_node, nbrhoods, edge_to_idx, last_nodes, 2)
    else:
        acc = accuracy(y, preds)
    return ce, acc

# test dataset
# print(eval_dataset(*sample_dataset()))


# synthetic dataset
G, (B1, B2), flows, last_nodes, target_nodes, y, edge_to_idx, idx_to_edge, train_mask, test_mask = synthetic_dataset()

print('Avg degree:', 2 * len(G.edges) / len(G.nodes))
print(flows.shape, y.shape)

# All points
# print(eval_dataset(G, None, last_nodes, y.T, edge_to_idx, idx_to_edge, target_nodes, B1, B2, flows=flows.T))
#
# # Standard test set
last_nodes_test, y_test, target_nodes_test, flows_test = last_nodes[test_mask == 1], \
                                                         y[test_mask == 1].T, target_nodes[test_mask == 1], \
                                                         flows[test_mask == 1].T

print(eval_dataset(G, None, last_nodes_test, y_test, edge_to_idx, idx_to_edge, target_nodes_test, B1, B2, flows=flows_test))

# # Reversed
folder = '../synthetic_analysis/trajectory_data_1hop_schaub'
flows_rev, last_nodes_rev, target_nodes_rev, targets_rev = tuple(map(np.load, (folder + '/rev_flows_in.npy', folder + '/rev_last_nodes.npy', folder + '/rev_target_nodes.npy', folder + '/rev_targets.npy')))
print(eval_dataset(G, None, last_nodes_rev[test_mask == 1], targets_rev.reshape(targets_rev.shape[:-1])[test_mask == 1].T, edge_to_idx, idx_to_edge, target_nodes_rev[test_mask == 1], B1, B2, flows=flows_rev.reshape(flows_rev.shape[:-1])[test_mask == 1].T))


# 2-target
print(eval_dataset(G, None, last_nodes_test, y_test, edge_to_idx, idx_to_edge, target_nodes_test, B1, B2, flows=flows_test, two_target=True))

# 2-hop
# target_nodes_2hop = np.load('../synthetic_analysis/trajectory_data_2hop_schaub/target_nodes.npy')
# print(eval_dataset(G, None, last_nodes_test, y_test, edge_to_idx, idx_to_edge, target_nodes_test, B1, B2, flows=flows_test, two_hop=True, target_nodes_2hop=target_nodes_2hop[test_mask == 1]))

# regional
regional_mask = np.array([1 if i % 3 == 0 else 0 for i in range(y.shape[0])])
print(eval_dataset(G, None, last_nodes[regional_mask == 1], y[regional_mask == 1].T, edge_to_idx, idx_to_edge, target_nodes[regional_mask == 1], B1, B2, flows=flows[regional_mask == 1].T, max_deg=13))

# todo: fix 2hop


