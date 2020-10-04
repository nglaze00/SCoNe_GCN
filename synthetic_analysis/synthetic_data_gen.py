import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import os

def random_SC_graph(n, holes=True):
    """
    Randomly generates a graph of simplicial complexes, made up of n nodes.
    Graph has holes in top left and bottom right regions.

    :param n: # of nodes in graph

    Returns:
        NetworkX DiGraph object G
        Sorted list of nodes V
        Sorted list of edges E
        Map  (edge tuples -> indices in E) edge_to_idx
        List of faces
        List of valid node indexes (nodes not in either hole)

    """
    np.random.seed(0)
    coords = np.random.rand(n,2)
    np.random.seed(1030)
    tri = Delaunay(coords)

    valid_idxs = np.where((np.linalg.norm(coords - [1/4, 3/4], axis=1) > 1/8) \
                          & (np.linalg.norm(coords - [3/4, 1/4], axis=1) > 1/8))[0]

    if not holes:
        valid_idxs = np.array(range(len(coords)))
    faces = np.array(sorted([sorted(t) for t in tri.simplices if np.in1d(t, valid_idxs).all()]))

    # SC matrix construction
    G = nx.OrderedDiGraph()
    G.add_nodes_from(np.arange(n)) # add nodes that are excluded to keep indexing easy

    for f in faces:
        [a,b,c] = sorted(f)
        G.add_edge(a,b)
        G.add_edge(b,c)
        G.add_edge(a,c)

    V = np.array(sorted(G.nodes))
    E = list(sorted(G.edges))
    edge_to_idx = {E[i]: i for i in range(len(E))}

    return G, V, E, faces, edge_to_idx, coords, valid_idxs

def incidience_matrices(G, V, E, faces):
    """
    Returns incidence matrices B1 and B2

    :param G: NetworkX DiGraph
    :param V: list of nodes
    :param E: list of edges
    :param faces: list of faces in G

    Returns B1 (|V| x |E|) and B2 (|E| x |faces|)
    B1[i][j]: -1 if node is is tail of edge j, 1 if node is head of edge j, else 0 (tail -> head) (smaller -> larger)
    B2[i][j]: 1 if edge i appears sorted in edge j, -1 if edge i appears reversed in face j, else 0; given faces with sorted node order
    """
    B1 = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense())
    B2 = np.zeros([len(E),len(faces)])

    for f_idx, face in enumerate(faces):
        for e_idx, edge in enumerate(E):
            [tail, head] = sorted(edge)

            if np.in1d(edge, face).all(): # if edge in face
                [a, b, c] = face
                if (tail == a and head == b) or \
                    (tail == b and head == c) or \
                    (tail == c and head == a):
                    B2[e_idx, f_idx] = 1
                else:
                    B2[e_idx, f_idx] = -1
    return B1, B2

def generate_random_walks(G, points, valid_idxs, m=1000):
    """
    Generates m random walks over the valid nodes in G.

    trajectories will look like one of the following:
        BEGIN -> A0 -> B0 -> END (top left regions)
        BEGIN -> A1 -> B1 -> END (middle regions)
        BEGIN -> A2 -> B2 -> END (bottom right regions

    :param G: NetworkX digraph
    :param points: list of (x, y) points that make up the graph's nodes
    :param valid_idxs: list of valid node indexes in
    :param E: sorted list of edges in E
    :param edge_to_idx: map (edge tuple -> index
    :param m: # of walks to generate

    Returns:
        paths: List of walks (each walk is a list of nodes)
        flows: E x m matrix:
            index i,j is 1 if flow j contains edge e in the direction of increasing node #
            i,j is -1 if decreasing node #
            else 0
    """
    points_valid = points[valid_idxs]

    # "partition" node space
    BEGIN = valid_idxs[np.sum(points_valid, axis=1) < 1 / 4]
    END = valid_idxs[np.sum(points_valid, axis=1) > 7 / 4]

    A012 = valid_idxs[(np.sum(points_valid, axis=1) > 1 / 4) & (np.sum(points_valid, axis=1) < 1)]
    A0 = A012[(points[A012, 1] - points[A012, 0] < 1 / 2) & (points[A012, 1] - points[A012, 0] > -1 / 2)]
    A1 = A012[points[A012, 1] - points[A012, 0] > 1 / 2]
    A2 = A012[points[A012, 1] - points[A012, 0] < -1 / 2]

    B012 = valid_idxs[(np.sum(points_valid, axis=1) < 7 / 4) & (np.sum(points_valid, axis=1) > 1)]
    B0 = B012[(points[B012, 1] - points[B012, 0] < 1 / 2) & (points[B012, 1] - points[B012, 0] > -1 / 2)]
    B1_ = B012[points[B012, 1] - points[B012, 0] > 1 / 2]
    B2_ = B012[points[B012, 1] - points[B012, 0] < -1 / 2]

    paths = []
    G_undir = G.to_undirected()
    i = 0
    while len(paths) < m:
        v_begin = np.random.choice(BEGIN)
        if i % 3 == 0:
            v_1 = np.random.choice(A0)
            v_2 = np.random.choice(B0)
        elif i % 3 == 1:
            v_1 = np.random.choice(A1)
            v_2 = np.random.choice(B1_)
        else:
            v_1 = np.random.choice(A2)
            v_2 = np.random.choice(B2_)
        v_end = np.random.choice(END)

        path = nx.shortest_path(G_undir, v_begin, v_1)[:-1] + \
               nx.shortest_path(G_undir, v_1, v_2)[:-1] + \
               nx.shortest_path(G_undir, v_2, v_end)
        if len(path) == len(set(path)):
            paths.append(path)
            i += 1

    return G_undir, paths

def split_paths(paths):
    """
    Truncates paths, then splits each into prefix + suffix
    """
    paths_truncated = [p[:4 + np.random.choice(range(2, len(p) - 4))] for p in paths]
    prefixes = [p[:-2] for p in paths_truncated]
    suffixes = [p[-2:] for p in paths_truncated]
    last_nodes = [p[-1] for p in prefixes]
    return prefixes, suffixes, last_nodes

def conditional_incidence_matrix(B1, Nv, D):
    '''
    B1: numpy array
    Nv: row indices of B1 to extract
    D: max degree, for zero padding
    '''
    B_cond = np.zeros([D,B1.shape[1]])
    B_cond[:len(Nv),:] = B1[Nv]
    return B_cond

def generate_Bconds(G_undir, B1, last_nodes, max_degree):
    """
    Generates the conditional incidence matrix for each "last node" in a path, padded to the size of the max degree
    """
    B_conds = []
    for n in last_nodes:
        n_nbrs = np.array(sorted(G_undir[n]))
        B_cond = conditional_incidence_matrix(B1, n_nbrs, max_degree)
        B_conds.append(B_cond)
    return B_conds

def neighborhood(G, v):
    '''
    G: networkx undirected graph
    v: node label
    '''
    return np.array(sorted(G[v]))

def neighborhood_to_onehot(Nv, w, D):
    '''
    Nv: numpy array
    w: integer, presumably present in Nv
    D: max degree, for zero padding
    '''
    onehot = (Nv==w).astype(np.float)
    onehot_final = np.zeros(D)
    onehot_final[:onehot.shape[0]] = onehot
    return np.array([onehot_final]).T

def flow_to_path(flow, E, last_node):
    """
    Given a flow vector and the last node in the path, returns the path
    """
    # get edges in path
    path = [last_node]
    edges = set()
    for i in np.where(flow != 0)[0]:
        if flow[i] == 1:
            edges.add(E[i])
        elif flow[i] == -1:
            edges.add(E[i][::-1])
    # order edges
    cur_node = last_node
    # print(last_node, edges)
    while edges:
        next_edge = None
        for e in edges:
            if e[1] == cur_node:
                next_edge = e
        if next_edge is None:
            raise ValueError
        path.append(next_edge[0])
        edges.remove(next_edge)
        cur_node = next_edge[0]

    return path[::-1]

def path_to_flow(path, edge_to_idx, m):
    '''
    path: list of nodes
    E_lookup: dictionary mapping edge tuples to indices
    m: number of edges
    '''
    l = len(path)
    f = np.zeros([m,1])
    for j in range(l-1):
        v0 = path[j]
        v1 = path[j+1]
        if v0 < v1:
            k = edge_to_idx[(v0,v1)]
            f[k] = 1
        else:
            k = edge_to_idx[(v1,v0)]
            f[k] = -1
    return f

def path_dataset_1hop(G_undir, E, edge_to_idx, paths, max_degree):
    """
    Builds necessary matrices for 1-hop learning, from a list of paths
    """
    prefixes, suffixes, last_nodes = split_paths(paths)
    suffixes_1hop = [s[0] for s in suffixes]
    prefix_flows = np.array([path_to_flow(p, edge_to_idx, len(E)) for p in prefixes])
    # B_conds = generate_Bconds(G_undir, B1, last_nodes, max_degree)
    targets = np.array(
        [neighborhood_to_onehot(neighborhood(G_undir, prefix[-1]), suffix, max_degree) for prefix, suffix in
         zip(prefixes, suffixes_1hop)])
    # print(len([p for p in paths if len(p) != len(set(p))]))
    # print(prefixes[0], last_nodes[0])
    #
    #
    # print(prefixes[5], last_nodes[5])
    # # print(E)
    # print([E[x] for x in np.where(prefix_flows[5] != 0)[0]])
    # print()
    return prefix_flows, targets, last_nodes, suffixes_1hop

def path_dataset_2hop(G_undir, E, edge_to_idx, paths, max_degree):
    """
    Builds necessary matrices for 2-hop learning, from a list of paths
    """
    prefixes_1hop, suffixes, last_nodes_1hop = split_paths(paths)
    prefixes_2hop = [np.concatenate([p, [s[0]]]) for p, s in zip(prefixes_1hop, suffixes)]
    suffixes_2hop = [s[1] for s in suffixes]
    last_nodes_2hop = [s[0] for s in suffixes]
    prefix_flows_2hop = np.array([path_to_flow(p, edge_to_idx, len(E)) for p in prefixes_2hop])
    # B_conds_2hop = generate_Bconds(G_undir, B1, last_nodes_2hop, max_degree)
    targets_2hop = np.array(
        [neighborhood_to_onehot(neighborhood(G_undir, prefix[-1]), suffix, max_degree) for prefix, suffix in
         zip(prefixes_2hop, suffixes_2hop)])
    return prefix_flows_2hop, targets_2hop, last_nodes_2hop, suffixes_2hop

def generate_dataset(n, m, folder, holes=True):
    # generate graph
    G, V, E, faces, edge_to_idx, coords, valid_idxs = random_SC_graph(n, holes=holes)

    # B1, B2
    B1, B2 = incidience_matrices(G, V, E, faces)
    G_undir, paths = generate_random_walks(G, coords, valid_idxs, m=m)
    rev_paths = [path[::-1] for path in paths]

    # train / test masks
    train_mask = np.asarray([1] * int(len(paths) * 0.8) + [0] * int(len(paths) * 0.2))
    np.random.shuffle(train_mask)
    test_mask = 1 - train_mask


    max_degree = np.max([deg for n, deg in G_undir.degree()])
    # 1-hop
    prefix_flows_1hop, targets_1hop, last_nodes_1hop, suffixes_1hop = path_dataset_1hop(G_undir, E, edge_to_idx, paths, max_degree)

    # reversed 1hop
    rev_prefix_flows_1hop, rev_targets_1hop, rev_last_nodes_1hop, rev_suffixes_1hop = path_dataset_1hop(G_undir, E, edge_to_idx, rev_paths, max_degree)

    dataset_1hop = [prefix_flows_1hop, B1, B2, targets_1hop, train_mask, test_mask, G_undir, last_nodes_1hop, suffixes_1hop, rev_prefix_flows_1hop, rev_targets_1hop, rev_last_nodes_1hop, rev_suffixes_1hop]


    # 2-hop
    prefix_flows_2hop, targets_2hop, last_nodes_2hop, suffixes_2hop = path_dataset_2hop(G_undir, E, edge_to_idx, paths, max_degree)
    rev_prefix_flows_2hop, rev_targets_2hop, rev_last_nodes_2hop, rev_suffixes_2hop = path_dataset_2hop(G_undir, E, edge_to_idx, rev_paths, max_degree)

    dataset_2hop = [prefix_flows_2hop, B1, B2, targets_2hop, train_mask, test_mask, G_undir, last_nodes_2hop, suffixes_2hop, rev_prefix_flows_2hop, rev_targets_2hop, rev_last_nodes_2hop, rev_suffixes_2hop]

    # save datasets
    folder_1hop = 'trajectory_data_1hop_' + folder
    folder_2hop = 'trajectory_data_2hop_' + folder
    try:
        os.mkdir(folder_1hop), os.mkdir(folder_2hop)
    except:
        pass

    filenames = ('flows_in', 'B1', 'B2', 'targets', 'train_mask', 'test_mask', 'G_undir', 'last_nodes', 'target_nodes', 'rev_flows_in', 'rev_targets', 'rev_last_nodes', 'rev_target_nodes')
    for arr_1hop, arr_2hop, filename in zip(dataset_1hop, dataset_2hop, filenames):
        if filename == 'G_undir':
            nx.readwrite.gpickle.write_gpickle(G_undir, os.path.join(folder_1hop, filename + '.pkl'))
            nx.readwrite.gpickle.write_gpickle(G_undir, os.path.join(folder_2hop, filename + '.pkl'))
        else:
            np.save(os.path.join(folder_1hop, filename + '.npy'), arr_1hop)
            np.save(os.path.join(folder_2hop, filename + '.npy'), arr_2hop)

def load_dataset(folder):
    """
    Loads training data from trajectory_data folder
    """
    file_paths = [os.path.join(folder, ar + '.npy') for ar in ('flows_in', 'B1', 'B2', 'targets', 'train_mask',
                                                               'test_mask', 'G_undir', 'last_nodes', 'target_nodes')]
    G_undir = nx.readwrite.gpickle.read_gpickle(file_paths[6][:-4] + '.pkl')
    remap = {node: int(node) for node in G_undir.nodes}
    G_undir = nx.relabel_nodes(G_undir, remap)
    # print(B_matrices[0][10])

    return np.load(file_paths[0]), [np.load(p) for p in file_paths[1:3]], np.load(file_paths[3]), \
           np.load(file_paths[4]), np.load(file_paths[5]), G_undir, np.load(file_paths[7]), np.load(file_paths[8])


# generate_dataset(400, 1000, 'schaub')