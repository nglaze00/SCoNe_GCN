"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)

Code for converting ocean drifter data from Schaub's format to ours.
"""

import h5py
from trajectory_analysis.synthetic_data_gen import *

dataset_folder = 'buoy'

f = h5py.File('dataBuoys.jld2', 'r')
print(f.keys())

### Load arrays from file

## Graph

# elist (edge list)
edge_list = f['elist'][:] - 1 # 1-index -> 0-index

# tlist (triangle list)
face_list = f['tlist'][:] - 1

# NodeToHex (map node id <-> hex coords) # nodes are 1-indexed in data source
node_hex_map = [tuple(f[x][()]) for x in f['NodeToHex'][:]]
hex_node_map = {tuple(hex_coords): node for node, hex_coords in enumerate(node_hex_map)}


## trajectories

# coords
hex_coords = np.array([tuple(x) for x in f['HexcentersXY'][()]])

# nodes
traj_nodes = [[f[x][()] - 1 for x in f[ref][()]] for ref in f['TrajectoriesNodes'][:]]

#### Convert to SCoNe dataset

# generate graph + faces
G = nx.Graph()
G.add_edges_from([(edge_list[0][i], edge_list[1][i]) for i in range(len(edge_list[0]))])

V, E = np.array(sorted(G.nodes)), np.array([sorted(x) for x in sorted(G.edges)])
faces = np.array(sorted([[face_list[j][i] for j in range(3)] for i in range(len(face_list[0]))]))

edge_to_idx = {tuple(e): i for i, e in enumerate(E)}
coords = hex_coords
valid_idxs = np.arange(len(coords))

# B1, B2
B1, B2 = incidence_matrices(G, V, E, faces, edge_to_idx)

# Trajectories
G_undir = G.to_undirected()
stripped_paths = strip_paths(traj_nodes)
paths = [path[-10:] for path in stripped_paths if len(path) >= 5]

# Print graph info
print(np.mean([len(G[i]) for i in V]))
print('# nodes: {}, # edges: {}, # faces: {}'.format(*B1.shape, B2.shape[1]))
print('# paths: {}, # paths with prefix length >= 3: {}'.format(len(traj_nodes), len(paths)))

rev_paths = [path[::-1] for path in paths]

# Save graph image to file
color_faces(G, V, coords, faces_from_B2(B2, E), filename='madagascar_graph_faces_paths.pdf', paths=[paths[1], paths[48], paths[125]])

# train / test masks
np.random.seed(1)
train_mask = np.asarray([1] * round(len(paths) * 0.8) + [0] * round(len(paths) * 0.2))
np.random.shuffle(train_mask)
test_mask = 1 - train_mask

max_degree = np.max([deg for n, deg in G_undir.degree()])

## Consolidate dataset

# forward
prefix_flows_1hop, targets_1hop, last_nodes_1hop, suffixes_1hop, \
    prefix_flows_2hop, targets_2hop, last_nodes_2hop, suffixes_2hop = path_dataset(G_undir, E, edge_to_idx,
                                                                                paths, max_degree, include_2hop=True,
                                                                                truncate_paths=False)


# reversed
rev_prefix_flows_1hop, rev_targets_1hop, rev_last_nodes_1hop, rev_suffixes_1hop, \
    rev_prefix_flows_2hop, rev_targets_2hop, rev_last_nodes_2hop, rev_suffixes_2hop = path_dataset(G_undir, E, edge_to_idx,
                                                                                                rev_paths, max_degree,
                                                                                                include_2hop=True,
                                                                                                truncate_paths=False)

dataset_1hop = [prefix_flows_1hop, B1, B2, targets_1hop, train_mask, test_mask, G_undir, last_nodes_1hop,
                suffixes_1hop, rev_prefix_flows_1hop, rev_targets_1hop, rev_last_nodes_1hop, rev_suffixes_1hop]
dataset_2hop = [prefix_flows_2hop, B1, B2, targets_2hop, train_mask, test_mask, G_undir, last_nodes_2hop,
                suffixes_2hop, rev_prefix_flows_2hop, rev_targets_2hop, rev_last_nodes_2hop, rev_suffixes_2hop]

print('Train samples:', sum(train_mask))
print('Test samples:', sum(test_mask))

### Save datasets

folder_1hop = '../trajectory_analysis/trajectory_data_1hop_' + dataset_folder
folder_2hop = '../trajectory_analysis/trajectory_data_2hop_' + dataset_folder

try:
    os.mkdir(folder_1hop)
except:
    pass
try:
    os.mkdir(folder_2hop)
except:
    pass

# Save files
filenames = (
'flows_in', 'B1', 'B2', 'targets', 'train_mask', 'test_mask', 'G_undir', 'last_nodes', 'target_nodes', 'rev_flows_in',
'rev_targets', 'rev_last_nodes', 'rev_target_nodes')

for arr_1hop, arr_2hop, filename in zip(dataset_1hop, dataset_2hop, filenames):
    if filename == 'G_undir':
        nx.readwrite.gpickle.write_gpickle(G_undir, os.path.join(folder_1hop, filename + '.pkl'))
        nx.readwrite.gpickle.write_gpickle(G_undir, os.path.join(folder_2hop, filename + '.pkl'))
    else:
        np.save(os.path.join(folder_1hop, filename + '.npy'), arr_1hop)
        np.save(os.path.join(folder_2hop, filename + '.npy'), arr_2hop)


# Save prefixes file
edge_set = set()
for path in paths:
    for i in range(1, len(path)):
        edge = tuple(sorted(path[i-1:i+1]))
        edge_set.add(edge)

np.save(folder_1hop + '/prefixes.npy', [path[:-2] for path in paths])
