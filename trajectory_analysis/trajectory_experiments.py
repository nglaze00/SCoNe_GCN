"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)

This is where you actually train models. See below for docs:

Generate a synthetic graph, with holes in upper left and lower right regions, + paths over the graph:
    python3 synthetic_data_gen.py
    -Edit main of synthetic_data_gen.py to change graph size / # of paths

Train a SCoNe model on a dataset:
    python3 trajectory_experiments.py [args]

    Command to run standard training / experiment with defaults:
        python3 trajectory_experiments.py -data_folder_suffix suffix_here


Arguments + default values for trajectory_experiments.py:
   'epochs': 1000; # of training epochs
   'learning_rate': 0.001; starting learning rate
   'weight_decay': 0.00005; ridge regularization constant
   'batch_size': 100; # of samples per batch (randomly selected)
   'reverse': 0;  if 1, also compute accuracy over the test set, but reversed (Reverse experiment)
   'data_folder_suffix': 'schaub2'; set suffix of folder to import data from (trajectory_data_Nhop_suffix)
   'regional': 0; if 1, trains a model over upper graph region and tests over lower region (Transfer experiment)

   'hidden_layers': 3_16_3_16_3_16 (corresponds to [(3, 16), (3, 16), (3, 16)]; each tuple is a layer (# of shift matrices, # of units in layer) )
   'describe': 1; describes the dataset being used
   'load_data': 1; if 0, generate new data; if 1, load data from folder set in data_folder_suffix
   'load_model': 0; if 0, train a new model, if 1, load model from file model_name.npy. Must set hidden_layers regardless of choice
   'markov': 0; include tests using a 2nd-order Markov model
   'model_name': 'model'; name of model to use when load_model = 1

   'flip_edges': 0; if 1, flips orientation of a random subset of edges. with tanh activation, should perform equally

   'multi_graph': '': if not '', also tests on paths over the graph with the folder suffix set here
   'holes': 1; if generation new data, sets whether the graph should have holes

More examples:
    python3 trajectory_experiments.py -model_name tanh -reverse 1 -epochs 1100 -load_model 1 -multi_graph no_holes
        -loads model tanh.npy from models folder, tests it over reversed test set, and also tests over another graph saved in trajectory_data_Nhop_no_holes
    python3 trajectory_experiments.py load_data 0 -holes 0 -model_name tanh_no_holes -hidden_layers [(3, 32), (3,16)] -data_folder_suffix no_holes2
        -generates a new graph with no holes; saves dataset to trajectory_data_Nhop_no_holes2;
            trains a new model with 2 layers (32 and 16 channels, respectively) for 1100 epochs, and saves its weights to tanh_no_holes.npy
    python3 trajectory_experiments.py -load_data 0 -holes 1 -data_folder_suffix holes
        -make a dataset with holes, save with folder suffix holes (just stop the program once training starts if you just want to make a new dataset)
    python3 trajectory_experiments.py load_data 0 -holes 0 -data_folder_suffix no_holes -model_name tanh_no_holes -multi_graph holes
        -create a dataset using folder suffix no_holes, train a model over it using default settings, and test it over the graph with data folder suffix holes
"""
import os, sys

import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as onp

try:
    from trajectory_analysis.synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
    from trajectory_analysis.scone_trajectory_model import Scone_GCN
    from trajectory_analysis.markov_model import Markov_Model
except Exception:
    from synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
    from scone_trajectory_model import Scone_GCN
    from markov_model import Markov_Model

def hyperparams():
    """
    Parse hyperparameters from command line

    For hidden_layers, input [(3, 8), (3, 8)] as 3_8_3_8
    """
    args = sys.argv
    hyperparams = {'epochs': 1000,
                   'learning_rate': 0.001,
                   'weight_decay': 0.00005,
                   'batch_size': 100,
                   'hidden_layers': [(3, 16), (3, 16), (3, 16)],
                   'describe': 1,
                   'reverse': 0,
                   'load_data': 1,
                   'load_model': 0,
                   'markov': 0,
                   'model_name': 'model',
                   'regional': 0,
                   'flip_edges': 0,
                   'data_folder_suffix': 'working',
                   'multi_graph': '',
                   'holes': 1}

    for i in range(len(args) - 1):
        if args[i][0] == '-':
            if args[i][1:] == 'hidden_layers':
                nums = list(map(int, args[i + 1].split("_")))

                hyperparams['hidden_layers'] = []
                for j in range(0, len(nums), 2):
                    hyperparams['hidden_layers'] += [(nums[j], nums[j + 1])]
            elif args[i][1:] in ['model_name', 'data_folder_suffix', 'multi_graph']:
                hyperparams[args[i][1:]] = str(args[i+1])
            else:
                hyperparams[args[i][1:]] = float(args[i+1])
    return hyperparams

HYPERPARAMS = hyperparams()

### Model definition ###

# Activation functions
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# SCoNe function
def scone_func(weights, S_lower, S_upper, Bcond_func, last_node, flow):
    """
    SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / 3
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i * 3] \
                  + S_lower @ cur_out @ weights[i*3 + 1] \
                  + S_upper @ cur_out @ weights[i*3 + 2]

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

def data_setup(hops=(1,), load=True, folder_suffix='schaub'):
    """
    Imports and sets up flow, target, and shift matrices for model training. Supports generating data for multiple hops
        at once
    """

    inputs_all, y_all, target_nodes_all = [], [], []

    if HYPERPARAMS['flip_edges']:
        # Flip orientation of a random subset of edges
        onp.random.seed(1)
        _, _, _, _, _, G_undir, _, _ = load_dataset('trajectory_data_1hop_' + folder_suffix)
        flips = onp.random.choice([1, -1], size=len(G_undir.edges), replace=True, p=[0.8, 0.2])
        F = np.diag(flips)


    if not load:
        # Generate new data
        generate_dataset(400, 1000, folder=folder_suffix, holes=HYPERPARAMS['holes'])
        raise Exception('Data generation done')


    for h in hops:
        # Load data
        folder = 'trajectory_data_' + str(h) + 'hop_' + folder_suffix
        X, B_matrices, y, train_mask, test_mask, G_undir, last_nodes, target_nodes = load_dataset(folder)
        B1, B2 = B_matrices
        target_nodes_all.append(target_nodes)


        inputs_all.append([None, np.array(last_nodes), X])
        y_all.append(y)

        # Define shifts
        L1_lower = B1.T @ B1
        L1_upper = B2 @ B2.T
        if HYPERPARAMS['flip_edges']:
            L1_lower = F @ L1_lower @ F
            L1_upper = F @ L1_upper @ F

        shifts = [L1_lower, L1_upper]
        # shifts = [L1_lower, L1_lower]

    # Build E_lookup for multi-hop training
    e = onp.nonzero(B1.T)[1]
    edges = onp.array_split(e, len(e)/2)
    E, E_lookup = [], {}
    for i, e in enumerate(edges):
        E.append(tuple(e))
        E_lookup[tuple(e)] = i

    # set up neighborhood data
    last_nodes = inputs_all[0][1]

    max_degree = max(G_undir.degree, key=lambda x: x[1])[1]
    nbrhoods_dict = {node: onp.array(list(map(int, G_undir[node]))) for node in
                     map(int, sorted(G_undir.nodes))}
    n_nbrs = onp.array([len(nbrhoods_dict[n]) for n in last_nodes])

    # Bconds function
    nbrhoods = np.array([list(sorted(G_undir[n])) + [-1] * (max_degree - len(G_undir[n])) for n in range(max(G_undir.nodes) + 1)])
    nbrhoods = nbrhoods

    # load prefixes if they exist
    try:
        prefixes = list(np.load('trajectory_data_1hop_' + folder_suffix + '/prefixes.npy', allow_pickle=True))
    except:
        prefixes = [flow_to_path(inputs_all[0][-1][i], E, last_nodes[i]) for i in range(len(last_nodes))]

    B1_jax = np.append(B1, np.zeros((1, B1.shape[1])), axis=0)

    if HYPERPARAMS['flip_edges']:
        B1_jax = B1_jax @ F
        for i in range(len(inputs_all)):
            print(inputs_all[i][-1].shape)
            n_flows, n_edges = inputs_all[i][-1].shape[:2]
            inputs_all[i][-1] = inputs_all[i][-1].reshape((n_flows, n_edges)) @ F
            inputs_all[i][-1] = inputs_all[i][-1].reshape((n_flows, n_edges, 1))

    def Bconds_func(n):
        """
        Returns rows of B1 corresponding to neighbors of node n
        """
        Nv = nbrhoods[n]
        return B1_jax[Nv]

    for i in range(len(inputs_all)):
        inputs_all[i][0] = Bconds_func

    return inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes

def train_model():
    """
    Trains a model to predict the next node in each input path (represented as a flow)
    """

    # load dataset
    inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes = data_setup(hops=(1,2), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['data_folder_suffix'])

    (inputs_1hop, inputs_2hop), (y_1hop, y_2hop) = inputs_all, y_all

    last_nodes = inputs_1hop[1]

    in_axes = tuple([None, None, None, None, 0, 0])

    # Train Markov model
    if HYPERPARAMS['markov'] == 1:
        order = 1
        markov = Markov_Model(order)
        paths = onp.array([prefix + [target1, target2] for prefix, target1, target2 in zip(prefixes, target_nodes_all[0], target_nodes_all[1])], dtype='object')

        paths_train = paths[train_mask == 1]
        prefixes_train, target_nodes_1hop_train, target_nodes_2hop_train = onp.array(prefixes)[train_mask == 1], \
                                                                           target_nodes_all[0][train_mask == 1], \
                                                                           target_nodes_all[1][train_mask == 1]
        prefixes_test, target_nodes_1hop_test, target_nodes_2hop_test = onp.array(prefixes, dtype='object')[test_mask == 1], \
                                                    target_nodes_all[0][test_mask == 1], \
                                                    target_nodes_all[1][test_mask == 1]

        # forward paths
        markov.train(G_undir, paths_train)
        print("train accs")
        print(markov.test(prefixes_train, target_nodes_1hop_train, 1))
        print(markov.test(prefixes_train, target_nodes_2hop_train, 2))
        print(markov.test_2_target(prefixes_train, target_nodes_1hop_train))
        print("test accs")
        print(markov.test(prefixes_test, target_nodes_1hop_test, 1))
        print(markov.test(prefixes_test, target_nodes_2hop_test, 2))
        print(markov.test_2_target(prefixes_test, target_nodes_1hop_test))



        # reversed test paths
        rev_paths = [path[::-1] for path in paths]
        rev_prefixes = onp.array([p[:-2] for p in rev_paths], dtype='object')
        rev_prefixes_test = rev_prefixes[test_mask == 1]
        rev_target_nodes_1hop, rev_target_nodes_2hop = onp.array([p[-2] for p in rev_paths], dtype='object'), \
                                                       onp.array([p[-1] for p in rev_paths], dtype='object')
        rev_target_nodes_1hop_test = rev_target_nodes_1hop[test_mask == 1]
        rev_target_nodes_2hop_test = rev_target_nodes_2hop[test_mask == 1]
        print("Reversed test accs")
        print(markov.test(rev_prefixes_test, rev_target_nodes_1hop_test, 1))
        print(markov.test(rev_prefixes_test, rev_target_nodes_2hop_test, 2))

        # half forward, half backward
        fwd_mask = onp.array([True] * int(len(paths) / 2) + [False] * int(len(paths) / 2))
        onp.random.shuffle(fwd_mask)
        bkwd_mask = ~fwd_mask

        # mixed dataset
        mixed_paths = onp.concatenate((onp.array(paths)[fwd_mask == 1], onp.array(rev_paths)[bkwd_mask == 1]))
        mixed_prefixes = onp.concatenate((onp.array(prefixes)[fwd_mask==1], rev_prefixes[bkwd_mask==1]))
        mixed_target_nodes_1hop = onp.concatenate((target_nodes_all[0][fwd_mask == 1], rev_target_nodes_1hop[bkwd_mask == 1]))
        mixed_target_nodes_2hop = onp.concatenate((target_nodes_all[1][fwd_mask == 1], rev_target_nodes_2hop[bkwd_mask == 1]))

        # train / test splits
        mixed_paths_train = mixed_paths[train_mask == 1]
        mixed_prefixes_train, mixed_prefixes_test = mixed_prefixes[train_mask == 1], mixed_prefixes[test_mask == 1]
        mixed_target_nodes_1hop_train, mixed_target_nodes_1hop_test = mixed_target_nodes_1hop[train_mask == 1], \
                                                                      mixed_target_nodes_1hop[test_mask == 1]
        mixed_target_nodes_2hop_train, mixed_target_nodes_2hop_test = mixed_target_nodes_2hop[train_mask == 1], \
                                                                      mixed_target_nodes_2hop[test_mask == 1]

        markov.train(G_undir, mixed_paths_train)

        print("Mixed train accs")
        print(markov.test(mixed_prefixes_train, mixed_target_nodes_1hop_train, 1))
        print(markov.test(mixed_prefixes_train, mixed_target_nodes_2hop_train, 2))
        print("Mixed test accs")
        print(markov.test(mixed_prefixes_test, mixed_target_nodes_1hop_test, 1))
        print(markov.test(mixed_prefixes_test, mixed_target_nodes_2hop_test, 2))

        # train on middle, test on middle
        mid_train_mask = [i % 3 == 0 and train_mask[i] == 1 for i in range(len(train_mask))]
        mid_test_mask = [i % 3 == 0 and test_mask[i] == 1 for i in range(len(test_mask))]

        mid_paths_train, mid_paths_test = paths[mid_train_mask], paths[mid_test_mask]
        mid_prefixes_train, mid_prefixes_test = [p[:-2] for p in mid_paths_train], [p[:-2] for p in mid_paths_test]

        mid_targets_1hop_train, mid_targets_1hop_test = target_nodes_all[0][mid_train_mask], \
                                                        target_nodes_all[0][mid_test_mask]
        mid_targets_2hop_train, mid_targets_2hop_test = target_nodes_all[1][mid_train_mask], \
                                                        target_nodes_all[1][mid_test_mask]

        markov.train(G_undir, mid_paths_train)
        print("Middle region train accs")
        print(markov.test(mid_prefixes_train, mid_targets_1hop_train, 1))
        print(markov.test(mid_prefixes_train, mid_targets_2hop_train, 2))
        print("Middle region test accs")
        print(markov.test(mid_prefixes_test, mid_targets_1hop_test, 1))
        print(markov.test(mid_prefixes_test, mid_targets_2hop_test, 2))


        # train on upper, test on lower
        paths_upper = [paths[i] for i in range(len(paths)) if i % 3 == 1]
        prefixes_upper = [p[:-2] for p in paths_upper]
        targets_1hop_upper = [target_nodes_all[0][i] for i in range(len(paths)) if i % 3 == 1]
        targets_2hop_upper = [target_nodes_all[1][i] for i in range(len(paths)) if i % 3 == 1]

        paths_lower = [paths[i] for i in range(len(paths)) if i % 3 == 2]
        prefixes_lower = [p[:-2] for p in paths_lower]
        targets_1hop_lower = [target_nodes_all[0][i] for i in range(len(paths)) if i % 3 == 2]
        targets_2hop_lower = [target_nodes_all[1][i] for i in range(len(paths)) if i % 3 == 2]

        markov.train(G_undir, paths_upper)
        print("Upper region train accs")
        print(markov.test(prefixes_upper, targets_1hop_upper, 1))
        print(markov.test(prefixes_upper, targets_2hop_upper, 2))
        print("Lower region accs")
        print(markov.test(prefixes_lower, targets_1hop_lower, 1))
        print(markov.test(prefixes_lower, targets_2hop_lower, 2))
        raise Exception

    # Initialize SCoNe model
    scone = Scone_GCN(HYPERPARAMS['epochs'], HYPERPARAMS['learning_rate'], HYPERPARAMS['batch_size'], HYPERPARAMS['weight_decay'])

    scone.setup(scone_func, HYPERPARAMS['hidden_layers'], shifts, inputs_1hop, y_1hop, in_axes, train_mask)

    if HYPERPARAMS['regional']:
        # Train either on upper region only or all data (synthetic dataset)
        # 0: middle, 1: top, 2: bottom
        train_mask = np.array([1 if i % 3 == 1 else 0 for i in range(len(y_1hop))])
        test_mask = np.array([1 if i % 3 == 2 else 0 for i in range(len(y_1hop))])

    # describe dataset
    if HYPERPARAMS['describe'] == 1:
        print('Graph nodes: {}, edges: {}, avg degree: {}'.format(len(G_undir.nodes), len(G_undir.edges),
                                                                  np.average([G_undir.degree[node] for node in
                                                                              G_undir.nodes])))
        print('Training paths: {}, Test paths: {}'.format(train_mask.sum(), test_mask.sum()))

    # load a model from file + train it more
    if HYPERPARAMS['load_model']:
        scone.weights = onp.load('models/' + HYPERPARAMS['model_name'] + '.npy', allow_pickle=True)
        if HYPERPARAMS['epochs'] != 0:
            # train model for additional epochs
            scone.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs)
            try:
                os.mkdir('models')
            except:
                pass
            onp.save('models/' + HYPERPARAMS['model_name'], scone.weights)

        (train_loss, train_acc), (test_loss, test_acc) = scone.test(inputs_1hop, y_1hop, train_mask, n_nbrs), \
                                                         scone.test(inputs_1hop, y_1hop, test_mask, n_nbrs)

    else:

        train_loss, train_acc, test_loss, test_acc = scone.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs)

        try:
            os.mkdir('models')
        except:
            pass
        onp.save('models/' + HYPERPARAMS['model_name'], scone.weights)

    # standard experiment
    print('standard test set:')
    train_2target, test_2target = scone.two_target_accuracy(shifts, inputs_1hop, y_1hop, train_mask, n_nbrs), \
                                  scone.two_target_accuracy(shifts, inputs_1hop, y_1hop, test_mask, n_nbrs)

    scone.test(inputs_1hop, y_1hop, test_mask, n_nbrs)
    print('2-target accs:', train_2target, test_2target)


    if HYPERPARAMS['reverse']:
        # reverse direction of test flows
        rev_flows_in, rev_targets_1hop, rev_targets_2hop, rev_last_nodes = \
            onp.load('trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_flows_in.npy'), onp.load('trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), \
            onp.load('trajectory_data_2hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), onp.load('trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_last_nodes.npy')
        rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
        print('Reverse experiment:')
        scone.test([inputs_1hop[0], rev_last_nodes, rev_flows_in], rev_targets_1hop, test_mask, rev_n_nbrs)



    # print('Multi hop accs:',
    #       scone.multi_hop_accuracy_dist(shifts, inputs_1hop, target_nodes_all[1], [train_mask, test_mask], nbrhoods,
    #                                     E_lookup, last_nodes, prefixes, 2))


if __name__ == '__main__':
    train_model()