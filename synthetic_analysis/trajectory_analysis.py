"""
Usage:
python3 trajectory_analysis.py ...
Argument defaults:
   'epochs': 1000; # of training epochs
   'learning_rate': 0.001; starting learning rate
   'batch_size': 100; # of samples per batch (randomly selected)
   'hidden_layers': [(3, 16), (3, 16), (3, 16)]; each tuple is a layer (# of shift matrices, # of channels in layer)
   'describe': 0; ask for a description of this test, print the description at the end
   'reverse': 0;  also compute accuracy over the test set, but reversed
   'load_data': 1; if 0, generate new data; if 1, load data from folder set in data_folder_suffix
   'load_model': 0; if 0, train a new model, if 1, load model from file model_name.npy. Must set hidden_layers regardless of choice
   'markov': 0; include tests using a 2nd-order Markov model
   'model_name': 'model'; name of model to use when load_model = 1
   'regional': 0; if 1 and load_model = 1, trains a model over upper graph region and tests over lower region
   'flip_edges': 0; if 1, flips orientation of a random subset of edges. with tanh activation, should perform equally
   'data_folder_suffix': 'working'; set suffix of folder to import data from (trajectory_data_Nhop_suffix)
   'multi_graph': '': if not '', also tests on paths over the graph with the folder suffix set here
   'holes': 1; if generation new data, sets whether the graph should have holes
   }

Examples:
    python3 trajectory_analysis.py -model_name tanh -reverse 1 -epochs 1100 -load_model 1 -multi_graph no_holes
        -loads model tanh.npy from models folder, tests it over reversed test set, and also tests over another graph saved in trajectory_data_Nhop_no_holes
    python3 trajectory_analysis.py load_data 0 -holes 0 -model_name tanh_no_holes -hidden_layers [(3, 32), (3,16)] -data_folder_suffix no_holes2
        -generates a new graph with no holes; saves dataset to trajectory_data_Nhop_no_holes2;
            trains a new model with 2 layers (32 and 16 channels, respectively) for 1100 epochs, and saves its weights to tanh_no_holes.npy
    python3 trajectory_analysis.py -load_data 0 -holes 1 -data_folder_suffix holes
        -make a dataset with holes, save with folder suffix holes (just stop the program once training starts if you just want to make a new dataset)
    python3 trajectory_analysis.py load_data 0 -holes 0 -data_folder_suffix no_holes -model_name tanh_no_holes -multi_graph holes
        -create a dataset using folder suffix no_holes, train a model over it using default settings, and test it over the graph with data folder suffix holes
Results:

-hidden_layers = [(3,16),(3,16),(3,16)], learning rate = 0.001, epochs = 1000
    standard train/test splits:
        normal shifts:
            activation = relu
                train loss: 0.828606 -- train acc 0.691 -- test loss 1.304349 -- test acc 0.545
                2hop binary: 0.14874999 0.12
                2hop dist: 0.233 0.182
                2-target: 0.856 0.853
                reversed test loss: 1.516947, Test acc: 0.475
                -Result: flipping orientations of random edges severely worsens performance
                    Train loss: 2.352922, Train acc: 0.394
                    Test loss: 2.375667, Test acc: 0.425


        --> activation = tanh
                train loss: 1.067673 -- train acc 0.607 -- test loss 1.240596 -- test acc 0.580
                2hop dist: 0.183 0.163
                2-target: 0.815, 0.815
                reversed test loss: 1.013826, Test acc: 0.580
                -Result: flipping orientations of random edges does not affect performance

            activation = tanh, epochs = 2000
                train loss: 1.012932 -- train acc 0.624 -- test loss 1.199164 -- test acc 0.580
                2hop dist: 0.180 0.171
                2-target: 0.72375 0.74
                reversed test loss: 1.072159, Test acc: 0.530

            activation = sigmoid, epochs = 1000
                train loss: 1.226788 -- train acc 0.577 -- test loss 1.225680 -- test acc 0.520
                Multi hop accs: [0.174, 0.167]
                2-target accs: 0.67875 0.755
                Test loss: 1.243790, Test acc: 0.480

        shifts = L_lower, L_lower:
            activation = relu
                train loss: 1.157995 -- train acc 0.556 -- test loss 1.311329 -- test acc 0.495
                2hop dist: 0.179 0.158
                2-target: 0.72625 0.745
                reversed test loss: 1.672432, Test acc: 0.325

            activation = tanh:
                train loss: 1.336956 -- train acc 0.524 -- test loss 1.304922 -- test acc 0.525
                2hop dist: 0.121 0.126
                2-target accs: 0.764375 0.7775
                Reversed Test loss: 1.164868, Test acc: 0.545
            activation = tanh, epochs = 1500:
                train loss: 1.285150 -- train acc 0.522 -- test loss 1.258232 -- test acc 0.535
                2hop dist: 0.108 0.118S
                2-target: 0.7075 0.75
                reversed test loss: 1.219265, Test acc: 0.520
            activation = tanh, epochs = 2000:
                train loss: 1.231843 -- train acc 0.529 -- test loss 1.237455 -- test acc 0.535
                2-hop dist: 0.130 0.139
                2-target accs: 0.71875 0.695
                reversed test loss: 1.127183, Test acc: 0.520

    train on upper paths, test on lower paths:
        normal shifts:
            train loss: 0.883628 -- train acc 0.676 -- test loss 1.685690 -- test acc 0.517
            2-hop dist: 0.220, 0.161
            2-target: 0.721 0.679

        activation = tanh:
            train loss: 1.403847 -- train acc 0.492 -- test loss 1.324031 -- test acc 0.565
            Multi hop accs: [0.083, 0.105]
            2-target accs: 0.640 0.703

        shifts = L_lower, L_lower:
            train loss: 1.402770 -- train acc 0.523 -- test loss 1.964938 -- test acc 0.255
            2-hop dist: 0.125, 0.05
            2-target: 0.667 0.529

    train on normal graph, test on a different one:
        same structure:
            normal shifts, activation = tanh (model = tanh.npy)
                Test loss: 1.442726, Test acc: 0.514
                2-target accs: 0.76 0.7925
            shifts = L_lower, L_lower
                Test loss: 1.502356, Test acc: 0.454
                2-target accs: 0.758125 0.75

        no holes:
            normal shifts, activation = tanh
                Test loss: 1.678532, Test acc: 0.445
                2-target accs: 0.744375 0.6825
            shifts = L_lower, L_lower
                Test loss: 1.700533, Test acc: 0.379
                2-target accs: 0.701875 0.6675

# todo
# 3-hop?
# boomerang flows?
# generalization tests:
#   train / test on different graphs (done)
#   try graphs with more holes
#   train w/holes, test without holes (done)


# orientation flip:
    B1 @ F
    F L1_lower F (flip each row and column in F)
    F L1_upper F
    flow @ F
    use same weights; should be the same only for tanh, not for relu


"""
import os

import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as onp


# from synthetic_analysis.synthetic_sc_walk import load_training_data, generate_training_data, save_training_data
# from synthetic_analysis.hodge_trajectory_model import Hodge_GCN
try:
    from synthetic_analysis.synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
    from synthetic_analysis.hodge_trajectory_model import Hodge_GCN
    from synthetic_analysis.markov_model import Markov_Model
except Exception:
    from synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
    from hodge_trajectory_model import Hodge_GCN
    from markov_model import Markov_Model

import sys



def hyperparams():
    """
    Parse hyperparameters from command line

    -hyperparam value

    For hidden_layers, input [(3, 8), (3, 8)] as 3_8_3_8
    """
    args = sys.argv
    hyperparams = {'epochs': 1000,
                   'learning_rate': 0.001,
                   'batch_size': 100,
                   'hidden_layers': [(3, 16), (3, 16), (3, 16)],
                   'describe': 0,
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
                print(nums)
                hyperparams['hidden_layers'] = []
                for j in range(0, len(nums), 2):
                    hyperparams['hidden_layers'] += [(nums[j], nums[j + 1])]
            elif args[i][1:] in ['model_name', 'data_folder_suffix', 'multi_graph']:
                hyperparams[args[i][1:]] = str(args[i+1])
            else:
                hyperparams[args[i][1:]] = float(args[i+1])
    print(hyperparams)
    return hyperparams

HYPERPARAMS = hyperparams()

# Define a model
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def hodge_parallel_variable(weights, S_lower, S_upper, Bcond_func, last_node, flow):
    """
    Hodge parallel model with variable number of layers
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

def hodge_parallel(weights, S0, S1, Bcond, flows):
    W00, W10, W20, W01, W11, W21, Wf = weights

    g0 = np.dot(flows, W00) + np.dot(np.dot(S0, flows), W10) + np.dot(np.dot(S1, flows), W20)
    g1 = relu(g0)

    h0 = np.dot(g1, W01) + np.dot(np.dot(S0, g1), W11) + np.dot(np.dot(S1, g1), W21)
    h1 = relu(h0)

    logits = np.dot(np.dot(Bcond, h1), Wf)

    return logits - logsumexp(logits)


def data_setup(hops=(1,), load=True, folder_suffix='working'):
    """
    Imports and sets up flow, target, and shift matrices for model training. Supports generating data for multiple hops
        at once

    todo organize
    """

    inputs_all, y_all, target_nodes_all = [], [], []

    if HYPERPARAMS['flip_edges']:
        # Flip orientation of a random subset of edges
        onp.random.seed(1)
        _, _, _, _, _, G_undir, _, _ = load_dataset('trajectory_data_1hop_' + folder_suffix)
        flips = onp.random.choice([1, -1], size=len(G_undir.edges), replace=True, p=[0.8, 0.2])
        F = np.diag(flips)
    if not load:
        # Generate data
        generate_dataset(400, 1000, folder=folder_suffix, holes=HYPERPARAMS['holes'])
        raise Exception
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
    print(last_nodes[0])
    max_degree = max(G_undir.degree, key=lambda x: x[1])[1]
    nbrhoods_dict = {node: onp.array(list(map(int, G_undir[node]))) for node in
                     map(int, sorted(G_undir.nodes))}
    n_nbrs = onp.array([len(nbrhoods_dict[n]) for n in last_nodes])

    # Bconds function
    nbrhoods = np.array([list(sorted(G_undir[n])) + [-1] * (max_degree - len(G_undir[n])) for n in range(max(G_undir.nodes) + 1)])
    nbrhoods = nbrhoods

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
        Nv = nbrhoods[n]
        return B1_jax[Nv]

    for i in range(len(inputs_all)):
        inputs_all[i][0] = Bconds_func



    return inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes

def train_model():
    """
    Trains a model to predict the next node in each input path (represented as a flow)
    """
    # Hyperparameters (from args)



    inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes = data_setup(hops=(1,2), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['data_folder_suffix'])

    (inputs_1hop, inputs_2hop), (y_1hop, y_2hop) = inputs_all, y_all

    last_nodes = inputs_1hop[1]



    if HYPERPARAMS['describe'] == 1:
        desc = input("Describe this test: ")

    in_axes = tuple([None, None, None, None, 0, 0])

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

        raise Exception

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

    # Create model
    hodge = Hodge_GCN(HYPERPARAMS['epochs'], HYPERPARAMS['learning_rate'], HYPERPARAMS['batch_size'])
    hodge.setup(hodge_parallel_variable, HYPERPARAMS['hidden_layers'], shifts, inputs_1hop, y_1hop, in_axes, train_mask)

    if HYPERPARAMS['load_model']:
        hodge.weights = onp.load('models/' + HYPERPARAMS['model_name'] + '.npy', allow_pickle=True)
        if HYPERPARAMS['epochs'] != 1000:
            # train model for additional epochs
            hodge.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs)
            try:
                os.mkdir('models')
            except:
                pass
            onp.save('models/' + HYPERPARAMS['model_name'], hodge.weights)

        (train_loss, train_acc), (test_loss, test_acc) = hodge.test(inputs_1hop, y_1hop, train_mask, n_nbrs), \
                                                         hodge.test(inputs_1hop, y_1hop, test_mask, n_nbrs)

    else:
        # Train either on upper region only or all data
        if HYPERPARAMS['regional']:
            train_mask = np.array([1 if i % 3 == 1 else 0 for i in range(len(y_1hop))])
            test_mask = np.array([1 if i % 3 == 2 else 0 for i in range(len(y_1hop))])

        train_loss, train_acc, test_loss, test_acc = hodge.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs)

        try:
            os.mkdir('models')
        except:
            pass
        onp.save('models/' + HYPERPARAMS['model_name'], hodge.weights)

    if HYPERPARAMS['reverse']:
        rev_flows_in, rev_targets_1hop, rev_targets_2hop, rev_last_nodes = \
            onp.load('trajectory_data_1hop_working/rev_flows_in.npy'), onp.load('trajectory_data_1hop_working/rev_targets.npy'), \
            onp.load('trajectory_data_2hop_working/rev_targets.npy'), onp.load('trajectory_data_1hop_working/rev_last_nodes.npy')
        rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
        print('Reversed ', end='')
        hodge.test([inputs_1hop[0], rev_last_nodes, rev_flows_in], rev_targets_1hop, test_mask, rev_n_nbrs)

    if HYPERPARAMS['multi_graph'] != '':
        inputs_all_o, y_all_o, train_mask_o, test_mask_o, shifts_o, G_undir_o, E_lookup_o, nbrhoods_o, n_nbrs_o, target_nodes_all_o, prefixes_o = data_setup(
            hops=(1, 2), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['multi_graph'])

        (inputs_1hop_o, inputs_2hop_o), (y_1hop_o, y_2hop_o) = inputs_all_o, y_all_o

        last_nodes_o = inputs_1hop[1]
        print('Different graph ' + HYPERPARAMS['multi_graph'], end=' ')
        hodge.shifts = shifts_o
        hodge.test(inputs_1hop_o, y_1hop_o, onp.array([1] * len(y_1hop_o)), n_nbrs_o)
        train_2target_o, test_2target_o = hodge.two_target_accuracy(shifts_o, inputs_1hop_o, y_1hop_o, train_mask_o, n_nbrs_o), \
                                      hodge.two_target_accuracy(shifts_o, inputs_1hop_o, y_1hop_o, test_mask_o, n_nbrs_o)
        print('2-target accs:', train_2target_o, test_2target_o)
        print('Multi hop accs:',
              hodge.multi_hop_accuracy_dist(shifts_o, inputs_1hop_o, target_nodes_all_o[1], [train_mask_o, test_mask_o], nbrhoods_o,
                                            E_lookup_o, last_nodes_o, prefixes_o, 2))
        hodge.shifts = shifts

    print('standard test set:')
    train_2target, test_2target = hodge.two_target_accuracy(shifts, inputs_1hop, y_1hop, train_mask, n_nbrs), \
                                  hodge.two_target_accuracy(shifts, inputs_1hop, y_1hop, test_mask, n_nbrs)

    print('2-target accs:', train_2target, test_2target)
    print('Multi hop accs:',
          hodge.multi_hop_accuracy_dist(shifts, inputs_1hop, target_nodes_all[1], [train_mask, test_mask], nbrhoods,
                                        E_lookup, last_nodes, prefixes, 2))

    if HYPERPARAMS['describe'] == 1:
        print(desc)

if __name__ == '__main__':
    train_model()