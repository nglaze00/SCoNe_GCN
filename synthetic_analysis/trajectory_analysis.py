"""
Results:
Over training data:
    -Defaults (Epochs: 100, learning rate: 0.0005, batch size: 100, hidden_layers = (3,8),(3,8), shifts = L_lower, L_upper
        loss: 1.263, acc: 0.524
    -hidden_layers = [(3,8),(3,8),(3,8)]
        loss: 1.055, acc: 0.605
    -hidden_layers = [(3,16),(3,16),(3,16)]
        loss: 0.881, acc: 0.648
    -hidden_layers = [(3,16),(3,16),(3,16)], epochs = 200
        loss: 0.768, acc: 0.697
    -step_size = 0.0001
        loss: 1.303, acc: 0.517
    -step_size = 0.01
        loss: 1.262, acc: 0.517
    -L_upper = 0
        loss: 1.668, acc: 0.318
    -L_upper = L_lower
        loss: 1.658, acc: 0.368
    -L_lower = 0, L_upper = 0 (no shifts)
        no improvement
    -L_lower = 0
        no improvement

With train / test splits:
    -Defaults:
        Training loss: 1.308, training acc: 0.514
        Test loss: 1.312, Test acc: 0.520
    -hidden_layers = [(3,16),(3,16),(3,16)], epochs = 200
        Training loss: 0.7154, training acc: 0.710
        Test loss: 0.8287, Test acc: 0.685
    -hidden_layers = [(3,16),(3,16),(3,16)], epochs = 200; L_upper = L_lower
        gets to loss: 1.207, acc: 0.544, then NaNs


# todo
#   predict distributions, multihop, etc (stuff from paper)
#   try other accuracy measurements
#   Experiment: reversing flows, then testing w/ ours and GRETEL

## Other accuracy measurements:
    todo 2-target acc

## Multi hop:
    todo regenerate
"""

import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as onp

# from synthetic_analysis.synthetic_sc_walk import load_training_data, generate_training_data, save_training_data
# from synthetic_analysis.hodge_trajectory_model import Hodge_GCN
try:
    from synthetic_analysis.synthetic_sc_walk import load_training_data, generate_training_data, save_training_data
    from synthetic_analysis.hodge_trajectory_model import Hodge_GCN
except Exception:
    from synthetic_sc_walk import load_training_data, generate_training_data, save_training_data
    from hodge_trajectory_model import Hodge_GCN

import sys

def hyperparams():
    """
    Parse hyperparameters from command line

    -hyperparam value

    For hidden_layers, input [(3, 8), (3, 8)] as 3_8_3_8
    """
    args = sys.argv
    hyperparams = {'epochs': 100,
                   'learning_rate': 0.0005,
                   'batch_size': 100,
                   'hidden_layers': [(3, 8), (3, 8)],
                   'describe': 0}

    for i in range(len(args) - 1):
        if args[i][0] == '-':
            if args[i][1:] == 'hidden_layers':
                nums = list(map(int, args[i + 1].split("_")))
                print(nums)
                hyperparams['hidden_layers'] = []
                for j in range(0, len(nums), 2):
                    hyperparams['hidden_layers'] += [(nums[j], nums[j + 1])]

            else:
                hyperparams[args[i][1:]] = float(args[i+1])
    print(hyperparams)
    return hyperparams['epochs'], hyperparams['learning_rate'], hyperparams['batch_size'], hyperparams['hidden_layers'], \
            hyperparams['describe']

# Define a model
def relu(x):
    return np.maximum(x, 0)

def hodge_parallel_variable(weights, S_lower, S_upper, Bcond, flows):
    """
    Hodge parallel model with variable number of layers
    """
    n_layers = (len(weights) - 1) / 3
    assert n_layers % 1 == 0, 'wrong number of weights'

    cur_out = flows
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i * 3] \
                  + S_lower @ cur_out @ weights[i*3 + 1] \
                  + S_upper @ cur_out @ weights[i*3 + 2]

        cur_out = relu(cur_out)

    logits = Bcond @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

def hodge_parallel(weights, S0, S1, Bcond, flows):
    W00, W10, W20, W01, W11, W21, Wf = weights

    g0 = np.dot(flows, W00) + np.dot(np.dot(S0, flows), W10) + np.dot(np.dot(S1, flows), W20)
    g1 = relu(g0)

    h0 = np.dot(g1, W01) + np.dot(np.dot(S0, g1), W11) + np.dot(np.dot(S1, g1), W21)
    h1 = relu(h0)

    logits = np.dot(np.dot(Bcond, h1), Wf)

    return logits - logsumexp(logits)


def data_setup(hops=(1,), load=True):
    """
    Imports and sets up flow, target, and shift matrices for model training. Supports generating data for multiple hops
        at once
    """
    # X, B_matrices, y, train_mask, test_mask = generate_training_data(400, 1000)
    # save_training_data(X, *B_matrices, y, train_mask, test_mask)
    inputs_all, y_all = [], []
    if not load:
        # Generate data
        Xs, B_matrices, ys, train_mask, test_mask = generate_training_data(400, 1000, hops=hops)

    for i in range(len(hops)):
        if load:
            # Load data
            folder = 'trajectory_data_' + str(hops[i]) + 'hop'
            X, B_matrices, y, train_mask, test_mask = load_training_data(folder)
            B1, B2, Bconds = B_matrices
        else:
            B1, B2, Bconds = B_matrices[0], B_matrices[1], B_matrices[2][i]
            X, y = Xs[i], ys[i]
            save_training_data(Xs[i], B1, B2, Bconds, ys[i], train_mask, test_mask, 'trajectory_data_' + str(hops[i]) + 'hop')

        inputs_all.append([Bconds, X])
        y_all.append(y)

        # Define shifts
        L1_lower = B1.T @ B1
        L1_upper = B2 @ B2.T
        shifts = [L1_lower, L1_upper]

    # Build E_lookup for multi-hop training
    E_lookup = {}
    B1_idxs = onp.nonzero(B1.T)  # [columns], [rows]
    for i in range(0, len(B1_idxs[0]), 2):
        edge = (B1_idxs[1][i], B1_idxs[0][i])
        E_lookup[edge] = i
        E_lookup[(edge[1], edge[0])] = i

    return inputs_all, y_all, train_mask, test_mask, shifts, E_lookup

def train_model():
    """
    Trains a model to predict the next node in each input path (represented as a flow)
    """
    inputs_all, y_all, train_mask, test_mask, shifts, E_lookup = data_setup(hops=(1,2), load=True)



    (inputs_1hop, inputs_2hop), (y_1hop, y_2hop) = inputs_all, y_all


    # Hyperparameters (from args)
    epochs, learning_rate, batch_size, hidden_layers, describe = hyperparams()

    if describe == 1:
        desc = input("Describe this test: ")

    in_axes = tuple([None] * (len(shifts) + 1) + [0] * len(inputs_1hop))


    # Create model
    hodge = Hodge_GCN(epochs, learning_rate, batch_size, verbose=True)

    # Train
    loss, acc = hodge.train(hodge_parallel_variable, hidden_layers, shifts, inputs_1hop, y_1hop, in_axes, train_mask)

    # Test
    test_loss, test_acc = hodge.test(inputs_1hop, y_1hop, test_mask)



    if describe == 1:
        print(desc)

if __name__ == '__main__':
    train_model()