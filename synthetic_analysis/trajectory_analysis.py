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


"""

import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as onp
from synthetic_analysis.synthetic_sc_walk import load_training_data
from synthetic_analysis.hodge_trajectory_model import Hodge_GCN

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
    n_layers = (len(weights) - 1) / (len(shifts) + 1)
    assert n_layers % 1 == 0, 'wrong number of weights'

    cur_out = flows
    for i in range(int(n_layers)):
        cur_out = np.dot(cur_out, weights[i * 3]) \
                  + np.dot(np.dot(S_lower, cur_out), weights[i*3 + 1]) \
                  + np.dot(np.dot(S_upper, cur_out), weights[i*3 + 2])

        cur_out = relu(cur_out)

    logits = np.dot(np.dot(Bcond, cur_out), weights[-1])
    return logits - logsumexp(logits)

def hodge_parallel(weights, S0, S1, Bcond, flows):
    W00, W10, W20, W01, W11, W21, Wf = weights

    g0 = np.dot(flows, W00) + np.dot(np.dot(S0, flows), W10) + np.dot(np.dot(S1, flows), W20)
    g1 = relu(g0)

    h0 = np.dot(g1, W01) + np.dot(np.dot(S0, g1), W11) + np.dot(np.dot(S1, g1), W21)
    h1 = relu(h0)

    logits = np.dot(np.dot(Bcond, h1), Wf)

    return logits - logsumexp(logits)

# X, B_matrices, y, train_mask, test_mask = generate_training_data(400, 1000)
# save_training_data(X, *B_matrices, y, train_mask, test_mask)

# Load data
X, B_matrices, y, train_mask, test_mask = load_training_data('trajectory_data')
B1, B2, Bconds = B_matrices
inputs = [Bconds, X]


# Define shifts
L1_lower = B1.T @ B1
L1_upper = B2 @ B2.T
shifts = [L1_lower, L1_upper]

# train & test splits
def mask(A, m):
    """
    Masks a 3-D array along its first axis

    :param A: 3D array
    :param m: 1-D binary array of length A.shape[0]
    """
    return (onp.multiply(A.transpose(), m)).transpose()

X_train = mask(X, train_mask)
X_test = mask(X, test_mask)

inputs_train = [Bconds, X_train]
inputs_test = [Bconds, X_test]


y_train = mask(y, train_mask)
y_test = mask(y, test_mask)
# print(X.shape, X_train.shape, y.shape, y_train.shape)


# Hyperparameters (from args)
epochs, learning_rate, batch_size, hidden_layers, describe = hyperparams()

if describe == 1:
    desc = input("Describe this test: ")

in_axes = tuple([None] * (len(shifts) + 1) + [0] * len(inputs_train))


# Create model
hodge = Hodge_GCN(epochs, learning_rate, batch_size, verbose=True)

# Train
loss, acc = hodge.train(hodge_parallel_variable, hidden_layers, shifts, inputs, y, in_axes, train_mask)

# Test
test_loss, test_acc = hodge.test(inputs, y, test_mask)

if describe == 1:
    print(desc)