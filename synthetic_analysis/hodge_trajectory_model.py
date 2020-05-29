import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
from jax.experimental.optimizers import adam

class Hodge_GCN():
    def __init__(self, epochs, step_size, batch_size, verbose=True):
        """
        :param epochs: # of training epochs
        :param step_size: step size for use in training model
        :param batch_size: # of data points to train over in each gradient step
        :param verbose: whether to print training progress
        """

        self.random_targets = None

        self.trained = False
        self.model = None
        self.shifts = None
        self.weights = None

        self.epochs = int(epochs)
        self.step_size = step_size
        self.batch_size = int(batch_size)

        self.verbose = verbose

    def loss(self, weights, inputs, y, mask):
        """
        Computes cross-entropy loss per flow
        """
        preds = self.model(weights, *self.shifts, *inputs)[mask==1]
        return -np.sum(preds * y[mask==1])

    def accuracy(self, shifts, inputs, y, mask, n_nbrs):
        target_choice = np.argmax(y[mask==1], axis=1)
        preds = onp.array(self.model(self.weights, *shifts, *inputs))

        # make best choice out of each node's neighbors
        for i in range(len(preds)):
            preds[i, n_nbrs[i]:] = -100

        pred_choice = np.argmax(preds[mask==1], axis=1)
        return np.mean(pred_choice == target_choice)

    def two_target_accuracy(self, shifts, inputs, y, mask, n_nbrs):
        """
        Computes the ratio of the time the model correctly identifies which of the true target and a random target
            is correct.
        """
        if self.random_targets == None:
            self.random_targets = onp.random.randint(0, high=n_nbrs, size=inputs[0].shape[0])

        preds = onp.array(self.model(self.weights, *shifts, *inputs))

        # make best choice out of each node's neighbors
        for i in range(len(preds)):
            preds[i, n_nbrs[i]:] = -100

        pred_choice = np.argmax(preds[mask == 1], axis=1)

        for i in range(preds.shape[0]):
            while self.random_targets[i] == pred_choice[i]:
               self.random_targets[i] = onp.random.randint(0, high=n_nbrs[i])


        all_row_idxs = range(len(self.random_targets))
        random_probs = preds[all_row_idxs, self.random_targets]

        true_choice = onp.argmax(y, axis=1).reshape((y.shape[0],))
        true_probs = preds[all_row_idxs, true_choice]
        # print([r[0] for r in random_probs[:20]], [r[0][0] for r in target_probs[:20]])

        return onp.average(true_probs[mask==1] > random_probs[mask==1])

    def multi_hop_accuracy(self, shifts, inputs, y, mask, nbrhoods, E_lookup, last_nodes, n_nbrs, hops):
        """
        Returns the accuracy of the model in making multi-hop predictions
        """

        cur_inputs = list(inputs)
        cur_nodes = onp.array(last_nodes)
        for h in range(hops):
            preds = onp.array(self.model(self.weights, *shifts, *cur_inputs))
            # make best choice out of each node's neighbors
            for i in range(len(preds)):
                preds[i, n_nbrs[i]:] = -100

            pred_choice = onp.argmax(preds, axis=1)

            for i in range(len(pred_choice)):
                assert pred_choice[i][0] < n_nbrs[i], 'aa'

            if h == hops - 1:
                return np.average(pred_choice[mask==1] == onp.argmax(y[mask == 1], axis=1))

            cur_nbrhoods = onp.array(nbrhoods)[cur_nodes]
            next_nodes = []
            for Nv, c in zip(cur_nbrhoods, pred_choice):
                next_node = Nv[c[0]]
                next_nodes.append(next_node)

            # categorize new edges into +1 and -1 orientation
            next_edge_cols_pos, next_edge_cols_neg = [], []
            next_edge_rows_pos, next_edge_rows_neg = [], []
            for idx, (i, j) in enumerate(zip(cur_nodes, next_nodes)):
                if j is None:
                    print(idx, i, j)
                    raise Exception
                try:
                    next_edge_cols_pos.append(E_lookup[(i, j)])
                    next_edge_rows_pos.append(idx)
                except KeyError:
                    next_edge_cols_neg.append(E_lookup[(j, i)])
                    next_edge_rows_neg.append(idx)




            cur_inputs[1][next_edge_rows_pos, next_edge_cols_pos] = 1
            cur_inputs[1][next_edge_rows_neg, next_edge_cols_neg] = -1

            # index last node's neighborhood with pred_choice to get what node is next
            # add (last_node, new_node) to flow


    def generate_weights(self, in_channels, hidden_layers, out_channels):
        """
        :param in_channels: # of channels in model inputs
        :param hidden_layers: see :function train:
        :param out_channels: # of channels in model outputs
        """
        weight_shapes = []
        if len(hidden_layers) > 0:
            weight_shapes += [(in_channels, hidden_layers[0][1])] * hidden_layers[0][0]

            for i in range(len(hidden_layers) - 1):
                for _ in range(hidden_layers[i+1][0]):
                    weight_shapes += [(hidden_layers[i][1], hidden_layers[i+1][1])]

            weight_shapes += [(hidden_layers[-1][1], out_channels)]

            self.weights = []
            for s in weight_shapes:
                self.weights.append(0.01 * onp.random.randn(*s))

        else:
            self.weights = [(in_channels, out_channels)]

    def setup(self, model, hidden_layers, shifts, inputs, y, in_axes, train_mask):
        """
        Set up model for training / calling
        """
        n_train_samples = sum(train_mask)

        self.shifts = shifts

        # set up model for batching
        self.model = vmap(model, in_axes=in_axes)

        # generate weights

        in_channels, out_channels = inputs[-1].shape[-1], y.shape[-1]
        self.generate_weights(in_channels, hidden_layers, out_channels)

    def train(self, inputs, y, train_mask, test_mask, n_nbrs):
        """
        Trains a batched GCN model to predict y using the given X and shift operators.
        Model can have any number of shifts and inputs.


        :param model: NN function
        :param hidden_layers: list of tuples (# weight matrices, # of channels) for each hidden layer
        :param imputs: inputs to model; X matrix must be last
        :param y: desired outputs
        :param in_axes: axes of model inputs to batch over
        :param test_ratio: ratio of data used as test data
        :param train_mask: 1-D binary array
        :param hops: number of steps to take before returning prediction todo implement
        """

        X = inputs[-1]
        N = X.shape[0]
        n_train_samples = sum(train_mask)
        n_test_samples = N - n_train_samples
        n_batches = n_train_samples // self.batch_size

        batch_mask = ''

        @jit
        def gradient_step(weights, inputs, y):
            grads = grad(self.loss)(weights, inputs, y, batch_mask)

            for i in range(len(weights)):
                weights[i] -= self.step_size * grads[i]

            return weights



        init_fun, update_fun, get_params = adam(self.step_size)


        def adam_step(i, opt_state, inputs, y):
            g = grad(self.loss)(self.weights, inputs, y, batch_mask)
            return update_fun(i, g, opt_state)

        self.adam_state = init_fun(self.weights)
        unshuffled_batch_mask = onp.array([1] * self.batch_size + [0] * (N - self.batch_size))

        # train
        for i in range(self.epochs * n_batches):
            batch_mask = onp.array(unshuffled_batch_mask)
            onp.random.shuffle(batch_mask)

            batch_mask = onp.logical_and(batch_mask, train_mask)

            # self.weights = gradient_step(self.weights, batch_inputs, batch_y)
            self.adam_state = adam_step(i, self.adam_state, inputs, y)
            self.weights = get_params(self.adam_state)

            if i % n_batches == n_batches - 1:
                train_loss = self.loss(self.weights, inputs, y, train_mask) / n_train_samples
                train_acc = self.accuracy(self.shifts, inputs, y, train_mask, n_nbrs)
                test_loss = self.loss(self.weights, inputs, y, test_mask) / n_test_samples
                test_acc = self.accuracy(self.shifts, inputs, y, test_mask, n_nbrs)
                print('Epoch {} -- train loss: {:.6f} -- train acc {:.3f} -- test loss {:.6f} -- test acc {:.3f}'
                      .format(i // n_batches, train_loss, train_acc, test_loss, test_acc))


        print("Epochs: {}, learning rate: {}, batch size: {}, model: {}".format(
            self.epochs, self.step_size, self.batch_size, self.model.__name__)
        )
        return train_loss, train_acc, test_loss, test_acc

    def test(self, test_inputs, y, test_mask, n_nbrs):
        """
        Return the loss and accuracy for the given inputs
        """
        loss = self.loss(self.weights, test_inputs, y, test_mask) / sum(test_mask)
        acc = self.accuracy(self.shifts, test_inputs, y, test_mask, n_nbrs)

        if self.verbose:
            print("Test loss: {:.6f}, Test acc: {:.3f}".format(loss, acc))
        return loss, acc