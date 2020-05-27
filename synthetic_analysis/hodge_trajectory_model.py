import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp

class Hodge_GCN():
    def __init__(self, epochs, step_size, batch_size, verbose=True):
        """
        :param epochs: # of training epochs
        :param step_size: step size for use in training model
        :param batch_size: # of data points to train over in each gradient step
        :param verbose: whether to print training progress
        """

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

    def accuracy(self, shifts, inputs, y, mask):
        target_choice = np.argmax(y[mask==1], axis=1)
        preds = self.model(self.weights, *shifts, *inputs)
        pred_choice = np.argmax(preds[mask==1], axis=1)
        return np.mean(pred_choice == target_choice)

    def multi_hop_accuracy(self, shifts, inputs, y, mask, nbrhoods, E_lookup, last_nodes, hops):
        """
        Returns the accuracy of the model in making multi-hop predictions
        """

        cur_inputs = list(inputs)
        cur_nodes = onp.array(last_nodes)
        cur_mask = onp.array(mask)
        for h in range(hops):
            preds = self.model(self.weights, *shifts, *cur_inputs)
            pred_choice = onp.argmax(preds[cur_mask == 1], axis=1)



            if h == hops - 1:
                return np.sum(pred_choice == onp.argmax(y[cur_mask == 1], axis=1)) / onp.sum(mask)
            cur_nbrhoods = onp.array(nbrhoods)[cur_nodes]

            next_nodes = []
            for Nv, c in zip(cur_nbrhoods, pred_choice):
                next_node = Nv[c[0]]
                next_nodes.append(next_node)

            # categorize new edges into +1 and -1 orientation
            next_edge_cols_pos, next_edge_cols_neg = [], []
            next_edge_rows_pos, next_edge_rows_neg = [], []
            for idx, (i, j) in enumerate(zip(cur_nodes, next_nodes)):
                if j == -1:
                    # Impossible prediction made; don't update its flow, and remove it from the mask for future hops
                    cur_mask[idx] = 0
                    continue
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

    def train(self, inputs, y, train_mask):
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

        @jit
        def gradient_step(weights, inputs, y):
            grads = grad(self.loss)(weights, inputs, y, train_mask)

            for i in range(len(weights)):
                weights[i] -= self.step_size * grads[i]

            return weights



        # train
        other_choice = onp.random.randint(0, high=y.shape[1], size=n_train_samples)
        for i in range(self.epochs * 10 * N // self.batch_size):
            batch_indices = onp.random.choice(N, self.batch_size, replace=False)
            batch_inputs = [inp[batch_indices] for inp in inputs]
            batch_y = y[batch_indices]
            self.weights = gradient_step(self.weights, batch_inputs, batch_y)

            if self.verbose and i % (N // self.batch_size) == 0:
                cur_loss = self.loss(self.weights, inputs, y, train_mask) / n_train_samples
                cur_acc = self.accuracy(self.shifts, inputs, y, train_mask)
                print('Epoch {} -- loss: {:.6f} -- acc {:.3f}'.format(i // 100, cur_loss, cur_acc))

        if self.verbose:
            print("Epochs: {}, learning rate: {}, batch size: {}, model: {}".format(
                self.epochs, self.step_size, self.batch_size, self.model.__name__)
            )
            print("Training loss: {:.6f}, training acc: {:.3f}".format(cur_loss, cur_acc))
        return self.loss(self.weights, inputs, y, train_mask), self.accuracy(self.shifts, inputs, y, train_mask)

    def test(self, test_inputs, y, test_mask):
        """
        Return the loss and accuracy for the given inputs
        """
        loss = self.loss(self.weights, test_inputs, y, test_mask) / sum(test_mask)
        acc = self.accuracy(self.shifts, test_inputs, y, test_mask)

        if self.verbose:
            print("Test loss: {:.6f}, Test acc: {:.3f}".format(loss, acc))
        return loss, acc