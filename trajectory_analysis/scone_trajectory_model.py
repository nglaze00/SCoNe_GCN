import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap
from jax.experimental.optimizers import adam
from treelib import Tree
import matplotlib.pyplot as plt

onp.random.seed(1030)

class Scone_GCN():
    def __init__(self, epochs, step_size, batch_size, weight_decay, verbose=True):
        """
        :param epochs: # of training epochs
        :param step_size: step size for use in training model
        :param batch_size: # of data points to train over in each gradient step
        :param verbose: whether to print training progress
        :param weight_decay: ridge regularization constant
        """

        self.random_targets = None

        self.trained = False
        self.model = None
        self.model_single = None
        self.shifts = None
        self.weights = None

        self.epochs = int(epochs)
        self.step_size = step_size
        self.batch_size = int(batch_size)
        self.weight_decay = weight_decay

        self.verbose = verbose

    def loss(self, weights, inputs, y, mask):
        """
        Computes cross-entropy loss per flow
        """
        preds = self.model(weights, *self.shifts, *inputs)[mask==1]

        # cross entropy + ridge regularization
        return -np.sum(preds * y[mask==1]) / np.sum(mask) + (self.weight_decay * (np.linalg.norm(weights[:3])**2 + np.linalg.norm(weights[3:9])**2 + np.linalg.norm(weights[9])**2))

    def accuracy(self, shifts, inputs, y, mask, n_nbrs):
        """
        Computes ratio of correct predictions
        """
        target_choice = np.argmax(y[mask==1], axis=1)
        preds = onp.array(self.model(self.weights, *shifts, *inputs))

        # make best choice out of each node's neighbors
        for i in range(len(preds)):
            preds[i, n_nbrs[i]:] = -100

        pred_choice = np.argmax(preds[mask==1], axis=1)
        return np.mean(pred_choice == target_choice)

    def two_target_accuracy(self, shifts, inputs, y, mask, n_nbrs):
        """
        Computes the ratio of the time the model correctly identifies which of the true target and a random, different
            target is correct.
        """
        if type(self.random_targets) != onp.ndarray:
            self.random_targets = onp.random.randint(0, high=n_nbrs, size=inputs[1].shape[0])

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

        correct = 0
        true_masked, random_masked = true_probs[mask==1], random_probs[mask==1]
        for t, r in zip(true_masked, random_masked):
            if t > r:
                correct += 1
            elif t == r:
                correct += 0.5
        return correct / sum(mask)

    def multi_hop_accuracy_binary(self, shifts, inputs, y, mask, nbrhoods, E_lookup, last_nodes, n_nbrs, hops):
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




            cur_inputs[-1][next_edge_rows_pos, next_edge_cols_pos] = 1
            cur_inputs[-1][next_edge_rows_neg, next_edge_cols_neg] = -1

    def multi_hop_accuracy_dist(self, shifts, inputs, target_nodes, masks, nbrhoods, E_lookup, last_nodes, prefixes, hops):
        """
        Returns accuracy of the model in making multi-hop predictions, using distributions at each intermediate hop
            instead of binary choices
        """
        nbrhoods_unpadded = [nbrhood[onp.where(nbrhood != -1)] for nbrhood in nbrhoods]
        path_trees = [Tree() for _ in range(inputs[-1].shape[0])]
        # initialize leaves
        for i in range(len(path_trees)):
            path_trees[i].create_node(tag=last_nodes[i], identifier=str(last_nodes[i]), data=[inputs[-1][i], 1])

        # build trees
        for h in range(hops):
            for i in range(len(path_trees)):
                for leaf in path_trees[i].leaves():
                    flow = leaf.data[0]

                    probs = onp.array(onp.exp(self.model_single(self.weights, *shifts, inputs[0], leaf.tag, leaf.data[0])))


                    nbrs = onp.array(nbrhoods_unpadded[leaf.tag])
                    for j in range(len(nbrs)):
                        new_edge = (int(leaf.tag), nbrs[j])
                        new_flow = onp.array(flow)
                        if new_edge[0] < new_edge[1]:
                            flow_val = 1

                        else:
                            flow_val = -1
                        new_flow[E_lookup[tuple(sorted(new_edge))]] = flow_val




                        prob_so_far = leaf.data[1]
                        path_trees[i].create_node(tag=nbrs[j], identifier=leaf.identifier + str(nbrs[j]),
                                                  data=[new_flow, prob_so_far * probs[j]], parent=leaf.identifier)


        # find prob that target node is reached for each flow
        target_probs = onp.zeros(len(path_trees))
        for i in range(len(path_trees)):
            target_prob = 0
            valid_paths = 0

            for leaf in path_trees[i].leaves():
                if leaf.tag == target_nodes[i]:
                    valid_paths += 1
                    target_prob += leaf.data[1]
            target_prob /= valid_paths
            target_probs[i] = target_prob

        return [onp.average(target_probs[mask == 1]) for mask in masks]



        # get all paths from last_node to target_node (using nx.all_simple_paths, cutoff 2)
        # compute + memoize the target vector at each node in each path; build a dict of probability that path is followed
        #   as each path is stepped through
        # return average of that dict's values

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
        self.model_single = model

        # generate weights

        in_channels, out_channels = inputs[-1].shape[-1], y.shape[-1]
        self.generate_weights(in_channels, hidden_layers, out_channels)

    def train(self, inputs, y, train_mask, test_mask, n_nbrs):
        """
        Trains a batched SCoNe model to predict y using the given X and shift operators.
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
        orig_upper_weights = [self.weights[i*3 + 2] for i in range(3)]

        X = inputs[-1]
        N = X.shape[0]
        n_train_samples = sum(train_mask)
        n_test_samples = sum(test_mask)
        n_batches = n_train_samples // self.batch_size

        batch_mask = ''


        @jit
        def gradient_step(weights, inputs, y):
            grads = grad(self.loss)(weights, inputs, y, batch_mask)

            for i in range(len(weights)):
                weights[i] -= self.step_size * grads[i]

            return weights



        init_fun, update_fun, get_params = adam(self.step_size)

        # track gradients
        non_faces_all, non_faces = [], []
        faces_all, faces = [], []

        def adam_step(i, opt_state, inputs, y):
            g = grad(self.loss)(self.weights, inputs, y, batch_mask)
            non_faces.append(onp.mean([onp.mean(onp.abs(g[i*3])) for i in range(3)] + [onp.mean(onp.abs(g[i*3 + 1])) for i in range(3)]))
            faces.append(onp.mean([onp.mean(onp.abs(g[i*3 + 2])) for i in range(3)]))
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
                train_loss = self.loss(self.weights, inputs, y, train_mask)
                train_acc = self.accuracy(self.shifts, inputs, y, train_mask, n_nbrs)
                test_loss = self.loss(self.weights, inputs, y, test_mask)
                test_acc = self.accuracy(self.shifts, inputs, y, test_mask, n_nbrs)
                print('Epoch {} -- train loss: {:.6f} -- train acc {:.3f} -- test loss {:.6f} -- test acc {:.3f}'
                      .format(i // n_batches, train_loss, train_acc, test_loss, test_acc))

                non_faces_all.append(onp.mean(non_faces))
                faces_all.append(onp.mean(faces))

        print("Epochs: {}, learning rate: {}, batch size: {}, model: {}".format(
            self.epochs, self.step_size, self.batch_size, self.model.__name__)
        )

        # Save gradient magnitudes to file
        plt.plot(non_faces_all)
        plt.plot(faces_all)
        plt.legend(['non-face gradients', 'face gradients'])
        plt.savefig('grads/grads_over_training.png')
        np.save('grads/non_faces_grads_tanh.npy', non_faces_all)
        np.save('grads/faces_grads_tanh.npy', faces_all)


        return train_loss, train_acc, test_loss, test_acc

    def test(self, test_inputs, y, test_mask, n_nbrs):
        """
        Return the loss and accuracy for the given inputs
        """
        loss = self.loss(self.weights, test_inputs, y, test_mask)
        acc = self.accuracy(self.shifts, test_inputs, y, test_mask, n_nbrs)

        if self.verbose:
            print("Test loss: {:.6f}, Test acc: {:.3f}".format(loss, acc))
        return loss, acc