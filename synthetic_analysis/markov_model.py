"""
Code for using a Markov model for path prediction

train accs
0.70625
0.5125
test accs
0.605
0.39
Reversed test accs
0.28
0.06


"""
import numpy as np
import networkx as nx

class Markov_Model():
    def __init__(self, order):
        """
        :param order: number of prior states to consider when making a prediction
        """
        self.order = order
        self.weights = {}

    def n_hop_paths(self, G, n):
        """
        Returns all valid n-hop paths in graph G
        """
        if n == 0:
            return [[node] for node in G.nodes]
        else:
            sub_paths = self.n_hop_paths(G, n - 1)
            res = []
            for node in G.nodes:
                for c in sub_paths:
                    if G.has_edge(c[-1], node):
                        res.append(c + [node])
            return res

    def neighborhood(self, G, v):
        """
        Returns the neighborhood of node v in NetworkX graph G
        """
        return np.array(sorted(G[v]))

    def train(self, G, paths):
        """
        :param G: NetworkX graph
        :param paths: paths over G
        """

        for prefix in self.n_hop_paths(G, self.order - 1):
            self.weights[tuple(prefix)] = {n: 0 for n in self.neighborhood(G, prefix[-1])}

        for path in paths:
            if len(path) >= self.order + 1:
                for i in range(self.order - 1, len(path) - 1):
                    prefix = tuple(path[i-(self.order-1):i+1])
                    self.weights[prefix][path[i+1]] += 1
        for prefix, dist in self.weights.items():
            total_samples = sum(dist.values())
            for nbr in dist.keys():
                if total_samples != 0:
                    self.weights[prefix][nbr] /= total_samples

    def predict(self, prefix):
        """
        Predicts which node will be visited next, given that prefix was just visited
        """
        best_nbr, best_prob = None, -1
        others = []
        for nbr, prob in self.weights[tuple(prefix)].items():
            if prob > best_prob:
                best_nbr, best_prob = nbr, prob
                others = []
            elif prob == best_prob:
                others.append(nbr)

        if len(others) > 0:
            return np.random.choice(others + [best_nbr])
        else:
            return best_nbr

    def test(self, prefixes, target_nodes, hops):
        """
        Returns the model's accuracy over the given prefixes and targets
        """
        cur_prefixes = np.array([list(prefix) for prefix in prefixes], dtype='object')
        for h in range(hops):
            for i in range(len(prefixes)):
                # print(i)
                if len(prefixes[i]) >= self.order:
                    cur_prefixes[i].append(self.predict(cur_prefixes[i][-self.order:]))


        # print([p[-2:] for p in cur_prefixes])
        pred_nodes = np.array([p[-1] for p in cur_prefixes])
        return np.average(target_nodes == pred_nodes)



# G = nx.Graph()
# G.add_edges_from(((0, 1), (1, 2), (0, 2), (0, 3), (3, 1)))
# markov = Markov_Model(2)
# print(markov.n_hop_paths(G, 2))