"""
Code for processing Wikispeedia dataset for path inferencing using flows

# todo questions:
    Exclude backtracking paths?
        -If yes -> use directed graph & flows all +1?
        -If no -> use undirected graph & flows +-1 (i don't like this as much since backtracking cancels to 0 -> seems weird in data)
    How to order edges / faces? is this necessary?
    1/4 of the paths aren't long enough to have a suffix of length n - 4; these can be ignored, right?
    For experiments predicting destination more than one hop away, how will Bconds work? will max degree be # of nodes within that # of hops?
        -Max degree in G is 1845, average is 52


# answers:
    -Include backtracking, and make graph undirected
    -Order same as in synthetic example
    -Ignore short paths
    -Worry about multi hop prediction later - just do one-hop prediction multiple times

    todo rerun flows with int8
"""
import csv
from collections import defaultdict
from urllib.parse import unquote
import networkx, numpy as np



# print(unquote("%C3%85land	Crimean_War"))


def import_from_file(path, delimiter=' ', lookup=False):
    """
    :param path: file path to articles file

    Returns:
        articles (list)
        articles_lookup (map, article name -> index)
    """
    rows = []
    if lookup:
        rows_lookup = {}
    for line in csv.reader(open(path), delimiter=delimiter):
        if len(line) > 0:
            if line[0][0] != "#":
                unquoted = tuple(unquote(l) for l in line)
                if delimiter == ' ':
                    unquoted = unquoted[0]
                rows.append(unquoted)
                if lookup:
                    rows_lookup[unquoted] = len(rows) - 1
    if lookup:
        return rows, rows_lookup
    else:
        return rows

def import_all_files(folder_path):
    """
    Imports all relevant Wikispeedia data files
    """

    (articles, articles_lookup), categories, edges, paths_finished, paths_unfinished = \
        import_from_file(folder_path + '/articles.tsv', lookup=True), \
        import_from_file(folder_path + '/categories.tsv', delimiter='\t'), \
        import_from_file(folder_path + '/links.tsv', delimiter='\t'), \
        import_from_file(folder_path + '/paths_finished.tsv', delimiter='\t'), \
        import_from_file(folder_path + '/paths_unfinished.tsv', delimiter='\t')

    paths_finished = [p[3].split(';') for p in paths_finished]
    paths_unfinished = [p[3].split(';') for p in paths_unfinished]
    return articles, articles_lookup, categories, edges, paths_finished, paths_unfinished

def build_graph(articles_lookup, edges, undirected=False):
    """
    Returns a NetworkX graph representing the links between articles, along with sorted node and edge lists
    """
    G = networkx.DiGraph()
    for edge in edges:
        G.add_edge(articles_lookup[edge[0]], articles_lookup[edge[1]])

    if undirected:
        G_undir = G.to_undirected()
        return G_undir, list(sorted(G_undir.nodes)), list(sorted(G_undir.edges))
    else:
        return G, list(sorted(G.nodes)), list(sorted(G.edges))

def find_triangles(G):
    """
    Return all triangles in a NetworkX graph G.

    Copied from networkx.simple_cycles(G)
    """

    def _unblock(thisnode, blocked, B):
        stack = {thisnode}
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()

    # Johnson's algorithm requires some ordering of the nodes.
    # We assign the arbitrary ordering given by the strongly connected comps
    # There is no need to track the ordering as each node removed as processed.
    # Also we save the actual graph so we can mutate it. We only take the
    # edges because we do not want to copy edge and node attributes here.
    subG = type(G)(G.edges())
    sccs = [scc for scc in networkx.strongly_connected_components(subG)
            if len(scc) > 1]

    # Johnson's algorithm exclude self cycle edges like (v, v)
    # To be backward compatible, we record those cycles in advance
    # and then remove from subG
    for v in subG:
        if subG.has_edge(v, v):
            # yield [v]
            subG.remove_edge(v, v)

    while sccs:
        scc = sccs.pop()
        sccG = subG.subgraph(scc)
        # order of scc determines ordering of nodes
        startnode = scc.pop()
        # Processing node runs "circuit" routine from recursive version
        path = [startnode]
        blocked = set()  # vertex: blocked from search?
        closed = set()  # nodes involved in a cycle
        blocked.add(startnode)
        B = defaultdict(set)  # graph portions that yield no elementary circuit
        stack = [(startnode, list(sccG[startnode]))]  # sccG gives comp nbrs
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()
                if nextnode == startnode and len(path) == 3:
                    yield path[:]
                    closed.update(path)
                #                        print "Found a cycle", path, closed
                elif nextnode not in blocked and len(path) < 3:
                    path.append(nextnode)
                    stack.append((nextnode, list(sccG[nextnode])))
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue
            # done with nextnode... look for more neighbors
            if not nbrs:  # no more nbrs
                if thisnode in closed:
                    _unblock(thisnode, blocked, B)
                else:
                    for nbr in sccG[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()
                #                assert path[-1] == thisnode
                path.pop()
        # done processing this node
        H = subG.subgraph(scc)  # make smaller to avoid work in SCC routine
        sccs.extend(scc for scc in networkx.strongly_connected_components(H)
                    if len(scc) > 1)

def find_triangles_undir(G_undir):
    """
    Finds all the triangles in an undirected graph G_undir
    """
    V, E = G_undir.nodes, G_undir.edges
    N = len(V)
    found = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if G_undir.has_edge(i, j) \
                    and G_undir.has_edge(j, k) \
                    and G_undir.has_edge(i, k):
                    found += 1
                    if found % 1000 == 0:
                        print(found)
    print('done {}'.format(found))

def build_L_upper(triangles, n_edges, E_lookup):
    """
    Builds the |E| x |E| upper Laplacian matrix

    :param triangles: list of all triangles in the graph
    :param n_edges: # of edges in the graph
    :param E_lookup: map (edge tuple -> index)
    """
    L_upper = np.zeros((n_edges, n_edges), dtype='int8')
    for i, (a,b,c) in enumerate(triangles):
        ab, ac, bc = E_lookup[(a,b)], E_lookup[(a,c)], E_lookup[(b,c)]
        # diagonal
        L_upper[ab, ab] += 1
        L_upper[bc, bc] += 1
        L_upper[ac, ac] += 1
        # off-diagonal
        L_upper[ab, bc] = 1
        L_upper[ab, ac] = -1
        L_upper[ac, bc] = -1
        i += 1
        if i % 1000 == 0:
            print(i)
    return L_upper

def build_L_lower(E):
    """
    Builds |E| x |E| lower Laplacian matrix

    :param V: nodes
    :param E: edges
    """
    L_lower = np.zeros((len(E), len(E)), dtype='int8')
    for i, (a, b) in enumerate(E):
        for j, (c, d) in enumerate(E):
            if i == j:
                L_lower[i, j] = 2       # 2 along diagonal
                continue
            if a == c or b == d:
                L_lower[i, j] = 1       # (a, b), (a, c)
            elif a == d or b == c:
                L_lower[i, j] = -1      # (a, b), (b, c)
            # else 0

    return L_lower


def process_paths(paths, articles_lookup, prefix_length=4):
    """
    Group paths by length, converts nodes to their indices, and divide them into prefixes and suffixes.

    Return a map (path length -> [prefixes, suffixes])
    """
    paths_grouped = defaultdict(lambda: ([], []))
    excluded_paths = 0
    for path in paths:
        path_idxs = []
        # Reformat backtracking '<' symbols to article names
        for i in range(len(path)):
            if path[i] == '<':
                path[i] = path[i-2]
                path_idxs.append(articles_lookup[path[i-2]])
            else:
                path_idxs.append(articles_lookup[path[i]])

        # Group path by length; divide into prefix and suffix
        l = len(path_idxs)
        if l <= prefix_length:
            excluded_paths += 1
        else:
            paths_grouped[l][0].append(path_idxs[:prefix_length]) # prefix
            paths_grouped[l][1].append(path_idxs[prefix_length:]) # suffix

    print("#, ratio of too-short paths: {}, {:.3f}".format(excluded_paths, excluded_paths / len(paths)))
    return paths_grouped

def path_to_flow(path, E_lookup, m):
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
            k = E_lookup[(v0,v1)]
            f[k] += 1
        else:
            k = E_lookup[(v1,v0)]
            f[k] -= 1
    return f


def build_prefix_flows_and_targets(file_prefix, paths, articles_lookup, G, E, E_lookup):
    """
    Converts all path prefixes to flow matrices and saves them to file.
    Also, extracts the target node following each prefix and one-hot encodes it among the last prefix node's neighbors
    """
    max_degree = max(g[1] for g in G.degree)


    paths_grouped = process_paths(paths, articles_lookup)


    for path_length, (prefixes, suffixes) in sorted(paths_grouped.items()):
        print(path_length)

        flows = []
        targets = []
        for pref, suff in zip(prefixes, suffixes):

            # flows
            try:
               flow = path_to_flow(pref, E_lookup, len(E))
            except KeyError:
               print('invalid path; skipping...')
               continue

            # target
            target = suff[0]
            if (pref[-1], suff[0]) not in E_lookup.keys():
                print('invalid path (missing edge); skipping...')
                continue

            nbrs = np.array(sorted(G[pref[-1]]))


            if target not in nbrs:
                print('invalid path (target not a neighbor); skipping...')
                continue

            onehot = (nbrs==target).astype(np.float)
            onehot_final = np.zeros(max_degree)
            onehot_final[:onehot.shape[0]] = onehot

            flows.append(flow)
            targets.append(np.array([onehot_final]).T)



        np.save('wiki_data/flow_data/flows_{}{}'.format(file_prefix, path_length), flows)
        np.save('wiki_data/target_data/targets_{}{}'.format(file_prefix, path_length), targets)

def preprocess_data(folder_path):
    """
    Returns preprocessed Wikispeedia data.

    :dict flows_in: map (path_length -> flows matrix) # todo saved on server
    :arrays B1, B2, Bconds: node->edge, edge->face, and node->nbr edges incidence matrices # todo big af
    :dict targets: map (path_length -> targets matrix) # todo code written, but not computed
    """
    # import data
    articles, articles_lookup, categories, edges, paths_finished, paths_unfinished = \
        import_all_files(folder_path)


    # build graph
    G, V, E = build_graph(articles_lookup, edges, undirected=True)

    E_lookup = {tuple(sorted(E[i])): i for i in range(len(E))}
    E_lookup.update({tuple(sorted(E[i]))[::-1]: i for i in range(len(E))})

    # find triangles
    # triangles = find_triangles_undir(G)
    # np.save('wiki_data/triangles.npy', triangles)
    # raise Exception
    triangles = np.load('wiki_data/triangles.npy')

    # build Bconds
    # B1, B2, Bconds = todo
    # np.save('wiki_data/B1', B1)
    # np.save('wiki_data/B1', B2)
    # np.save('wiki_data/B1', Bconds)

    # Build Laplacian matrices

    # L_lower = build_L_lower(V, E)
    # np.save('wiki_data/L_lower.npy', L_lower)

    # L_upper = build_L_upper(triangles, len(E), E_lookup)
    # np.save('wiki_data/L_upper.npy', L_upper)

    L_lower = np.load('wiki_data/L_lower.npy')
    L_upper = np.load('wiki_data/L_upper.npy')

    # build/save flow and target matrices
    build_prefix_flows_and_targets('finished', paths_finished, articles_lookup, G, E, E_lookup)
    build_prefix_flows_and_targets('unfinished', paths_unfinished, articles_lookup, G, E, E_lookup)




if __name__ == '__main__':
    preprocess_data('wiki_data')
