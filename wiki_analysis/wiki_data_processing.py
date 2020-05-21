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

def build_graph(articles_lookup, edges):
    """
    Returns a NetworkX graph representing the links between articles, along with sorted node and edge lists
    """
    G = networkx.DiGraph()
    for edge in edges:
        G.add_edge(articles_lookup[edge[0]], articles_lookup[edge[1]])

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
                    yield [i, j, k]
                    if found % 1000 == 0:
                        print(found)




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
I = 0
def path_to_flow(path, E_lookup, m):
    '''
    path: list of nodes
    E_lookup: dictionary mapping edge tuples to indices
    m: number of edges
    '''
    global I
    l = len(path)
    f = np.zeros([m,1])
    for j in range(l-1):
        v0 = path[j]
        v1 = path[j+1]
        f[E_lookup[(v0, v1)]] += 1 # todo if directed

        # if v0 < v1:               todo if undirected
        #     k = E_lookup[(v0,v1)]
        #     f[k] += 1
        # else:
        #     k = E_lookup[(v1,v0)]
        #     f[k] -= 1
    return f

def preprocess_data(folder_path):
    """
    Returns preprocessed Wikispeedia data.

    :dict flows_in: map (path_length -> flows matrix)
    :arrays B1, B2, Bconds: node->edge, edge->face, and node->nbr edges incidence matrices
    """
    # import data
    articles, articles_lookup, categories, edges, paths_finished, paths_unfinished = \
        import_all_files(folder_path)

    # build graph & shift matrices
    G, V, E = build_graph(articles_lookup, edges)
    G_undir = G.to_undirected()
    triangles = list(find_triangles_undir(G_undir))
    np.save('wiki_data/triangles.npy', triangles)
    raise Exception
    triangles = np.load('wiki_data/triangles.npy')
    # B1, B2 = todo


    # build flows
    paths_grouped = process_paths(paths_finished, articles_lookup)
    E_lookup = {E[i]: i for i in range(len(E))}

    flows_in = {}
    for path_length, (prefixes, suffixes) in sorted(paths_grouped.items()):
        print(path_length)
        flows = [path_to_flow(pref, E_lookup, len(E)) for pref in prefixes]
        flows_in[path_length] = np.array(flows)


    # targets = {} todo
    # return flows_in, [B1, B2, Bconds], targets


#     [(400, 1017), (1017, 660), (1000, 11, 1017)]

preprocess_data('wiki_data')