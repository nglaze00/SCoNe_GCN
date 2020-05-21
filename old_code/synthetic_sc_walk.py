import numpy as np
import networkx as nx
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.cm as cm

n = 400

points = np.random.rand(n,2)
tri = Delaunay(points)

valid_idxs = np.where((np.linalg.norm(points - [1/4, 3/4], axis=1) > 1/8) \
                      & (np.linalg.norm(points - [3/4, 1/4], axis=1) > 1/8))[0]
points_valid = points[valid_idxs]
faces = [t for t in tri.simplices if np.in1d(t,valid_idxs).all()]

# SC matrix construction
G = nx.DiGraph()
G.add_nodes_from(np.arange(n)) # add nodes that are excluded to keep indexing easy

for f in faces:
    [a,b,c] = sorted(f)
    G.add_edge(a,b)
    G.add_edge(b,c)
    G.add_edge(a,c)
    
V = sorted(G.nodes)
E = sorted(G.edges)
E_lookup = dict(zip(E,range(len(E))))

B1 = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense())
B2 = np.zeros([len(E),len(faces)])

for e_idx, e in enumerate(E):
    [e0,e1] = sorted(e)
    for f_idx, f in enumerate(faces):
        if np.in1d(e,f).all():
            [a,b,c] = sorted(f)
            if e0==a:
                if e1==b:
                    B2[e_idx, f_idx] = 1
                else:
                    B2[e_idx, f_idx] = -1
            else:
                B2[e_idx, f_idx] = 1

# "partition" node space
BEGIN = valid_idxs[np.sum(points_valid, axis=1) < 1/4]
END = valid_idxs[np.sum(points_valid, axis=1) > 7/4]

A012 = valid_idxs[(np.sum(points_valid, axis=1) > 1/4) & (np.sum(points_valid, axis=1) < 1)]
A0 = A012[(points[A012,1]-points[A012,0] < 1/2) & (points[A012,1]-points[A012,0] > -1/2)]
A1 = A012[points[A012,1]-points[A012,0] > 1/2]
A2 = A012[points[A012,1]-points[A012,0] < -1/2]

B012 = valid_idxs[(np.sum(points_valid, axis=1) < 7/4) & (np.sum(points_valid, axis=1) > 1)]
B0 = B012[(points[B012,1]-points[B012,0] < 1/2) & (points[B012,1]-points[B012,0] > -1/2)]
B1_ = B012[points[B012,1]-points[B012,0] > 1/2]
B2_ = B012[points[B012,1]-points[B012,0] < -1/2]

# trajectories will look like one of the following:
# BEGIN -> A0 -> B0 -> END
# BEGIN -> A1 -> B1 -> END
# BEGIN -> A2 -> B2 -> END

paths = []
G_undir = G.to_undirected()

m = 1000

for i in range(m):
    v_begin = np.random.choice(BEGIN)
    if i%3==0:
        v_1 = np.random.choice(A0)
        v_2 = np.random.choice(B0)
    elif i%3==1:
        v_1 = np.random.choice(A1)
        v_2 = np.random.choice(B1_)
    else:
        v_1 = np.random.choice(A2)
        v_2 = np.random.choice(B2_)
    v_end = np.random.choice(END)

    path = nx.shortest_path(G_undir, v_begin, v_1)[:-1] + \
        nx.shortest_path(G_undir, v_1, v_2)[:-1] + \
        nx.shortest_path(G_undir, v_2, v_end)

    paths.append(path)

flows = np.zeros([len(E),m])

for i,path in enumerate(paths):
    l = len(path)
    for j in range(l-1):
        v0 = path[j]
        v1 = path[j+1]
        if v0 < v1:
            k = E_lookup[(v0,v1)]
            flows[k,i] += 1
        else:
            k = E_lookup[(v1,v0)]
            flows[k,i] -= 1
