"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)

code for building the network from this paper https://arxiv.org/pdf/2012.06010.pdf
    (Bunch 2012)

norm_L1 = D2 B1.T D1.inv + B2 D3 B2.T D2.inv
"""
import numpy as np
from numpy.linalg import inv, pinv
from synthetic_data_gen import load_dataset, incidence_matrices


def get_faces(G):
    """
    Returns a list of the faces in an undirected graph
    """
    edges = list(G.edges)
    faces = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            if e1[0] == e2[0]:
                shared = e1[0]
                e3 = (e1[1], e2[1])
            elif e1[1] == e2[0]:
                shared = e1[1]
                e3 = (e1[0], e2[1])
            elif e1[0] == e2[1]:
                shared = e1[0]
                e3 = (e1[1], e2[0])
            elif e1[1] == e2[1]:
                shared = e1[1]
                e3 = (e1[0], e2[0])
            else:  # edges don't connect
                continue

            if e3[0] in G[e3[1]]:  # if 3rd edge is in graph
                faces.append(tuple(sorted((shared, *e3))))
    return list(sorted(set(faces)))


def compute_D2(B):
    """
    Computes D2 = max(diag(dot(|B|, 1)), I)
    """
    B_rowsum = np.abs(B).sum(axis=1)

    D2 = np.diag(np.maximum(B_rowsum, 1))
    return D2

def compute_D5(B2):
    """
    Computes D5 = diag(dot(|B2|, 1))
    """
    B2_rowsum = np.abs(B2).sum(axis=1)

    D5 = np.diag(B2_rowsum)
    return D5

def compute_D1(B1, D2):
    """
    Computes D1 = 2 * max(diag(|B1|) .* D2
    """
    rowsum = (np.abs(B1) @ D2).sum(axis=1)
    D1 = 2 * np.diag(rowsum)

    return D1

def compute_bunch_matrices(B1, B2):
    """
    Computes normalized A0 and A1 matrices (up and down),
        and returns all matrices needed for Bunch model shift operators
    """
    # print(B1.shape, B2.shape)

    # D matrices
    D2_2 = compute_D2(B2)
    D2_1 = compute_D2(B1)
    D3_n = np.identity(B1.shape[1]) # (|E| x |E|)
    D1 = compute_D1(B1, D2_2)
    D3 = np.identity(B2.shape[1]) / 3 # (|F| x |F|)
    D4 = np.identity(B2.shape[1]) # (|F| x |F|)
    D5 = compute_D5(B2)

    # L matrices
    D1_pinv = pinv(D1)
    D5_pinv = pinv(D5)
    D2_2_inv = inv(D2_2)

    L0u = B1 @ D3_n @ B1.T @ inv(D2_1)
    L1u = D2_2 @ B1.T @ D1_pinv @ B1
    L1d = B2 @ D3 @ B2.T @ D2_2_inv
    L2d = D4 @ B2.T @ D5_pinv @ B2

    # A matrices
    D4_inv = inv(D4)

    A0u = D2_1 - (L0u @ D2_1)
    A1u = D2_2 - (L1u @ D2_2)
    A1d = D2_2_inv - (D2_2_inv @ L1d)
    A2d = D4_inv - (D4_inv @ L2d)

    # normalized A matries
    I_A0u = np.identity(A0u.shape[0])
    I_A1u = np.identity(A1u.shape[0])
    I_A1d = np.identity(A1d.shape[0])
    I_A2d = np.identity(A2d.shape[0])

    A0u_n = (A0u + I_A0u) @ inv(D2_1 + I_A0u)
    A1u_n = (A1u + I_A1u) @ inv(D2_2 + I_A1u)
    A1d_n = (D2_2 + I_A1d) @ (A1d + I_A1d)
    A2d_n = (D4 + I_A2d) @ (A2d + I_A2d)

    return (A0u_n, A1u_n, A1d_n, A2d_n), (D1_pinv, D2_2, D3, D4, D5_pinv)

def compute_shift_matrices(B1, B2):
    """
    Computes shift matrices for Bunch model
    """
    (A0u_n, A1u_n, A1d_n, A2d_n), (D1_pinv, D2_2, D3, D4, D5_pinv) = compute_bunch_matrices(B1, B2)

    # shift matrices: S_(prev level)(cur level)
    S_00 = A0u_n
    S_10 = D1_pinv @ B1

    S_01 = D2_2 @ B1.T @ D1_pinv
    S_11 = A1d_n + A1u_n
    S_21 = B2 @ D3

    S_12 = D4 @ B2.T @ D5_pinv
    S_22 = A2d_n

    return S_00, S_10, S_01, S_11, S_21, S_12, S_22


def compute_norm_L1(G):
    """
    Computes the normalized Laplacian matrix
    """
    edge_to_idx = {edge: i for i, edge in enumerate(G.edges)}

    B1, B2 = incidence_matrices(G, sorted(G.nodes), sorted(G.edges), get_faces(G), edge_to_idx)
    D2 = compute_D2(B2)
    D1 = compute_D1(B1, D2)

    D1_inv = np.linalg.pinv(D1)
    D2_inv = np.linalg.inv(D2)


    norm_L1 = (D2 @ B1.T @ D1_inv @ B1) + ((B2 / 3) @ B2.T @ D2 @ D2_inv)
    return norm_L1




if __name__ == '__main__':
    folder_suffix = 'schaub2'
    folder = 'trajectory_data_1hop_' + folder_suffix

    G = load_dataset(folder)[5]
    edge_to_idx = {edge: i for i, edge in enumerate(G.edges)}

    B1, B2 = incidence_matrices(G, sorted(G.nodes), sorted(G.edges), get_faces(G), edge_to_idx)


    compute_bunch_matrices(B1, B2)


    # norm_L1 = compute_norm_L1(G)