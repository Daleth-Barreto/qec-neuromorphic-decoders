"""Grafo de checks (syndrome graph) y adjacency_list (fix bug FPGA)."""
import numpy as np
import torch


def build_check_adjacency(H):
    num_checks, num_qubits = H.shape
    edge_set = set()
    for q in range(num_qubits):
        checks_sharing_q = np.nonzero(H[:, q])[0]
        for a in range(len(checks_sharing_q)):
            for b in range(a + 1, len(checks_sharing_q)):
                i, j = int(checks_sharing_q[a]), int(checks_sharing_q[b])
                edge_set.add((min(i, j), max(i, j)))

    edge_list = []
    for (i, j) in edge_set:
        edge_list.append((i, j))
        edge_list.append((j, i))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def build_adjacency_list(H):
    num_checks, _ = H.shape
    edge_index = build_check_adjacency(H)
    ei = edge_index.detach().cpu().numpy()
    adj = [[] for _ in range(num_checks)]
    for src, dst in zip(ei[0], ei[1]):
        adj[int(src)].append(int(dst))
    return adj, edge_index

def get_adjacency(H):
    adj, ei = build_adjacency_list(H)
    return adj, ei, max(len(x) for x in adj) if adj else 0