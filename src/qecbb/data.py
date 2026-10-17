"""Dataset y generación de datos (ruido variable)."""
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset

from .config import CONFIG


class BBCodeGraphDataset(GeometricDataset):
    def __init__(self, node_features, edge_index, labels):
        super().__init__()
        self.node_features = node_features
        self.edge_index = edge_index
        self.labels = labels

    def len(self):
        return len(self.node_features)

    def get(self, idx):
        x = torch.tensor(self.node_features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        data = Data(x=x, edge_index=self.edge_index, y=y)
        return data


def generate_bb_code_data(n_samples, Hz, Lz, rounds, p_dist_params, p_meas_dist_params,
                           balance=True, seed=None, return_errors=False):
    rng = np.random.default_rng(seed)
    num_checks, num_qubits = Hz.shape
    K = Lz.shape[0]

    feats = np.zeros((n_samples, num_checks, rounds), dtype=np.float32)
    labels = np.zeros((n_samples, K), dtype=np.float32)
    errors = np.zeros((n_samples, num_qubits), dtype=np.uint8)

    i = 0
    attempts = 0
    max_attempts = n_samples * 40
    while i < n_samples and attempts < max_attempts:
        attempts += 1

        current_p = rng.uniform(p_dist_params[0], p_dist_params[1]) if isinstance(p_dist_params, tuple) else p_dist_params
        current_p_meas = rng.uniform(p_meas_dist_params[0], p_meas_dist_params[1]) if isinstance(p_meas_dist_params, tuple) else p_meas_dist_params

        cumulative_error = np.zeros(num_qubits, dtype=np.uint8)
        prev_syndrome = np.zeros(num_checks, dtype=np.uint8)
        sample_feats = np.zeros((num_checks, rounds), dtype=np.float32)

        for r in range(rounds):
            new_flips = (rng.random(num_qubits) < current_p).astype(np.uint8)
            cumulative_error = (cumulative_error ^ new_flips)
            true_syndrome = ((Hz @ cumulative_error) % 2).astype(np.uint8)

            if r == rounds - 1:
                meas_syndrome = true_syndrome
            else:
                meas_noise = (rng.random(num_checks) < current_p_meas).astype(np.uint8)
                meas_syndrome = (true_syndrome ^ meas_noise)

            detection_event = (meas_syndrome ^ prev_syndrome).astype(np.float32)
            sample_feats[:, r] = detection_event
            prev_syndrome = meas_syndrome

        logical_flip = (Lz @ cumulative_error) % 2
        is_logical_error = int(logical_flip.any())

        if balance:
            want_error = (i % 2 == 0)
            if bool(is_logical_error) != want_error:
                continue

        feats[i] = sample_feats
        labels[i] = logical_flip.astype(np.float32)
        errors[i] = cumulative_error
        i += 1

    if i < n_samples:
        feats = feats[:i]
        labels = labels[:i]
        errors = errors[:i]
        print(f"[aviso] solo se generaron {i}/{n_samples} muestras balanceadas")

    any_logical_error = (labels.sum(axis=1) > 0).astype(np.int64)
    if return_errors:
        return feats, labels, any_logical_error, errors
    return feats, labels, any_logical_error