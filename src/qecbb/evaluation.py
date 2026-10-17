"""EvaluaciÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³n, latencia, hÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â­brido, ROC y Wilson CI."""
import math
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from torch_geometric.loader import DataLoader

from .config import CONFIG, SAVE_BASE_PATH, DEVICE, SEED
from .artifacts import save_artifact, save_pickle_drive
from .data import generate_bb_code_data, BBCodeGraphDataset
from .bbcodes import load_bb_code
from .graphs import build_check_adjacency
from .baselines import run_bposd_baseline

def wilson_ci(k_fail, n_trials, z=1.96):
    if n_trials == 0:
        return 0.0, 0.0, 0.0
    p_hat = k_fail / n_trials
    denom = 1.0 + (z ** 2) / n_trials
    center = (p_hat + (z ** 2) / (2.0 * n_trials)) / denom
    half_width = (z * math.sqrt(p_hat * (1.0 - p_hat) / n_trials + (z ** 2) / (4.0 * (n_trials ** 2)))) / denom
    return p_hat, max(0.0, center - half_width), min(1.0, center + half_width)

@torch.no_grad()
def evaluate_snn(model, Hz, Lz, rounds, p, p_meas, n_samples, seed, edge_index, device=DEVICE):
    feats, labels, _ = generate_bb_code_data(
        Hz=Hz, Lz=Lz, rounds=rounds,
        p_dist_params=(p, p),
        p_meas_dist_params=(p_meas, p_meas),
        n_samples=n_samples, seed=seed, balance=False)
    ds = BBCodeGraphDataset(feats, edge_index, labels)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    total_block_errors = 0
    total_bit_errors = 0
    total_samples = 0
    model = model.to(device)
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        preds = (torch.sigmoid(logits) > 0.5).long()
        targets = batch.y.view(-1, model.k_logical).long()
        mismatch = preds != targets
        total_block_errors += mismatch.any(dim=1).sum().item()
        total_bit_errors += mismatch.float().mean(dim=1).sum().item()
        total_samples += targets.size(0)
    return total_block_errors / total_samples, total_bit_errors / total_samples

@torch.no_grad()
@torch.no_grad()
@torch.no_grad()
def eval_accuracy(model, loader, k_logical=None, device=DEVICE):
    model = model.to(device)
    if k_logical is None or not isinstance(k_logical, (int, float, np.integer, np.floating)):
        try:
            kl = getattr(model, 'k_logical', 12)
            k_logical = int(kl)
        except Exception:
            k_logical = 12
    k_logical = int(k_logical)
    correct_bits, total_bits = 0, 0
    correct_all, total_samples = 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        true = batch.y
        if true.dim() == 3: true = true.squeeze(1)
        if true.dim() == 2 and true.size(0) == 1 and logits.size(0) != 1: pass  # no
        true = true.view(-1, k_logical)
        correct_bits += (preds == true).sum().item()
        total_bits += true.numel()
        correct_all += ((preds == true).all(dim=1)).sum().item()
        total_samples += true.size(0)
    return correct_bits / total_bits, correct_all / total_samples, None
