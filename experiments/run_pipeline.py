"""Experimentos: pipeline principal [[72,12,6]]."""
import os
import sys
import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from qecbb.config import CONFIG, SAVE_BASE_PATH, DEVICE, SEED, P_TRAIN_RANGE, P_MEAS_TRAIN_RANGE
from qecbb.artifacts import save_artifact, save_pickle_drive, sanitize_filename
from qecbb.bbcodes import load_bb_code, get_code_name_info
from qecbb.graphs import build_check_adjacency
from qecbb.data import generate_bb_code_data, BBCodeGraphDataset
from qecbb.model import GAT_LIF_BBDecoder
from qecbb.training import train_gat_lif_bb, quick_train
from qecbb.baselines import run_bposd_baseline
from qecbb.evaluation import wilson_ci, evaluate_snn, eval_accuracy


def ensure_cache_or_generate(code_name="[72,12,6]" if False else "[[72,12,6]]"):
    # code name
    if code_name != CONFIG["CODE_NAME"]:
        # override
        pass
    raw_name = str(CONFIG["CODE_NAME"])
    safe_code_name = re.sub(r"[\[\]\s]+", "", raw_name).replace(",", "_")
    # dataset
    dataset_filename = f"train_data_{safe_code_name}_N{CONFIG['N_SAMPLES_TRAIN']}_R{CONFIG['ROUNDS']}.npz"
    dataset_filepath = os.path.join(SAVE_BASE_PATH, dataset_filename)
    info = get_code_name_info()
    Hz_np, Lz_np = info["Hz_np"], info["Lz_np"]
    edge_index_bb = build_check_adjacency(Hz_np.astype("uint8") if Hz_np.dtype != np.uint8 else Hz_np)
    # load or gen
    if os.path.exists(dataset_filepath) and not CONFIG["OVERWRITE_EXISTING_FILES"]:
        loaded = np.load(dataset_filepath)
        train_feats = loaded["train_feats"]
        train_labels = loaded["train_labels"]
        train_any_err = loaded["train_any_err"]
        print("Dataset cargado desde cache")
    else:
        train_feats, train_labels, train_any_err = generate_bb_code_data(
            n_samples=CONFIG["N_SAMPLES_TRAIN"], Hz=Hz_np, Lz=Lz_np,
            rounds=CONFIG["ROUNDS"], p_dist_params=P_TRAIN_RANGE, p_meas_dist_params=P_MEAS_TRAIN_RANGE,
            balance=True, seed=SEED)
        np.savez_compressed(dataset_filepath, train_feats=train_feats, train_labels=train_labels,
                            train_any_err=train_any_err, p_train_range=np.array(P_TRAIN_RANGE),
                            p_meas_train_range=np.array(P_MEAS_TRAIN_RANGE))
        print("Dataset generado")
    return Hz_np, Lz_np, edge_index_bb, train_feats, train_labels


def main():
    Hz_np, Lz_np, edge_index_bb, train_feats, train_labels = ensure_cache_or_generate()
    info = get_code_name_info()
    train_ds = BBCodeGraphDataset(train_feats, edge_index_bb, train_labels)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = GAT_LIF_BBDecoder(in_channels=CONFIG["ROUNDS"], k_logical=info["K_LOGICAL"], spiking=True).to(DEVICE)
    # train or resume (cache reuse)
    model, history, total_epochs = train_gat_lif_bb(model, train_loader, total_epochs=12, device=DEVICE)
    # save history plot if desired not required; just finish
    print("OK pipeline base")

if __name__ == "__main__":
    main()
