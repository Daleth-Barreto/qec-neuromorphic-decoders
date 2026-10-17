"""Utilidades de artefactos (carga/guardado) replicando reglas del notebook."""
import os
import re
import pickle
import torch

from .config import SAVE_BASE_PATH


def sanitize_filename(filename: str) -> str:
    name = filename.replace("[[", "").replace("]]", "").replace(",", "_")
    return re.sub(r'[<>:"|?*]', '_', name)


def check_for_existing_files(filename: str) -> bool:
    clean_name = sanitize_filename(filename)
    filepath = os.path.join(SAVE_BASE_PATH, clean_name)
    return os.path.exists(filepath)


def save_artifact(artifact_data, filename: str, overwrite: bool = True, artifact_type: str = "model_state_dict"):
    clean_name = sanitize_filename(filename)
    full_filepath = os.path.join(SAVE_BASE_PATH, clean_name)

    if check_for_existing_files(clean_name) and not overwrite:
        print(f"[{artifact_type}] Archivo '{clean_name}' ya existe. Saltando guardado.")
        return

    os.makedirs(os.path.dirname(full_filepath), exist_ok=True)

    try:
        if artifact_type in ("model_state_dict", "checkpoint"):
            torch.save(artifact_data, full_filepath)
        elif artifact_type == "plot":
            artifact_data.savefig(full_filepath, bbox_inches="tight", dpi=150)
        elif artifact_type == "dataframe":
            artifact_data.to_csv(full_filepath, index=False)
        elif artifact_type == "pickle" or artifact_type == "data":
            with open(full_filepath, 'wb') as f:
                pickle.dump(artifact_data, f)
        else:
            print(f"Tipo desconocido: {artifact_type}. No guardado '{clean_name}'.")
            return

        if hasattr(os, "sync"):
            os.sync()

        size_kb = os.path.getsize(full_filepath) / 1024.0
        print(f"[{artifact_type}] GUARDADO OK ({size_kb:.1f} KB) -> {full_filepath}")
    except Exception as e:
        print(f"Error al guardar '{clean_name}': {e}")


def save_pickle_drive(obj, filename: str):
    filepath = os.path.join(SAVE_BASE_PATH, filename)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Drive] Guardado exitosamente: {filepath}")