"""Configuración central (fiel al notebook 2(3)).
Unifica rutas locales, dispositivo, CONFIG completo, LIF_THRESHOLD y constante GPU.
"""
import os
import torch
import numpy as np

# Reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Rutas
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAVE_BASE_PATH = os.path.join(PROJECT_ROOT, "bb_code_decoder_results")
try:
    os.makedirs(SAVE_BASE_PATH, exist_ok=True)
except OSError:
    # Entornos read-only (p. ej. datasets montados bajo /kaggle/input):
    # redirigir las escrituras al directorio de trabajo (writable).
    PROJECT_ROOT = os.getcwd()
    SAVE_BASE_PATH = os.path.join(PROJECT_ROOT, "bb_code_decoder_results")
    os.makedirs(SAVE_BASE_PATH, exist_ok=True)

# Dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Para pruebas CPU, usar device='cpu' explícito
LIF_THRESHOLD = 1.0

CONFIG = {
    "SEED": 42,
    "BB_CODES": {
        "[[72,12,6]]":   {"l": 6,  "m": 6,  "d": 6,
                           "A": [("x", 3), ("y", 1), ("y", 2)],
                           "B": [("y", 3), ("x", 1), ("x", 2)]},
        "[[144,12,12]]": {"l": 12, "m": 6,  "d": 12,
                           "A": [("x", 3), ("y", 1), ("y", 2)],
                           "B": [("y", 3), ("x", 1), ("x", 2)]},
        "[[216,12,12]]": {"l": 18, "m": 6,  "d": 12,
                           "A": [("x", 3), ("y", 1), ("y", 2)],
                           "B": [("y", 3), ("x", 1), ("x", 2)]},
        "[[288,12,18]]": {"l": 12, "m": 12, "d": 18,
                           "A": [("x", 3), ("y", 7), ("y", 2)],
                           "B": [("y", 3), ("x", 1), ("x", 2)]},
        "[[360,12,24]]": {"l": 30, "m": 6,  "d": 24,
                           "A": [("x", 9), ("y", 1), ("y", 2)],
                           "B": [("y", 3), ("x", 25), ("x", 26)]},
        "[[90,8,10]]":   {"l": 15, "m": 3,  "d": 10,
                           "A": [("x", 9), ("y", 1), ("y", 2)],
                           "B": [("__id", 0), ("x", 2), ("x", 7)]},
    },
    "BB_FAMILY_SWEEP": ["[[72,12,6]]", "[[144,12,12]]", "[[216,12,12]]",
                        "[[288,12,18]]", "[[360,12,24]]"],
    "CODE_NAME": "[[72,12,6]]",
    "OVERWRITE_EXISTING_FILES": False,
    "SNN_NUM_STEPS": 30,
    "SNN_BETA": 0.90,
    "HIDDEN_CHANNELS": 64,
    "READOUT_HIDDEN": 128,
    "NUM_LAYERS": 4,
    "DROPOUT": 0.1,
    "N_SAMPLES_TRAIN": 100000,
    "N_SAMPLES_EVAL": 3500,
    "ROUNDS": 8,
}

P_TRAIN_RANGE = (0.002, 0.004)
P_MEAS_TRAIN_RANGE = (0.002, 0.004)
