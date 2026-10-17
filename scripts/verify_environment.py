# Verifica entorno y importaciones para qecbb
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
args = parser.parse_args()

print("Python:", sys.version)
try:
    import torch
    import torch_geometric
    import numpy, pandas, matplotlib, sklearn, networkx
    from ldpc import bposd_decoder
    from bposd.css import css_code
    print("Imports OK")
    if args.device=="auto": d="cuda" if torch.cuda.is_available() else "cpu"
    else: d=args.device
    torch_device = torch.device(d)
    print("Device:", torch_device, "| cuda:", torch.cuda.is_available())
    # smoke forward minimal
    # smoke forward minimal (resolve src/ relative to the repo root)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_repo_root, "src"))
    from qecbb.bbcodes import load_bb_code
    from qecbb.graphs import build_check_adjacency
    from qecbb.data import generate_bb_code_data
    from qecbb.model import GAT_LIF_BBDecoder
    from qecbb.config import CONFIG, DEVICE
    c=load_bb_code("[[72,12,6]]")
    ei=build_check_adjacency(c.hz.astype('uint8'))
    print("bbcode OK", c.N, ei.shape)
except Exception as e:
    print("ERR", e)
    raise

