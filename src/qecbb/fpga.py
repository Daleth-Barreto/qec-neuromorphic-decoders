"""FPGA export/síntesis (redirige salidas a bb_code_decoder_results)."""
import os, json, shutil
import numpy as np
import torch
from amaranth import Elaboratable, Module, Signal, signed
from amaranth.back import verilog

from .config import CONFIG, SAVE_BASE_PATH
from .lif import LIF_THRESHOLD
from .bbcodes import get_code_name_info
from .graphs import get_adjacency

def export_fpga_q88(model, out_subdir="fpga_data_bb_real"):
    info = get_code_name_info()
    Hz_np = info['Hz_np']
    adj, ei, maxd = get_adjacency(Hz_np)
    N_CHECKS_Z = info['N_CHECKS_Z']
    N_QUBITS = info['N']
    K_LOGICAL = info['K_LOGICAL']

    FPGA_DIR = os.path.join(SAVE_BASE_PATH, out_subdir)
    if os.path.exists(FPGA_DIR):
        shutil.rmtree(FPGA_DIR)
    os.makedirs(FPGA_DIR, exist_ok=True)

    DATA_W = 16; FRAC_W = 8; SCALE_FPGA = 1 << FRAC_W
    model.eval()
    state = model.state_dict()

    def quantize_q88(t):
        arr = t.detach().cpu().numpy().astype(np.float32)
        q = np.rint(arr * SCALE_FPGA)
        q = np.clip(q, -32768, 32767).astype(np.int16)
        return q
    def save_mem_q88(t, filename):
        q = quantize_q88(t); flat = q.reshape(-1)
        path = os.path.join(FPGA_DIR, filename)
        with open(path,'w') as f:
            for v in flat: f.write(f"{int(v)&0xFFFF:04x}\n")
        return {"file":filename,"shape":list(q.shape),"num_values":int(q.size),"min":int(q.min()) if q.size else 0,"max":int(q.max()) if q.size else 0}

    manifest = {"format":"Q8.8","data_width":DATA_W,"fractional_width":FRAC_W,"scale":SCALE_FPGA,"code":CONFIG["CODE_NAME"],
                "n_qubits":N_QUBITS,"n_checks_z":N_CHECKS_Z,"k_logical":K_LOGICAL,"rounds":CONFIG["ROUNDS"],
                "hidden_channels":CONFIG["HIDDEN_CHANNELS"],"readout_hidden":CONFIG["READOUT_HIDDEN"],
                "num_layers":CONFIG["NUM_LAYERS"],"num_steps":CONFIG["SNN_NUM_STEPS"],"beta":float(CONFIG["SNN_BETA"]),
                "lif_threshold":float(LIF_THRESHOLD),"parameters":{}}
    for name,t in state.items():
        if not torch.is_tensor(t): continue
        safe=name.replace('.','_').replace('/','_')
        manifest["parameters"][name]=save_mem_q88(t,f"{safe}.mem")
    edge_np = ei.detach().cpu().numpy(); edges=edge_np.T.astype(np.int16)
    with open(os.path.join(FPGA_DIR,"edge_index.mem"),'w') as f:
        for src,dst in edges: f.write(f"{int(src)&0xFFFF:04x}\n"); f.write(f"{int(dst)&0xFFFF:04x}\n")
    with open(os.path.join(FPGA_DIR,"check_adjacency.json"),'w') as f:
        json.dump({"num_nodes":int(N_CHECKS_Z),"num_edges":int(edge_np.shape[1]),"adjacency":adj}, f, indent=2)
    hw={"DATA_W":DATA_W,"FRAC_W":FRAC_W,"SCALE":SCALE_FPGA,"BETA_HW":int(round(CONFIG["SNN_BETA"]*SCALE_FPGA)),
        "THRESH_HW":int(round(LIF_THRESHOLD*SCALE_FPGA)),"N_CHECKS":int(N_CHECKS_Z),"MAX_DEGREE":max(len(x) for x in adj) if adj else 0,
        "HIDDEN":CONFIG["HIDDEN_CHANNELS"],"READOUT_HIDDEN":CONFIG["READOUT_HIDDEN"],"K_LOGICAL":K_LOGICAL,
        "NUM_LAYERS":CONFIG["NUM_LAYERS"],"NUM_STEPS":CONFIG["SNN_NUM_STEPS"],"ROUNDS":CONFIG["ROUNDS"],"HEADS":2}
    with open(os.path.join(FPGA_DIR,"hardware_config.json"),'w') as f: json.dump(hw,f,indent=2)
    with open(os.path.join(FPGA_DIR,"manifest.json"),'w') as f: json.dump(manifest,f,indent=2)
    return FPGA_DIR


