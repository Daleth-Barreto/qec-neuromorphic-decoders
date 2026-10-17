import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from qecbb.config import CONFIG, SAVE_BASE_PATH
from qecbb.bbcodes import get_code_name_info
from qecbb.graphs import build_check_adjacency
from qecbb.data import BBCodeGraphDataset
from torch_geometric.loader import DataLoader
import numpy as np
info=get_code_name_info(); Hz_np,Lz_np=info['Hz_np'],info['Lz_np']; ei=build_check_adjacency(Hz_np)
safe=re.sub(r'[\[\]\s]+','',CONFIG['CODE_NAME']).replace(',','_')
fp=os.path.join(SAVE_BASE_PATH,f'train_data_{safe}_N{CONFIG["N_SAMPLES_TRAIN"]}_R{CONFIG["ROUNDS"]}.npz')
loaded=np.load(fp)
train_feats=loaded['train_feats'][:5000]; train_labels=loaded['train_labels'][:5000]
ds=BBCodeGraphDataset(train_feats,ei,train_labels); loader=DataLoader(ds,batch_size=64,shuffle=True)
from qecbb.model import GAT_LIF_BBDecoder; from qecbb.training import train_gat_lif_bb
model=GAT_LIF_BBDecoder(in_channels=CONFIG['ROUNDS'], k_logical=info['K_LOGICAL'], spiking=True)
model,hist,tot=train_gat_lif_bb(model,loader,total_epochs=2,device='cpu')
print('DONE', len(hist))
