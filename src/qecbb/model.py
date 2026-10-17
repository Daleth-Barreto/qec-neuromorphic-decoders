"""Modelo GAT_LIF_BBDecoder (fiel a notebook)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool

from .config import CONFIG
from .lif import spike_fn, LIF_THRESHOLD


class GAT_LIF_BBDecoder(nn.Module):
    def __init__(self, in_channels, k_logical,
                 hidden_channels=CONFIG["HIDDEN_CHANNELS"],
                 readout_hidden=CONFIG["READOUT_HIDDEN"],
                 num_steps=CONFIG["SNN_NUM_STEPS"],
                 beta=CONFIG["SNN_BETA"],
                 num_layers=CONFIG["NUM_LAYERS"],
                 dropout=CONFIG["DROPOUT"],
                 heads=2,
                 spiking=True):
        super().__init__()
        self.beta = beta
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.k_logical = k_logical
        self.spiking = spiking
        self.hidden_channels = hidden_channels

        self.gat_inner_dim = hidden_channels // heads

        self.input_proj = nn.Linear(in_channels, hidden_channels, bias=False)
        self.pe_input = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.pe_readout = nn.Linear(hidden_channels, hidden_channels)

        self.conv1 = GATConv(hidden_channels, self.gat_inner_dim, heads=heads,
                             concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, self.gat_inner_dim,
                                       heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.readout_fc1 = nn.Linear(hidden_channels, readout_hidden)
        self.readout_bn = nn.BatchNorm1d(readout_hidden)
        self.readout_fc2 = nn.Linear(readout_hidden, k_logical)

    def _build_node_pe(self, total_nodes, batch_size, device):
        num_nodes = total_nodes // batch_size
        pos = torch.arange(num_nodes, device=device, dtype=torch.float32).unsqueeze(1) / max(num_nodes, 1)
        half_dim = self.hidden_channels // 2
        freqs = torch.arange(1, half_dim + 1, device=device, dtype=torch.float32).unsqueeze(0) * (2.0 * math.pi)
        angles = pos * freqs
        pe_single = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return pe_single.repeat(batch_size, 1)

    def forward(self, data, return_spike_stats=False):
        x = data.x
        edge_index = data.edge_index
        batch_idx = data.batch
        if batch_idx is None:
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        batch_size = int(batch_idx.max().item()) + 1

        pe = self._build_node_pe(x.size(0), batch_size, x.device)
        node_active = (x.abs().sum(dim=-1, keepdim=True) > 0).float()

        curr_static = self.input_proj(x) + self.pe_input(pe) * node_active
        readout_mod = 1.0 + torch.tanh(self.pe_readout(pe))

        if not self.spiking:
            x_step = F.relu(self.bn1(self.conv1(curr_static, edge_index)))
            for conv, bn in zip(self.layers, self.bns):
                identity = x_step
                x_step = F.relu(bn(conv(x_step, edge_index)) + identity)
            pooled = global_add_pool(x_step * readout_mod, batch_idx)
            r = F.relu(self.readout_bn(self.readout_fc1(pooled)))
            final_out = self.readout_fc2(r)
            if return_spike_stats:
                return final_out, torch.tensor(0.0)
            return final_out

        mems = [torch.zeros(x.size(0), curr_static.size(1), device=x.device)
                for _ in range(self.num_layers)]
        out_acc = torch.zeros(batch_size, self.k_logical, device=x.device)
        total_spk_count = 0.0

        for t in range(self.num_steps):
            c1 = self.bn1(self.conv1(curr_static, edge_index))
            mems[0] = self.beta * mems[0] + c1
            spk1 = spike_fn(mems[0] - LIF_THRESHOLD)
            mems[0] = mems[0] - spk1
            total_spk_count += spk1.sum()
            x_step = spk1

            for idx, (conv, bn) in enumerate(zip(self.layers, self.bns), start=1):
                identity = x_step
                out_layer = bn(conv(x_step, edge_index)) + identity
                mems[idx] = self.beta * mems[idx] + out_layer
                spk_next = spike_fn(mems[idx] - LIF_THRESHOLD)
                mems[idx] = mems[idx] - spk_next
                total_spk_count += spk_next.sum()
                x_step = spk_next

            pooled = global_add_pool(x_step * readout_mod, batch_idx)
            r = F.relu(self.readout_bn(self.readout_fc1(pooled)))
            logits = self.readout_fc2(r)
            out_acc += logits

        final_out = out_acc / float(self.num_steps)
        if return_spike_stats:
            total_neurons = x.size(0) * curr_static.size(1) * self.num_steps * self.num_layers
            mean_spk_rate = total_spk_count / max(total_neurons, 1)
            return final_out, mean_spk_rate
        return final_out
