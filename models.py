import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=200, output_dim=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer with 200 hidden units
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Intermediate linear layer to transform the output of the LSTM
        self.fc_0 = nn.Linear(hidden_dim, 50) 
        
        # Output linear layer
        self.fc = nn.Linear(50, output_dim)  # This produces the final output

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass the input through the LSTM layer
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Pass the last time step's output through the first dense layer and then apply ReLU activation
        out = (self.fc_0(out[:, -1, :]))
        
        # Pass the output through the second dense layer to get the final prediction
        out = self.fc(out)
        return out
    
import numpy as np
import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
import torch.utils.checkpoint as cp

n_layer = 8
n_head = 8
n_embd = 512
batch_size = 32
block_size = 128
max_block_size = 512
vocab_size = 50304
grad_accum_steps = 4
n_features = 5

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.key_dim = n_embd // n_head
        self.scale = self.key_dim ** -0.5
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.key_dim * self.n_head, dim=2)
        q = q.view(B, T, self.n_head, self.key_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.key_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.key_dim).transpose(1, 2)
        weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.triu(weights.fill_(float('-inf')), diagonal=1)
        weights = F.softmax(weights, dim=-1)
        attn_output = torch.matmul(weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(attn_output)
        return y

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_heads)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        def attn_block(x):
            x = x + self.attn(F.layer_norm(x, (x.size()[-1],)))
            return x

        def mlp_block(x):
            x = x + self.mlp(F.layer_norm(x, (x.size()[-1],)))
            return x

        x = cp.checkpoint(attn_block, x, use_reentrant=False)
        x = cp.checkpoint(mlp_block, x, use_reentrant=False)
        return x
    

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, n_embd,n_heads, max_block_size, n_layer, device):
        super().__init__()
        self.transformer = nn.ModuleDict({
            'wte': nn.Linear(n_features, n_embd),  # Token embedding layer
            'wpe': nn.Embedding(max_block_size, n_embd),  # Positional embedding layer
            'h': nn.ModuleList([Block(n_embd,n_heads) for _ in range(n_layer)])  # Transformer blocks
        })
        self.lm_head = nn.Linear(n_embd, 1, bias=False)  # Output linear layer to predict one output
        self.device = device

    def forward(self, x):
        # Embedding input features
        x = self.transformer['wte'](x)  # Shape: (Batch, Seq_len, n_embd)
        pos = torch.arange(x.shape[-2], dtype=torch.long).unsqueeze(0).to(device=self.device)
        # Embedding positions
        position_embeddings = self.transformer['wpe'](pos)  # Shape: (Batch, Seq_len, n_embd)
        
        # Combine embeddings by element-wise addition
        x = x + position_embeddings
        
        # Pass through each transformer block
        for block in self.transformer['h']:
            x = block(x)
        
        # Apply the final linear layer to each element in the sequence
        x = self.lm_head(x)  # Shape: (Batch, Seq_len, 1)
        
        # Optionally, you might want to reshape or aggregate the output here, depending on the task
        # For example, if you want the output as a single value per sequence:
        x = x.mean(dim=1)  # Average or sum over the sequence length
        
        return x
