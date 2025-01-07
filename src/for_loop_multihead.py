###############################################################################################
# Mutihead attention implemented in a for loop
# Based on https://youtu.be/kCc8FmEb1nY
###############################################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F

class AttHead(nn.Module):
    def __init__(self, emd_dim, head_size):
        super().__init__()
        self.wq = nn.Linear(emd_dim, head_size, bias=False)
        self.wk = nn.Linear(emd_dim, head_size, bias=False)
        self.wv = nn.Linear(emd_dim, head_size, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        _, _, E = x.shape # (B, S, E)

        q:torch.Tensor = self.wq(x) # x (B, S, E) @ wq (E, H) = q (B, S, H)
        k:torch.Tensor = self.wk(x) # x (B, S, E) @ wk (E, H) = k (B, S, H)
        v:torch.Tensor = self.wv(x) # x (B, S, E) @ wv (E, H) = v (B, S, H)

        att = q @ k.transpose(-2, -1) # Words similarity q (B, S, H) @ k (B, H, S) = x (B, S, S)
        att:torch.Tensor = att * E**-0.5 # scale
        att = F.softmax(att, dim=-1)  # Final attention weights (B, S, S)
        att = self.dropout(att)

        x = att @ v # Weighted average att (B, S, S) @ v (B, S, H) = x (B, S, H)

        return x

class MultiHeadAtt(nn.Module):
    def __init__(self, emb_dim, head_size, n_heads):
        super().__init__()

        self.heads = nn.ModuleList([AttHead(emb_dim, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim) # projection to the residual connetion
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # cat in the embeddind slash head dimention. if head size is E/2 and n_heads is 2, out will still be (B, S, E) 
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)