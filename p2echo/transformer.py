"""Prompt decoder: prompt tokens self-attend, then cross-attend to the bottleneck image tokens."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sine_position_encoding_2d(height, width, dim):
    """2D sinusoidal encoding, returned as [height * width, 1, dim]."""
    channels = int(math.ceil(dim / 4) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def embed(n):
        angles = torch.einsum("i,j->ij", torch.arange(n, dtype=torch.float32), inv_freq)
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2, -1)

    emb = torch.zeros(height, width, 2 * channels)
    emb[:, :, :channels] = embed(height).unsqueeze(1)
    emb[:, :, channels:] = embed(width)
    return emb[:, :, :dim].reshape(height * width, 1, dim)


class PromptDecoderLayer(nn.Module):
    """Post-norm layer: self-attention, cross-attention to image tokens, GELU FFN."""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout_sa = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, pos):
        tgt = self.norm1(tgt + self.dropout_sa(self.self_attn(tgt, tgt, value=tgt)[0]))
        attended = self.cross_attn(query=tgt, key=memory + pos, value=memory)[0]
        tgt = self.norm2(tgt + self.dropout1(attended))
        ffn = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout2(ffn))


class PromptDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, num_layers=3, dropout=0.1):
        super().__init__()
        layer = PromptDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, prompts, memory, pos):
        """prompts: [N, B, C], memory: [HW, B, C], pos: [HW, 1, C]."""
        for layer in self.layers:
            prompts = layer(prompts, memory, pos)
        return self.norm(prompts)
