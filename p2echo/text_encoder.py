"""Frozen Qwen3-Embedding text encoder with an in-memory cache of prompt embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

INSTRUCTION = ("Instruct: Given an anatomical term query, retrieve the precise anatomical entity "
               "and location it represents\nQuery: ")


class TextEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", max_length=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().requires_grad_(False)
        self.max_length = max_length
        self.dim = self.model.config.hidden_size
        self.cache = {}

    @torch.no_grad()
    def forward(self, prompts, device):
        """Returns [len(prompts), dim] L2-normalised last-token embeddings."""
        self.model.to(device)
        new = [p for p in prompts if p not in self.cache]
        if new:
            tokens = self.tokenizer([INSTRUCTION + p for p in new], padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(device)
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"]
            if mask[:, -1].sum() == mask.shape[0]:          # left padding
                pooled = hidden[:, -1]
            else:
                pooled = hidden[torch.arange(hidden.shape[0], device=device), mask.sum(dim=1) - 1]
            pooled = F.normalize(pooled.float(), p=2, dim=-1)
            for p, e in zip(new, pooled):
                self.cache[p] = e.cpu()
        return torch.stack([self.cache[p] for p in prompts]).to(device)
