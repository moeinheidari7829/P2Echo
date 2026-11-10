"""Dynamic Image-Text Alignment (DyITA) and the ITA decoder block."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


def ste_argmax(scores):
    """Hard one-hot routing in the forward pass, softmax gradient in the backward pass."""
    one_hot = F.one_hot(scores.argmax(dim=-1), scores.shape[-1]).float()
    soft = F.softmax(scores, dim=-1)
    return one_hot + soft - soft.detach()


class DynamicProjection(nn.Module):
    """Q from image tokens, K/V from prompt tokens, plus routed redundancy projections Q', K'."""

    def __init__(self, img_dim, txt_dim, head_dim, n_heads, n_projectors=3):
        super().__init__()
        inner = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(img_dim, inner, bias=False)
        self.k_proj = nn.Linear(txt_dim, inner, bias=False)
        self.v_proj = nn.Linear(txt_dim, inner, bias=False)
        self.q_prime_banks = nn.ModuleList(
            [nn.Linear(img_dim, inner, bias=False) for _ in range(n_projectors)])
        self.k_prime_banks = nn.ModuleList(
            [nn.Linear(txt_dim, inner, bias=False) for _ in range(n_projectors)])
        self.router_q = nn.Linear(img_dim, n_projectors, bias=False)
        self.router_k = nn.Linear(txt_dim, n_projectors, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _route(tokens, router, banks):
        route = ste_argmax(router(tokens))                                # [B, L, P]
        candidates = torch.stack([bank(tokens) for bank in banks], dim=-1)  # [B, L, C, P]
        return (candidates * route.unsqueeze(2)).sum(dim=-1)

    def _heads(self, x):
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, img_tokens, txt_tokens):
        q = self.q_proj(img_tokens)
        k = self.k_proj(txt_tokens)
        v = self.v_proj(txt_tokens)
        q_prime = self._route(img_tokens, self.router_q, self.q_prime_banks)
        k_prime = self._route(txt_tokens, self.router_k, self.k_prime_banks)
        return tuple(self._heads(t) for t in (q, k, v, q_prime, k_prime))


class DynamicMeasureKernel(nn.Module):
    """Norm-preserving power map softplus(z)^gamma with a per-token routed exponent."""

    def __init__(self, dim, n_factors=9, gamma_init=3.0):
        super().__init__()
        self.gammas = nn.Parameter(torch.full((n_factors,), float(gamma_init)))
        self.router = nn.Linear(dim, n_factors, bias=False)
        nn.init.xavier_uniform_(self.router.weight)

    def forward(self, z):
        route = ste_argmax(self.router(z.detach()))
        gamma = (route * self.gammas.unsqueeze(0)).sum(dim=-1, keepdim=True).clamp(min=1.0)
        z = F.softplus(z)
        norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z_pow = z.pow(gamma)
        return z_pow / z_pow.norm(dim=-1, keepdim=True).clamp(min=1e-8) * norm


class DifferentialOperator(nn.Module):
    """Subtract routed redundancy (Q - l_q Q', K - l_k K') and apply softmax cross-attention."""

    def __init__(self, head_dim, n_factors=9, lambda_init=0.01):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.lambdas = nn.Parameter(torch.full((n_factors,), float(lambda_init)))
        self.router_q = nn.Linear(2 * head_dim, n_factors, bias=False)
        self.router_k = nn.Linear(2 * head_dim, n_factors, bias=False)
        nn.init.xavier_uniform_(self.router_q.weight)
        nn.init.xavier_uniform_(self.router_k.weight)

    def forward(self, q, q_prime, k, k_prime, v):
        route_q = ste_argmax(self.router_q(torch.cat([q, q_prime], dim=-1).detach()))
        route_k = ste_argmax(self.router_k(torch.cat([k, k_prime], dim=-1).detach()))
        lambda_q = (route_q * self.lambdas).sum(dim=-1, keepdim=True)
        lambda_k = (route_k * self.lambdas).sum(dim=-1, keepdim=True)
        q = q - lambda_q * q_prime
        k = k - lambda_k * k_prime
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        return torch.matmul(attn, v)


class DyITA(nn.Module):
    """Image tokens attend to the stage prompt embeddings through the three DyITA operations."""

    def __init__(self, dim, n_heads=2, n_projectors=3, n_kernel_factors=9, n_diff_factors=9,
                 gamma_init=3.0, lambda_init=0.01, dropout=0.0):
        super().__init__()
        head_dim = dim // n_heads
        self.dyn_proj = DynamicProjection(dim, dim, head_dim, n_heads, n_projectors)
        self.kernel_q = DynamicMeasureKernel(head_dim, n_kernel_factors, gamma_init)
        self.kernel_k = DynamicMeasureKernel(head_dim, n_kernel_factors, gamma_init)
        self.kernel_q_prime = DynamicMeasureKernel(head_dim, n_kernel_factors, gamma_init)
        self.kernel_k_prime = DynamicMeasureKernel(head_dim, n_kernel_factors, gamma_init)
        self.tdo = DifferentialOperator(head_dim, n_diff_factors, lambda_init)
        self.local_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prompts):
        b, c, h, w = x.shape
        local = self.local_residual(x)
        q, k, v, q_prime, k_prime = self.dyn_proj(x.flatten(2).transpose(1, 2), prompts)
        out = self.tdo(self.kernel_q(q), self.kernel_q_prime(q_prime),
                       self.kernel_k(k), self.kernel_k_prime(k_prime), v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = self.dropout(self.out_proj(out))
        return out.transpose(1, 2).view(b, c, h, w) + local


class GatedConvFFN(nn.Module):
    """1x1 expand -> SiLU -> depth-wise 3x3 on the value half -> gating -> 1x1 project."""

    def __init__(self, dim, hidden_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Conv2d(dim, 2 * hidden, 1, bias=False)
        self.act = nn.SiLU()
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.dwconv_bn = nn.BatchNorm2d(hidden)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, dim, 1, bias=False)
        self.drop = nn.Dropout(drop)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        value, gate = self.fc1(x).chunk(2, dim=1)
        value = self.act2(self.dwconv_bn(self.dwconv(self.act(value))))
        x = self.drop(value * self.act(gate))
        return self.drop(self.fc2(x))


class ITABlock(nn.Module):
    """x + LS1 * DyITA(BN(x), prompts), then x + LS2 * FFN(BN(x))."""

    def __init__(self, dim, n_heads=2, ffn_ratio=4.0, drop_path=0.0, init_value=0.1, **dyita_kwargs):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.dyita = DyITA(dim, n_heads, **dyita_kwargs)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = GatedConvFFN(dim, ffn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.layer_scale_1 = nn.Parameter(torch.full((1, dim, 1, 1), float(init_value)))
        self.layer_scale_2 = nn.Parameter(torch.full((1, dim, 1, 1), float(init_value)))

    def forward(self, x, prompts):
        x = x + self.drop_path(self.layer_scale_1 * self.dyita(self.norm1(x), prompts))
        return x + self.drop_path(self.layer_scale_2 * self.ffn(self.norm2(x)))
