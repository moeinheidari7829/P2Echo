"""PVTv2-B2 backbone (Wang et al., "PVT v2: Improved Baselines with Pyramid Vision Transformer")."""

import math
from functools import partial

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.dwconv(x.transpose(1, 2).view(B, C, H, W))
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        x = self.drop(self.act(self.dwconv(self.fc1(x), H, W)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x = self.sr(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, in_chans=3, embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8),
                 mlp_ratios=(8, 8, 4, 4), qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        self.depths = depths
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            blocks = nn.ModuleList([
                Block(embed_dims[i], num_heads[i], mlp_ratios[i], qkv_bias, drop_rate, attn_drop_rate,
                      dpr[cur + j], norm_layer, sr_ratios[i])
                for j in range(depths[i])])
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm_layer(embed_dims[i]))
            cur += depths[i]
        self.apply(_init_weights)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(1, 5):
            x, H, W = getattr(self, f"patch_embed{i}")(x)
            for blk in getattr(self, f"block{i}"):
                x = blk(x, H, W)
            x = getattr(self, f"norm{i}")(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


def pvt_v2_b2():
    return PyramidVisionTransformerV2()
