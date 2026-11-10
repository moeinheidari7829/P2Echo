"""P2Echo: PVTv2-B2 encoder, prompt decoder, four ITA decoder stages and prompt-conditioned mask heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dyita import ITABlock
from .pvtv2 import pvt_v2_b2
from .transformer import PromptDecoder, sine_position_encoding_2d


class EUCB(nn.Module):
    """Efficient up-convolution block: 2x upsample, depth-wise 3x3, BN, LeakyReLU, 1x1 projection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pwc = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.pwc(self.up_dwc(x))


class MaskHead(nn.Module):
    """One logit map per prompt: scaled dot product between pixel and prompt embeddings."""

    def __init__(self, image_dim, prompt_dim, mask_dim=64):
        super().__init__()
        self.scale = mask_dim ** -0.5
        self.image_projection = nn.Conv2d(image_dim, mask_dim, 1, bias=False)
        self.prompt_projection = nn.Linear(prompt_dim, mask_dim, bias=False)
        nn.init.xavier_uniform_(self.image_projection.weight)
        nn.init.xavier_uniform_(self.prompt_projection.weight)

    def forward(self, x, prompts):
        pixels = self.image_projection(x)
        prompts = self.prompt_projection(prompts)
        return torch.einsum("bmhw,bnm->bnhw", pixels, prompts) * self.scale


class ITADecoder(nn.Module):
    def __init__(self, channels=(512, 320, 128, 64), n_heads=(2, 2, 2, 2), mask_dim=64,
                 drop_path=0.1, **dyita_kwargs):
        super().__init__()
        c4, c3, c2, c1 = channels
        block = lambda dim, heads: ITABlock(dim, heads, drop_path=drop_path, **dyita_kwargs)
        self.dec4 = block(c4, n_heads[0])
        self.up3 = EUCB(c4, c3)
        self.dec3 = block(c3, n_heads[1])
        self.up2 = EUCB(c3, c2)
        self.dec2 = block(c2, n_heads[2])
        self.up1 = EUCB(c2, c1)
        self.dec1 = block(c1, n_heads[3])
        self.output_head = MaskHead(c1, c1, mask_dim)
        self.deep_supervision_heads = nn.ModuleList([MaskHead(c2, c2, mask_dim), MaskHead(c3, c3, mask_dim)])

    def forward(self, features, prompts):
        f1, f2, f3, f4 = features
        p4, p3, p2, p1 = prompts
        d4 = self.dec4(f4, p4)
        d3 = self.dec3(self.up3(d4) + f3, p3)
        ds3 = self.deep_supervision_heads[1](d3, p3)
        d2 = self.dec2(self.up2(d3) + f2, p2)
        ds2 = self.deep_supervision_heads[0](d2, p2)
        d1 = self.dec1(self.up1(d2) + f1, p1)
        logits = self.output_head(d1, p1)
        logits = F.interpolate(logits, size=(4 * f1.shape[-2], 4 * f1.shape[-1]),
                               mode="bilinear", align_corners=False)
        return logits, ds2, ds3


class P2Echo(nn.Module):
    """
    Args:
        img_size: input resolution (square, divisible by 32).
        text_dim: dimension of the frozen text-encoder embeddings.
        pretrained_encoder: optional path to ImageNet PVTv2-B2 weights.

    forward(image [B,3,H,W], prompt_embeddings [B,N,text_dim]) returns
    (logits [B,N,H,W], aux [B,N,H/8,W/8], aux [B,N,H/16,W/16]); output n follows prompt n.
    """

    def __init__(self, img_size=256, text_dim=1024, query_dim=256, mask_dim=64,
                 prompt_layers=3, prompt_heads=8, prompt_ffn_dim=1024, drop_path=0.1,
                 ita_heads=(2, 2, 2, 2), n_projectors=3, n_kernel_factors=9, n_diff_factors=9,
                 gamma_init=3.0, lambda_init=0.01, pretrained_encoder=None):
        super().__init__()
        channels = (512, 320, 128, 64)
        self.encoder = pvt_v2_b2()
        if pretrained_encoder:
            self._load_encoder(pretrained_encoder)

        self.project_bottleneck = nn.Linear(channels[0], query_dim)
        self.project_text = nn.Linear(text_dim, query_dim)
        side = img_size // 32
        self.register_buffer("pos_embed", sine_position_encoding_2d(side, side, query_dim))
        self.prompt_decoder = PromptDecoder(query_dim, prompt_heads, prompt_ffn_dim, prompt_layers)
        self.stage_prompt_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(query_dim, query_dim // 2), nn.GELU(), nn.Linear(query_dim // 2, c))
            for c in channels])
        self.decoder = ITADecoder(
            channels, ita_heads, mask_dim, drop_path, n_projectors=n_projectors,
            n_kernel_factors=n_kernel_factors, n_diff_factors=n_diff_factors,
            gamma_init=gamma_init, lambda_init=lambda_init)

        for module in [self.project_bottleneck, self.project_text, *self.stage_prompt_projections]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _load_encoder(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        own = self.encoder.state_dict()
        missing = [k for k in own if k not in state or state[k].shape != own[k].shape]
        if missing:
            raise RuntimeError(f"PVTv2-B2 weights at {path} are incomplete, e.g. {missing[:4]}")
        self.encoder.load_state_dict({k: state[k] for k in own})

    def forward(self, image, prompt_embeddings):
        features = self.encoder(image)
        memory = self.project_bottleneck(features[-1].flatten(2).permute(2, 0, 1))  # [HW, B, C]
        prompts = self.project_text(prompt_embeddings.transpose(0, 1))              # [N, B, C]
        prompts = self.prompt_decoder(prompts, memory, self.pos_embed).transpose(0, 1)
        stage_prompts = [proj(prompts) for proj in self.stage_prompt_projections]
        return self.decoder(features, stage_prompts)
