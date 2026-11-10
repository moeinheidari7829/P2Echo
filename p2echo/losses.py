"""Boundary Difference over Union loss (Sun et al., MICCAI 2023) with deep supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryDoULoss(nn.Module):
    """Applied per sample and per prompt to sigmoid probabilities; empty targets use mean(p)."""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def _bdou(self, score, target):
        kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=target.device)
        boundary = torch.conv2d(target[None, None], kernel[None, None], padding=1).squeeze() * target
        boundary[boundary == 5] = 0
        c = torch.count_nonzero(boundary)
        s = torch.count_nonzero(target)
        alpha = min((2 * (1 - (c + self.smooth) / (s + self.smooth)) - 1).item(), 0.8)
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return (z_sum + y_sum - 2 * intersect + self.smooth) / \
               (z_sum + y_sum - (1 + alpha) * intersect + self.smooth)

    def forward(self, logits, targets, valid):
        """logits/targets: [B, N, H, W]; valid: [B, N], 0 where the prompt is not supervised."""
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        losses = []
        for b in range(logits.shape[0]):
            for n in range(logits.shape[1]):
                if valid[b, n] < 0.5:
                    continue
                if targets[b, n].sum() > 0:
                    losses.append(self._bdou(probs[b, n], targets[b, n]))
                else:
                    losses.append(probs[b, n].mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class DeepSupervisionLoss(nn.Module):
    """Outputs ordered high to low resolution, weighted 4/7, 2/7, 1/7."""

    def __init__(self):
        super().__init__()
        self.loss = BoundaryDoULoss()

    def forward(self, outputs, targets, valid):
        weights = [1.0 / 2 ** i for i in range(len(outputs))]
        weights = [w / sum(weights) for w in weights]
        total = 0.0
        for w, out in zip(weights, outputs):
            total = total + w * self.loss(out, F.interpolate(targets, size=out.shape[-2:], mode="nearest"), valid)
        return total
