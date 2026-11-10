"""Echocardiography dataset, split files and training augmentation."""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def normalize_image(img, eps=1e-8):
    """Z-score normalisation over non-zero (foreground) pixels; background stays 0."""
    fg = img.ne(0).any(dim=0)
    if not fg.any():
        return (img - img.mean()) / img.std(unbiased=False).clamp_min(eps)
    fg = fg.unsqueeze(0).expand_as(img)
    out = img.clone()
    out[fg] = (img[fg] - img[fg].mean()) / img[fg].std(unbiased=False).clamp_min(eps)
    out[~fg] = 0
    return out


class EchoDataset(Dataset):
    """
    Items are dicts with image_path, mask_path, view (2CH/3CH/4CH/PSAX) and dataset.
    Masks are single-channel label maps: 0 BG, 1 LV, 2 MYO, 3 LA, 4 RV, 5 RA.
    """

    def __init__(self, items, image_size=256):
        self.items = items
        self.size = (image_size, image_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB").resize(self.size, Image.BILINEAR)
        mask = Image.open(item["mask_path"]).convert("L").resize(self.size, Image.NEAREST)
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
        return {
            "image": normalize_image(image),
            "mask": torch.from_numpy(np.array(mask)).long(),
            "view": str(item["view"]).upper(),
            "dataset": item["dataset"],
            "name": os.path.basename(item["image_path"]),
        }


def load_split(splits_json, split, data_root=""):
    """Read one split ("train", "val", "test" or "external") of a split file; paths are relative to data_root."""
    with open(splits_json) as f:
        items = json.load(f)[split]
    for item in items:
        for key in ("image_path", "mask_path"):
            item[key] = os.path.join(data_root, item[key])
    return items


def make_loader(items, batch_size, num_workers, shuffle=False, image_size=256):
    return DataLoader(EchoDataset(items, image_size), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)


class SafeGammaTransform:
    """Per-channel gamma on min-max rescaled intensities (valid for negative inputs)."""

    def __init__(self, gamma_range=(0.7, 1.5), p_per_sample=0.3):
        self.gamma_range = gamma_range
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data = data_dict["data"].astype(np.float32, copy=False)
        for b in range(data.shape[0]):
            if np.random.random() >= self.p_per_sample:
                continue
            for c in range(data.shape[1]):
                gamma = np.random.uniform(*self.gamma_range)
                x = data[b, c]
                lo = float(np.min(x))
                span = float(np.max(x - lo))
                if span > 0:
                    data[b, c] = np.power((x - lo) / (span + 1e-8), gamma, dtype=np.float32) * span + lo
        data_dict["data"] = data
        return data_dict


def build_train_transforms(image_size=256):
    """nnU-Net style 2D augmentation (batchgenerators)."""
    from batchgenerators.transforms.abstract_transforms import Compose
    from batchgenerators.transforms.color_transforms import (
        BrightnessMultiplicativeTransform, ContrastAugmentationTransform)
    from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
    from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
    from batchgenerators.transforms.spatial_transforms import SpatialTransform

    rotation = (-15.0 / 180.0 * np.pi, 15.0 / 180.0 * np.pi)
    return Compose([
        SpatialTransform(
            patch_size=(image_size, image_size), patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0.0, 0.0), sigma=(0.0, 0.0),
            do_rotation=True, angle_x=rotation, angle_y=(0.0, 0.0), angle_z=(0.0, 0.0),
            do_scale=True, scale=(0.7, 1.4), border_mode_data="constant", border_cval_data=0.0,
            order_data=3, border_mode_seg="constant", border_cval_seg=0, order_seg=0,
            random_crop=False, p_el_per_sample=0.0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False),
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform(blur_sigma=(0.5, 1.0), different_sigma_per_channel=True, p_per_sample=0.2),
        BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), p_per_sample=0.15),
        ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15),
        SafeGammaTransform(gamma_range=(0.7, 1.5), p_per_sample=0.3),
        SimulateLowResolutionTransform(zoom_range=(0.5, 1.0), per_channel=True, p_per_sample=0.25,
                                       order_downsample=0, order_upsample=3),
    ])
