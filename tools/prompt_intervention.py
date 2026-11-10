"""
Prompt intervention (Table VI). For every annotated structure in a stratified sample of test
images, only the text of that structure's prompt is replaced; all other prompts are kept fixed.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2echo import P2Echo, TextEncoder  # noqa: E402
from p2echo.data import EchoDataset, load_split  # noqa: E402
from p2echo.prompts import (DESCRIPTIONS, LABELS, VIEW_NAMES, VIEW_STRUCTURES, annotated_mask,  # noqa: E402
                            binary_targets, null_prompt, structure_prompt)

LONG_NAMES = {"LV": "left ventricle cavity", "MYO": "left ventricular myocardium", "LA": "left atrium",
              "RV": "right ventricle", "RA": "right atrium"}
PARAPHRASES = {
    "LV": "Delineate the left ventricular blood pool.",
    "MYO": "Outline the muscular wall of the left ventricle.",
    "LA": "Delineate the left atrial chamber.",
    "RV": "Outline the right ventricular blood pool.",
    "RA": "Delineate the right atrial chamber.",
}
CONDITIONS = ("canonical", "truncated", "clinical_paraphrase", "ambiguous", "irrelevant", "null")
TABLE_ORDER = ("canonical", "clinical_paraphrase", "truncated", "ambiguous", "irrelevant", "null")


def condition_prompt(condition, view, label):
    name = VIEW_NAMES[view]
    visible = ", ".join(sorted(VIEW_STRUCTURES[view]))
    if condition == "canonical":
        return structure_prompt(view, label)
    if condition == "truncated":
        return f"{name}: Segment the {LONG_NAMES[label]}."
    if condition == "clinical_paraphrase":
        if label not in VIEW_STRUCTURES[view]:
            return (f"{name}. The cardiac structures visible in this view are {visible}. "
                    f"The {LONG_NAMES[label]} is not visible in this view; output an empty mask.")
        return (f"{name}. The cardiac structures visible in this view are {visible}. "
                f"{PARAPHRASES[label]} {DESCRIPTIONS[view][label]}")
    if condition == "ambiguous":
        return f"{name}: Segment the cardiac chamber."
    if condition == "irrelevant":
        return f"{name}: Segment the liver."
    return null_prompt(view)


def stratified_sample(items, per_group, seed):
    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[(item["dataset"], item["view"])].append(i)
    rng = np.random.default_rng(seed)
    selected = []
    for key in sorted(groups):
        idx = np.asarray(groups[key], dtype=np.int64)
        selected += rng.choice(idx, size=min(per_group, len(idx)), replace=False).tolist()
    return selected


def dice_iou(prob, target, threshold=0.5):
    pred, target = prob >= threshold, target.bool()
    inter = (pred & target).sum(dim=(-2, -1)).float()
    p, t = pred.sum(dim=(-2, -1)).float(), target.sum(dim=(-2, -1)).float()
    return (2 * inter + 1e-6) / (p + t + 1e-6), (inter + 1e-6) / (p + t - inter + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--text_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--images_per_group", type=int, default=50, help="per dataset/view pair")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_csv", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    text_encoder = TextEncoder(args.text_model).to(device)
    model = P2Echo(text_dim=text_encoder.dim).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])

    items = load_split(args.splits, "test", args.data_root)
    for item in items:
        item["view"] = item["view"].upper()
    subset = stratified_sample(items, args.images_per_group, args.seed)
    loader = DataLoader(Subset(EchoDataset(items), subset), batch_size=min(args.batch_size, len(subset)))

    rows = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            views, datasets, names = batch["view"], batch["dataset"], batch["name"]
            annotated = annotated_mask(datasets, device).bool()
            targets = binary_targets(batch["mask"].to(device)).bool()
            positive = annotated & targets.flatten(2).any(dim=2)
            canonical = text_encoder(
                [structure_prompt(v, l) if annotated[b, i] else null_prompt(v)
                 for b, v in enumerate(views) for i, l in enumerate(LABELS)], device
            ).view(len(views), len(LABELS), -1)

            for i, label in enumerate(LABELS):
                sel = positive[:, i]
                if not sel.any():
                    continue
                idx = torch.nonzero(sel).flatten().tolist()
                for condition in CONDITIONS:
                    emb = canonical[sel].clone()
                    emb[:, i] = text_encoder([condition_prompt(condition, views[j], label) for j in idx], device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prob = torch.sigmoid(model(image[sel], emb)[0].float())[:, i]
                    dice, iou = dice_iou(prob, targets[sel, i])
                    for k, j in enumerate(idx):
                        rows.append({"dataset": datasets[j], "view": views[j], "image": names[j],
                                     "structure": label, "condition": condition,
                                     "dice": dice[k].item(), "iou": iou[k].item()})

    print(f"{len(subset)} images, {sum(r['condition'] == 'canonical' for r in rows)} structure instances")
    print(f"{'Condition':22s}{'Dice (%)':>18s}{'IoU (%)':>18s}")
    for condition in TABLE_ORDER:
        d = np.array([r["dice"] for r in rows if r["condition"] == condition]) * 100
        u = np.array([r["iou"] for r in rows if r["condition"] == condition]) * 100
        print(f"{condition:22s}{d.mean():10.2f} ± {d.std(ddof=1):5.2f}{u.mean():10.2f} ± {u.std(ddof=1):5.2f}")
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
