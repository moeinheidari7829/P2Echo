"""
Build a split file from a data folder laid out as <data_root>/<Dataset>/<View>/{images,masks}/<name>.png.
Development datasets are split 80/10/10 into train/val/test; held-out datasets go to "external".
"""

import argparse
import json
import os
import random

INTERNAL = ("Camus", "CardiacUDA", "EchoCP", "EchoNet-Pediatric", "SegRWMA")
EXTERNAL = ("CardiacNet", "HMCQU")


def collect(data_root, dataset):
    items = []
    for view in sorted(os.listdir(os.path.join(data_root, dataset))):
        image_dir = os.path.join(dataset, view, "images")
        for name in sorted(os.listdir(os.path.join(data_root, image_dir))):
            items.append({"image_path": os.path.join(image_dir, name),
                          "mask_path": os.path.join(dataset, view, "masks", name),
                          "view": view, "dataset": dataset})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", default="splits.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": [], "external": []}
    for dataset in INTERNAL:
        items = collect(args.data_root, dataset)
        rng.shuffle(items)
        n_train, n_val = int(0.8 * len(items)), int(0.1 * len(items))
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]
    for dataset in EXTERNAL:
        if os.path.isdir(os.path.join(args.data_root, dataset)):
            splits["external"] += collect(args.data_root, dataset)

    with open(args.output, "w") as f:
        json.dump(splits, f, indent=1)
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
