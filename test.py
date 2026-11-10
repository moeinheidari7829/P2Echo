import argparse
import json

import torch

from p2echo import P2Echo, TextEncoder
from p2echo.data import load_split, make_loader
from p2echo.engine import evaluate


def get_args():
    p = argparse.ArgumentParser(description="Evaluate P2Echo (Tables III and IV)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--split", default="test", choices=["val", "test", "external"])
    p.add_argument("--data_root", default="")
    p.add_argument("--text_model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    p.add_argument("--save_json", default=None)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = TextEncoder(args.text_model).to(device)
    model = P2Echo(text_dim=text_encoder.dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])

    loader = make_loader(load_split(args.splits, args.split, args.data_root), args.batch_size, args.num_workers)
    tracker = evaluate(model, text_encoder, loader, device, amp=not args.fp32)

    results = {}
    print(f"{'':24s}{'mDice':>8s}{'mIoU':>8s}{'HD95':>8s}")
    for group in ["all"] + [g for g in tracker.groups() if g != "all"]:
        s = tracker.summary(group)
        results[group] = s
        m = s["mean"]
        print(f"{group:24s}{100 * m['dice']:8.2f}{100 * m['iou']:8.2f}{m['hd95']:8.2f}")
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
