import argparse
import os
import random
import time

import numpy as np
import torch

from p2echo import P2Echo, TextEncoder
from p2echo.data import build_train_transforms, load_split, make_loader
from p2echo.engine import evaluate, prompt_embeddings
from p2echo.losses import DeepSupervisionLoss
from p2echo.prompts import annotated_mask, binary_targets, sample_active_prompts, shuffle_prompts


def get_args():
    p = argparse.ArgumentParser(description="Train P2Echo")
    p.add_argument("--splits", required=True, help="split file, see README")
    p.add_argument("--data_root", default="")
    p.add_argument("--pvt_weights", default="pretrained/pvt_v2_b2.pth")
    p.add_argument("--text_model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output", default="runs/p2echo")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--aug_clip", type=float, default=1.5)
    p.add_argument("--full_prompt_prob", type=float, default=0.25,
                   help="probability of requesting all annotated structures for a training image")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="path to last.pth")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model, weight_decay):
    """No weight decay on biases, norms, LayerScale and the DyITA gamma/lambda banks."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        n = name.lower()
        skip = (p.ndim <= 1 or n.endswith(".bias") or "norm" in n or "layer_scale" in n
                or n.endswith(".gammas") or n.endswith(".lambdas"))
        (no_decay if skip else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def augment(transform, image, mask, clip):
    out = transform(data=image.numpy().astype(np.float32), seg=mask.numpy()[:, None].astype(np.int16))
    data = np.nan_to_num(out["data"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    data = np.clip(data, -clip, clip).astype(np.float32, copy=False)
    return torch.from_numpy(data), torch.from_numpy(out["seg"][:, 0]).long()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.output, exist_ok=True)
    log_file = open(os.path.join(args.output, "train.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(vars(args))
    train_loader = make_loader(load_split(args.splits, "train", args.data_root), args.batch_size,
                               args.num_workers, shuffle=True, image_size=args.image_size)
    val_loader = make_loader(load_split(args.splits, "val", args.data_root), args.batch_size,
                             args.num_workers, image_size=args.image_size)
    transform = build_train_transforms(args.image_size)
    text_encoder = TextEncoder(args.text_model).to(device)
    model = P2Echo(img_size=args.image_size, text_dim=text_encoder.dim,
                   pretrained_encoder=args.pvt_weights).to(device)
    log(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.SGD(param_groups(model, args.weight_decay), lr=args.lr, momentum=args.momentum)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / total_steps) ** 0.9)
    scaler = torch.amp.GradScaler("cuda")
    criterion = DeepSupervisionLoss()

    start_epoch, best_dice = 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_dice = state["epoch"] + 1, state["best_dice"]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            image, mask = augment(transform, batch["image"], batch["mask"], args.aug_clip)
            image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            # Request a random subset of the annotated structures; the other slots receive the
            # null prompt and an empty target. Prompt order is shuffled per image.
            annotated = annotated_mask(batch["dataset"], device)
            active = sample_active_prompts(annotated, args.full_prompt_prob)
            targets = binary_targets(mask, active)
            emb = prompt_embeddings(text_encoder, batch["view"], active, device)
            emb, targets, annotated = shuffle_prompts(emb, targets, annotated)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(image, emb), targets, annotated)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            if step % 50 == 0:
                log(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        val = evaluate(model, text_encoder, val_loader, device).summary("all")["mean"]
        log(f"epoch {epoch} train_loss {running / len(train_loader):.4f} "
            f"val_dice {val['dice']:.4f} val_iou {val['iou']:.4f} val_hd95 {val['hd95']:.2f}")
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": best_dice},
                       os.path.join(args.output, "best.pth"))
            log(f"saved best.pth (val_dice {best_dice:.4f})")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_dice": best_dice}, os.path.join(args.output, "last.pth"))


if __name__ == "__main__":
    main()
