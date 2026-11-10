"""Inference and evaluation loop shared by training (validation) and testing."""

import torch

from .metrics import MetricTracker, decode, evaluated_labels
from .prompts import annotated_mask, prompt_texts


def prompt_embeddings(text_encoder, views, active, device):
    emb = text_encoder(prompt_texts(views, active.bool().cpu()), device)
    return emb.view(len(views), active.shape[1], -1)


@torch.no_grad()
def evaluate(model, text_encoder, loader, device, amp=True, threshold=0.5):
    """
    Every structure annotated by an image's dataset is prompted, the remaining slots get the
    null prompt. Returns a MetricTracker with groups 'all', 'view/<view>' and 'dataset/<name>'.
    """
    model.eval()
    tracker = MetricTracker()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        views, datasets = batch["view"], batch["dataset"]
        active = annotated_mask(datasets, device)
        emb = prompt_embeddings(text_encoder, views, active, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            probs = torch.sigmoid(model(image, emb)[0].float()).cpu().numpy()
        masks = batch["mask"].numpy()
        for i in range(len(views)):
            labels = evaluated_labels(datasets[i], views[i])
            if labels:
                pred = decode(probs[i], labels, threshold)
                tracker.update(pred, masks[i], labels, ["all", f"view/{views[i]}", f"dataset/{datasets[i]}"])
    return tracker
