"""Per-image Dice, IoU and HD95 (medpy), macro-averaged over structures."""

from collections import defaultdict

import numpy as np

if not hasattr(np, "bool"):  # older medpy releases use the removed np.bool alias
    np.bool = np.bool_
from medpy.metric.binary import dc, hd95, jc  # noqa: E402

from .prompts import DATASET_LABELS, LABEL_TO_ID, LABELS, VIEW_STRUCTURES

METRICS = ("dice", "iou", "hd95")


def evaluated_labels(dataset, view):
    """Structures that are both annotated by the dataset and visible in the view."""
    return [l for l in LABELS if l in DATASET_LABELS[dataset] and l in VIEW_STRUCTURES[view]]


def decode(probs, labels, threshold=0.5):
    """[5, H, W] per-prompt probabilities -> label map, restricted to the given structures."""
    probs = probs.copy()
    for i, label in enumerate(LABELS):
        if label not in labels:
            probs[i] = 0.0
    best = probs.max(axis=0)
    pred = (probs.argmax(axis=0) + 1).astype(np.uint8)
    pred[best < threshold] = 0
    return pred


def case_metrics(pred, gt, class_id):
    p, g = pred == class_id, gt == class_id
    if not p.any() and not g.any():
        return {"dice": 1.0, "iou": 1.0, "hd95": 0.0}
    if not p.any() or not g.any():
        return {"dice": 0.0, "iou": 0.0, "hd95": float("inf")}
    return {"dice": dc(p, g), "iou": jc(p, g), "hd95": hd95(p, g, voxelspacing=(1.0, 1.0))}


class MetricTracker:
    """Collects per-image, per-structure metrics under several groupings (e.g. view or dataset)."""

    def __init__(self):
        self.values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def update(self, pred, gt, labels, groups):
        for label in labels:
            m = case_metrics(pred, gt, LABEL_TO_ID[label])
            for group in groups:
                for key in METRICS:
                    if np.isfinite(m[key]):   # images where a structure is missed count 0 Dice but no HD95
                        self.values[group][label][key].append(float(m[key]))

    def summary(self, group):
        """{label: {metric: mean}} plus 'mean': macro average over structures."""
        out = {label: {k: float(np.mean(v)) for k, v in metrics.items()}
               for label, metrics in self.values[group].items()}
        out["mean"] = {k: float(np.mean([out[l][k] for l in out if k in out[l]])) for k in METRICS}
        return out

    def groups(self):
        return sorted(self.values)
