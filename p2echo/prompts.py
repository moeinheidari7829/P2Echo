"""View-aware text prompts and per-sample prompt sampling."""

import torch

LABELS = ("LV", "MYO", "LA", "RV", "RA")          # output classes 1..5, background is 0
LABEL_TO_ID = {name: i + 1 for i, name in enumerate(LABELS)}

VIEW_NAMES = {
    "2CH": "Apical 2-Chamber (2CH)",
    "3CH": "Apical 3-Chamber (3CH)",
    "4CH": "Apical 4-Chamber (4CH)",
    "PSAX": "Parasternal Short-Axis (PSAX)",
}

# Structures anatomically visible in each view.
VIEW_STRUCTURES = {
    "4CH": ("LV", "LA", "RA", "RV", "MYO"),
    "2CH": ("LV", "LA", "MYO"),
    "3CH": ("LV", "LA", "MYO"),
    "PSAX": ("LV", "RV", "MYO"),
}

# Structures annotated by each dataset.
DATASET_LABELS = {
    "Camus": ("LV", "MYO", "LA"),
    "CardiacUDA": ("LV", "LA", "RV", "RA"),
    "EchoCP": ("LV", "LA", "RV", "RA"),
    "EchoNet-Pediatric": ("LV",),
    "SegRWMA": ("LV",),
    "CardiacNet": ("LV", "LA", "RV", "RA"),
    "HMCQU": ("LV", "MYO"),
}

DESCRIPTIONS = {
    "4CH": {
        "LV": "The LV appears as a bullet-shaped chamber forming the apex.",
        "LA": "The LA is located posterior to the LV.",
        "RA": "The RA is located next to the LA and inferior to the RV.",
        "RV": "The RV appears triangular and usually smaller than the LV.",
        "MYO": "The LV myocardium includes inferoseptum, anterolateral, apical septal, and apical lateral segments.",
    },
    "2CH": {
        "LV": "The LV is fully visualized from base to apex.",
        "LA": "The LA is seen superior to the LV.",
        "MYO": "The LV myocardium includes anterior and inferior wall segments (basal, mid, apical).",
    },
    "3CH": {
        "LV": "The LV and LVOT are visible in long-axis.",
        "LA": "The LA is visible in long-axis.",
        "MYO": "The LV myocardium includes anteroseptum, inferolateral, apical septal, and apical lateral segments.",
    },
    "PSAX": {
        "LV": "The LV appears circular at mid-ventricle.",
        "RV": "The RV appears crescent-shaped, wrapping anteriorly around the LV.",
        "MYO": "The LV myocardium shows all 6 radial segments.",
    },
}

NULL_REQUEST = "No anatomical structure is requested. Return an empty segmentation mask."


def view_header(view):
    view = view.upper()
    return f"{VIEW_NAMES[view]}: Structures visualized: {', '.join(sorted(VIEW_STRUCTURES[view]))}."


def structure_prompt(view, label):
    """Prompt requesting one structure, e.g. view_header + ' Segment the LV: <description>'."""
    view = view.upper()
    if label in VIEW_STRUCTURES[view]:
        request = f"Segment the {label}: {DESCRIPTIONS[view][label]}"
    else:
        request = f"The {label} is not visualized in this view; do not segment it. Background only."
    return f"{view_header(view)} {request}"


def null_prompt(view):
    """Prompt for an output slot whose structure is not requested; its target is empty."""
    return f"{view_header(view)} {NULL_REQUEST}"


def prompt_texts(views, active):
    """Prompt strings for a batch, ordered sample by sample. active: [B, 5] bool."""
    return [structure_prompt(view, label) if active[b, i] else null_prompt(view)
            for b, view in enumerate(views) for i, label in enumerate(LABELS)]


def annotated_mask(datasets, device):
    """[B, 5] mask of structures annotated by each sample's dataset."""
    mask = torch.zeros(len(datasets), len(LABELS), device=device)
    for b, name in enumerate(datasets):
        if name not in DATASET_LABELS:
            raise KeyError(f"Unknown dataset '{name}'. Add its annotated labels to DATASET_LABELS.")
        for label in DATASET_LABELS[name]:
            mask[b, LABEL_TO_ID[label] - 1] = 1.0
    return mask


def sample_active_prompts(annotated, full_probability=0.25):
    """
    Randomly choose which annotated structures are requested for each training sample.
    With probability full_probability all of them are; otherwise a uniformly sized random
    subset (possibly empty), so every structure is also seen with a null prompt.
    """
    active = torch.zeros_like(annotated)
    for b in range(annotated.shape[0]):
        valid = torch.nonzero(annotated[b] >= 0.5).flatten()
        n = valid.numel()
        if n == 0:
            continue
        if torch.rand((), device=annotated.device).item() < full_probability:
            chosen = valid
        else:
            k = int(torch.randint(0, n + 1, (), device=annotated.device).item())
            chosen = valid[torch.randperm(n, device=annotated.device)[:k]]
        active[b, chosen] = 1.0
    return active


def binary_targets(mask, active=None):
    """[B, H, W] label map -> [B, 5, H, W] per-structure targets, emptied where not requested."""
    targets = torch.stack([(mask == i + 1) for i in range(len(LABELS))], dim=1).float()
    if active is not None:
        targets = targets * active[:, :, None, None]
    return targets


def shuffle_prompts(*tensors):
    """Apply the same random permutation of the prompt axis (dim 1) to every tensor, per sample."""
    b, n = tensors[0].shape[:2]
    perm = torch.stack([torch.randperm(n, device=tensors[0].device) for _ in range(b)])
    out = []
    for t in tensors:
        index = perm.view(b, n, *([1] * (t.ndim - 2))).expand(b, n, *t.shape[2:])
        out.append(t.gather(1, index))
    return out
