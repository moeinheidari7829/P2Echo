<h1 align="center">Prompt2Echo (P2Echo)</h1>
<h3 align="center">Text-Prompted Multi-View Echocardiography Segmentation with One Model</h3>

<p align="center">
  <a href="https://arxiv.org/abs/ARXIV_ID_PLACEHOLDER"><img src="https://img.shields.io/badge/arXiv-ARXIV_ID_PLACEHOLDER-b31b1b.svg" alt="arXiv"></a>
  <a href="https://moeinheidari7829.github.io/p2echo/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://drive.google.com/file/d/1wKF_MnWlMcpfYZTurb2wHOobENmhHQuj/view?usp=sharing"><img src="https://img.shields.io/badge/Checkpoint-Google%20Drive-34a853.svg" alt="Checkpoint"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://moeinheidari7829.github.io/p2echo/"><img src="assets/teaser.webp" width="100%" alt="P2Echo teaser"></a>
</p>

<p align="center"><b>One network. Four echocardiographic views. Five cardiac structures. Controlled by a text prompt.</b></p>

- **🫀 One model for every view.** A single P2Echo segments LV, MYO, LA, RV and RA in 2CH, 3CH, 4CH and PSAX,
  replacing a set of view-specific specialists.
- **💬 Text as the control signal.** A structured prompt names the view and the structure, e.g.
  *"Apical 4-Chamber (4CH): … Segment the LV: The LV appears as a bullet-shaped chamber forming the apex."*
  Each output map follows its prompt; unrelated or empty requests return an empty mask.
- **🔀 DyITA.** Dynamic Image-Text Alignment fuses the prompt into every decoder stage through dynamic projection,
  a dynamic measure kernel and a cross-modal differential operator.
- **⚡ Light and fast.** 37.47 M parameters, 6.90 GFLOPs and 64.68 FPS on one H100.

<p align="center">
  <img src="assets/architecture.png" width="100%" alt="P2Echo architecture">
  <br><sub>A PVTv2-B2 encoder and a frozen text encoder feed a prompt decoder; the resulting stage-specific
  prompt embeddings are injected into four ITA blocks, each built around DyITA.</sub>
</p>

## Results

<div align="center">
<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">mDice (%) ↑</th>
      <th colspan="4">Internal test set (multi-view)</th>
      <th colspan="2">External, held-out (4CH)</th>
    </tr>
    <tr>
      <th>2CH</th><th>3CH</th><th>4CH</th><th>PSAX</th><th>HMC-QU</th><th>CardiacNet</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>nnUNet</td><td>92.82</td><td>93.07</td><td>91.39</td><td>91.50</td><td>45.30</td><td>88.13</td></tr>
    <tr><td>SAM3 (fine-tuned)</td><td>93.39</td><td>92.70</td><td>90.85</td><td>91.18</td><td>43.46</td><td>90.62</td></tr>
    <tr><td>EchoONE</td><td>91.69</td><td>93.60</td><td>91.11</td><td>91.71</td><td>53.96</td><td>88.10</td></tr>
    <tr><td><b>P2Echo (ours)</b></td><td><b>96.94</b></td><td><b>94.71</b></td><td><b>93.87</b></td><td><b>92.00</b></td><td><b>82.22</b></td><td><b>92.95</b></td></tr>
  </tbody>
</table>
</div>

## Installation

```bash
git clone https://github.com/moeinheidari7829/P2Echo.git
cd P2Echo
conda create -n p2echo python=3.11 -y && conda activate p2echo
pip install -r requirements.txt
```

The Qwen3-Embedding-0.6B text encoder is downloaded from Hugging Face on first use.

## Data

We cannot redistribute the datasets. Please download them from their official sources:

| Dataset | Role | Link |
|---|---|---|
| CAMUS | train / test | [CREATIS CAMUS database](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html) |
| CardiacUDA | train / test | [GitHub repository](https://github.com/XiaoweiXu/CardiacUDA-dataset) |
| EchoCP | train / test | [GitHub repository](https://github.com/XiaoweiXu/EchoCP-An-Echocardiography-Dataset-in-Contrast-Transthoracic-Echocardiography-for-PFO-diagnosis) |
| EchoNet-Pediatric | train / test | [Project website](https://echonet.github.io/pediatric/) |
| SegRWMA | train / test | [GitHub repository](https://github.com/XiaoweiXu/Segment-level-Assessment-of-Regional-Wall-Motion-Abnormality-from-Echocardiography-Images) |
| CardiacNet | external | [Kaggle dataset page](https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos) |
| HMC-QU | external | [Kaggle dataset page](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset) |

Convert every annotated frame to a PNG image and a single-channel PNG label map with
`0 = background, 1 = LV, 2 = MYO, 3 = LA, 4 = RV, 5 = RA` (for HMC-QU the LV cavity is obtained by closing
and filling the myocardium contour, see the paper), and arrange them as

```
data/
├── Camus/{2CH,4CH}/{images,masks}/*.png
├── CardiacUDA/4CH/{images,masks}/*.png
├── EchoCP/4CH/...
├── EchoNet-Pediatric/{4CH,PSAX}/...
├── SegRWMA/{2CH,3CH,4CH}/...
├── CardiacNet/4CH/...
└── HMCQU/4CH/...
```

Then create a split file (80/10/10 per development dataset, external datasets held out):

```bash
python tools/make_splits.py --data_root data --output splits.json
```

The split file is a JSON with `train`, `val`, `test` and `external` lists of
`{"image_path", "mask_path", "view", "dataset"}` entries, with paths relative to `--data_root`.

## Pretrained weights

- ImageNet PVTv2-B2 (needed for training only):
  ```bash
  mkdir -p pretrained && wget -P pretrained https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
  ```
- P2Echo checkpoint: [Google Drive](https://drive.google.com/file/d/1wKF_MnWlMcpfYZTurb2wHOobENmhHQuj/view?usp=sharing). Place it at `checkpoints/p2echo.pth`.

## Training

```bash
python train.py --splits splits.json --data_root data --output runs/p2echo
```

Default arguments reproduce the paper setting: 200 epochs, SGD (lr 0.01, momentum 0.9, weight decay 1e-4),
batch size 32, polynomial decay, BF16 mixed precision, Boundary-DoU loss with deep supervision. The checkpoint
with the best validation mDice is saved as `runs/p2echo/best.pth`.

## Evaluation

Internal test set (per view, Table III) and external datasets (Table IV):

```bash
python test.py --checkpoint checkpoints/p2echo.pth --splits splits.json --data_root data --split test
python test.py --checkpoint checkpoints/p2echo.pth --splits splits.json --data_root data --split external
```

Prompt intervention (Table VI):

```bash
python tools/prompt_intervention.py --checkpoint checkpoints/p2echo.pth --splits splits.json --data_root data
```

Parameters, MACs and latency (Table VII):

```bash
python tools/benchmark.py --checkpoint checkpoints/p2echo.pth
```

## Using the model

```python
import torch
from p2echo import P2Echo, TextEncoder
from p2echo.prompts import structure_prompt

device = "cuda"
text_encoder = TextEncoder().to(device)
model = P2Echo().to(device).eval()
model.load_state_dict(torch.load("checkpoints/p2echo.pth", map_location="cpu")["model"])

prompts = [structure_prompt("4CH", s) for s in ("LV", "LA")]
emb = text_encoder(prompts, device).unsqueeze(0)              # [1, 2, 1024]
image = ...                                                     # [1, 3, 256, 256], see p2echo/data.py
probs = torch.sigmoid(model(image, emb)[0])                   # [1, 2, 256, 256], one map per prompt
```

## Citation

```bibtex
@article{heidari2026p2echo,
  title   = {Prompt2Echo (P2Echo): Text-Prompted Multi-View Echocardiography Segmentation with One Model},
  author  = {Heidari, Moein and Kimbowa, Alvin and Tabatabaei, Mahdi and Bozorgpour, Afshin and
             Ghorbani Kolahi, Sina and Jiang, River and Chen, Wenjin and Foran, David J. and
             Grewal, Jasmine and Merhof, Dorit and Hacihaliloglu, Ilker},
  journal = {arXiv preprint arXiv:ARXIV_ID_PLACEHOLDER},
  year    = {2026}
}
```

## Acknowledgements

This code builds on [PVTv2](https://github.com/whai362/PVT), [CENet](https://github.com/xmindflow/CENet), [DyDiLA](https://arxiv.org/abs/2601.13683v1)
[Boundary DoU loss](https://github.com/sunfan-bvb/BoundaryDoULoss), [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
and [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
