# TML 2026 — Assignment 3 (Robustness)

Code for an adversarially robust 9-class CIFAR-style image classifier.
This README explains **how to reproduce our best leaderboard result**.

## Best result
A **ResNet-50** trained with **PGD adversarial training** (L∞, ε = 8/255) plus
cosine LR, label smoothing, and EMA weight averaging.

| | clean | robust | unified score |
|---|---|---|---|
| validation (local) | 0.705 | 0.486 | **0.596** |
| public leaderboard | — | — | **0.604** |

## Reproduce

### 1. Environment
- Python 3.10+, PyTorch 2.3 + torchvision (the `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`
  Docker image used on the cluster already has these).

### 2. Data
Download `train.npz` into the repo folder:
```bash
wget "https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz"
```

### 3. Train the best model
```bash
python train2.py --arch resnet50 --method pgd \
    --cosine 1 --ls 0.1 --ema 0.999 \
    --epochs 200 --batch 192 --out model_best.pt
```
This saves `model_best.pt` = the EMA state_dict of the epoch with the best
`0.5*clean + 0.5*robust` score on a held-out split (guards against robust
overfitting). Training takes ~10 h on one GPU.

On the Saarland HPC cluster, instead submit the included job file (edit
`<USERNAME>` first):
```bash
mkdir -p runlogs
condor_submit job.sub
```

### 4. Submit to the leaderboard
Edit `submission.py` — set `API_KEY` (your team key), `MODEL_PATH="model_best.pt"`,
`MODEL_NAME="resnet50"` — then:
```bash
python submission.py
```

## Files
- `train.py` — minimal standalone PGD-AT baseline (ResNet-18).
- `train2.py` — full training script; all reported experiments are selected by flags.
- `job.sub` — HTCondor submit file for the best model.
- `submission.py` — leaderboard submission script.
- `EXPERIMENTS.md` — exact command for every experiment in the report.
