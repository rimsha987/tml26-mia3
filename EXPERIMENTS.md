# Experiments

Every row here corresponds 1:1 to a row in the report's results table. All use
`train2.py` except the baseline, which is the standalone `train.py`. "Local
score" is the best held-out `0.5*clean + 0.5*robust` reached during training.

| # | Description | Command | Local score |
|---|---|---|---|
| 1 | PGD-AT baseline (ResNet-18) | `python train.py` | 0.525 |
| 2 | TRADES + EMA (ResNet-18) | `python train2.py --arch resnet18 --method trades --beta 6 --lr 0.05 --ema 0.999 --epochs 100 --out m.pt` | 0.534 |
| 3 | PGD-AT, ResNet-50 (capacity) | `python train2.py --arch resnet50 --method pgd --ema 0.999 --epochs 100 --batch 192 --out m.pt` | 0.546 |
| 4 | TRADES, ResNet-34 (cos+LS+EMA) | `python train2.py --arch resnet34 --method trades --beta 6 --lr 0.05 --cosine 1 --ls 0.1 --ema 0.999 --epochs 150 --out m.pt` | 0.566 |
| 5 | TRADES, ResNet-50 (cos+LS+EMA) | `python train2.py --arch resnet50 --method trades --beta 6 --lr 0.05 --cosine 1 --ls 0.1 --ema 0.999 --epochs 150 --batch 192 --out m.pt` | 0.578 |
| 6 | PGD-AT + AWP, ResNet-50 | `python train2.py --arch resnet50 --method pgd --awp 0.01 --cosine 1 --ls 0.1 --ema 0.999 --epochs 150 --batch 192 --out m.pt` | 0.557 |
| 7 | TRADES + AWP, ResNet-50 | `python train2.py --arch resnet50 --method trades --awp 0.01 --beta 6 --lr 0.05 --cosine 1 --ls 0.1 --ema 0.999 --epochs 150 --batch 192 --out m.pt` | 0.556 |
| 8 | CutMix + PGD-AT, ResNet-50 | `python train2.py --arch resnet50 --method pgd --cutmix 1 --cosine 1 --ema 0.999 --epochs 200 --batch 192 --out m.pt` | 0.541 |
| 9 | **BEST — PGD-AT, ResNet-50 (cos+LS+EMA), 200 ep** | `python train2.py --arch resnet50 --method pgd --cosine 1 --ls 0.1 --ema 0.999 --epochs 200 --batch 192 --out model_best.pt` | **0.596** (leaderboard 0.604) |
| 10 | Clean-accuracy ceiling (diagnostic, no attack) | `python train2.py --arch resnet50 --method clean --cosine 1 --ls 0.1 --epochs 40 --batch 192 --out m.pt` | clean 0.81 |

Notes:
- All adversarial training uses L∞ ε=8/255, step α=2/255, 10 PGD steps; robust
  validation uses 20-step PGD on a held-out 4% split.
- TRADES uses a 5-epoch LR warmup + β warmup (0→β over 10 epochs) + gradient
  clipping (norm 1.0) + skipping non-finite batches for numerical stability.
- Row 9 is the submitted model and reproduces our best leaderboard score (see README).
