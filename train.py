"""
Adversarial (robust) training for TML 2026 - Assignment 3.

Trains a torchvision ResNet on the 50k/9-class 32x32 dataset using
PGD adversarial training (Madry et al., 2018). Saves the state_dict of the
checkpoint with the best UNIFIED score (0.5*clean + 0.5*robust) measured on a
held-out validation split, so we don't get hurt by "robust overfitting".

IMPORTANT constraints from the assignment:
- The eval server rebuilds a *plain* torchvision resnet18/34/50 and loads our
  state_dict. So we must NOT change the architecture (no custom normalization
  layer, no modified stem). Inputs are pixels in [0,1]; we train on [0,1] too.
- Output = 9 logits, input = 3x32x32.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.transforms as T

# ----------------------------- config ---------------------------------------
BASE      = Path(__file__).parent
DATA      = BASE / "train.npz"
OUT       = BASE / "model.pt"          # best checkpoint (state_dict only)

ARCH      = "resnet18"                  # resnet18 | resnet34 | resnet50
EPOCHS    = 100
BATCH     = 256
LR        = 0.1
MOMENTUM  = 0.9
WD        = 5e-4
MILESTONES = [50, 75]                   # LR x0.1 at these epochs (Madry schedule)

EPS       = 8 / 255                     # L-inf perturbation budget
ALPHA     = 2 / 255                     # PGD step size
STEPS     = 10                          # PGD steps during training
EVAL_STEPS = 20                         # stronger PGD for validation
VAL_FRAC  = 0.04                        # fraction held out for validation
EVAL_EVERY = 2                          # run (slow) robust eval every N epochs
SEED      = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------- data ------------------------------------------
d = np.load(DATA)
X = torch.from_numpy(d["images"]).float() / 255.0   # (N,3,32,32) in [0,1]
y = torch.from_numpy(d["labels"]).long()            # (N,) in [0,8]
N = len(X)
print(f"Loaded {N} images, shape {tuple(X.shape[1:])}, "
      f"labels {y.min().item()}..{y.max().item()}", flush=True)

perm = torch.randperm(N)
n_val = int(N * VAL_FRAC)
val_idx, tr_idx = perm[:n_val], perm[n_val:]
Xtr, ytr = X[tr_idx], y[tr_idx]
Xval, yval = X[val_idx].to(device), y[val_idx].to(device)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
])

class TrainDS(Dataset):
    def __init__(self, imgs, labels, tf):
        self.imgs, self.labels, self.tf = imgs, labels, tf
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        return self.tf(self.imgs[i]), self.labels[i]

train_loader = DataLoader(TrainDS(Xtr, ytr, train_tf), batch_size=BATCH,
                          shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

# ----------------------------- model -----------------------------------------
def build_model(arch):
    m = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}[arch](weights=None)
    m.fc = nn.Linear(m.fc.in_features, 9)
    return m

model = build_model(ARCH).to(device)
opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=MILESTONES, gamma=0.1)

# ----------------------------- PGD attack ------------------------------------
def pgd(model, x, y, eps, alpha, steps):
    """Return adversarial inputs in [0,1] that maximize the loss."""
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(model(x + delta), y)
        grad, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * grad.sign()).clamp(-eps, eps)
        delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    return (x + delta).detach()

# ----------------------------- evaluation ------------------------------------
@torch.no_grad()
def clean_acc(model, X, y, bs=512):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        pred = model(X[i:i+bs]).argmax(1)
        correct += (pred == y[i:i+bs]).sum().item()
    return correct / len(X)

def robust_acc(model, X, y, bs=256, n=2000):
    """PGD robust accuracy on first n validation samples."""
    model.eval()
    X, y = X[:n], y[:n]
    correct = 0
    for i in range(0, len(X), bs):
        xb, yb = X[i:i+bs], y[i:i+bs]
        xadv = pgd(model, xb, yb, EPS, ALPHA, EVAL_STEPS)
        with torch.no_grad():
            correct += (model(xadv).argmax(1) == yb).sum().item()
    return correct / len(X)

# ----------------------------- train loop ------------------------------------
best_score = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        model.eval()                       # attack with eval-mode BN stats
        xadv = pgd(model, xb, yb, EPS, ALPHA, STEPS)
        model.train()
        loss = F.cross_entropy(model(xadv), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
    sched.step()

    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
        ca = clean_acc(model, Xval, yval)
        ra = robust_acc(model, Xval, yval)
        score = 0.5 * ca + 0.5 * ra
        flag = ""
        if ca > 0.50 and score > best_score:
            best_score = score
            torch.save(model.state_dict(), OUT)
            flag = "  <-- saved best"
        print(f"epoch {epoch:3d}  loss {running/len(train_loader):.3f}  "
              f"clean {ca:.4f}  robust {ra:.4f}  score {score:.4f}{flag}", flush=True)
    else:
        print(f"epoch {epoch:3d}  loss {running/len(train_loader):.3f}", flush=True)

print(f"Done. Best validation unified score = {best_score:.4f}. Saved to {OUT}", flush=True)

# ----------------------------- sanity check ----------------------------------
m = build_model(ARCH)
m.load_state_dict(torch.load(OUT, map_location="cpu"))
m.eval()
with torch.no_grad():
    out = m(torch.randn(1, 3, 32, 32))
assert out.shape == (1, 9), out.shape
print("Sanity OK: state_dict reloads into a plain", ARCH, "and outputs", tuple(out.shape))
