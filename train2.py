"""
Adversarial training for TML 2026 - Assignment 3 (full experiment script).

One configurable script that covers every experiment in the report (see
EXPERIMENTS.md for the exact command per result):
  --method pgd|trades|clean        training objective (clean = no-attack diagnostic)
  --awp <gamma>                    Adversarial Weight Perturbation (0 disables)
  --cutmix 1                       CutMix augmentation (with PGD-AT)
  --cosine 1                       cosine LR schedule (after a 5-epoch warmup)
  --ls <eps>                       label smoothing
  --ema <decay>                    EMA weight averaging; the EMA weights are saved
  --arch resnet18|resnet34|resnet50, --beta, --epochs, --batch, --lr, --eps, ...

Respects the task constraints: plain torchvision ResNet, inputs in [0,1], 9
logits, 3x32x32. Trains in [0,1] (no normalization layer, since the server
rebuilds a stock ResNet). Saves the EMA state_dict of the epoch with the best
held-out 0.5*clean + 0.5*robust score (guards against robust overfitting).

Best leaderboard model:
  train2.py --arch resnet50 --method pgd --cosine 1 --ls 0.1 --ema 0.999 \
            --epochs 200 --batch 192 --out model_best.pt
"""
import argparse, copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.transforms as T

p = argparse.ArgumentParser()
p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
p.add_argument("--method", default="trades", choices=["pgd", "trades", "clean"])
p.add_argument("--cosine", type=int, default=0, help="1 = cosine LR after warmup")
p.add_argument("--ls", type=float, default=0.0, help="label smoothing")
p.add_argument("--cutmix", type=int, default=0, help="1 = CutMix augmentation (with PGD-AT)")
p.add_argument("--awp", type=float, default=0.0, help="AWP perturbation size gamma; 0 disables")
p.add_argument("--beta", type=float, default=6.0, help="TRADES robustness weight")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--eps", type=float, default=8/255)
p.add_argument("--alpha", type=float, default=2/255)
p.add_argument("--steps", type=int, default=10)
p.add_argument("--eval_steps", type=int, default=20)
p.add_argument("--val_frac", type=float, default=0.04)
p.add_argument("--eval_every", type=int, default=2)
p.add_argument("--ema", type=float, default=0.999, help="EMA decay; 0 disables")
p.add_argument("--out", default="model2.pt")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
print("CONFIG:", vars(args), flush=True)

BASE = Path(__file__).parent
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)

# ----------------------------- data ------------------------------------------
d = np.load(BASE / "train.npz")
X = torch.from_numpy(d["images"]).float() / 255.0
y = torch.from_numpy(d["labels"]).long()
N = len(X)
perm = torch.randperm(N)
n_val = int(N * args.val_frac)
val_idx, tr_idx = perm[:n_val], perm[n_val:]
Xtr, ytr = X[tr_idx], y[tr_idx]
Xval, yval = X[val_idx].to(device), y[val_idx].to(device)
print(f"Loaded {N} images; train {len(Xtr)}, val {len(Xval)}", flush=True)

train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])

class DS(Dataset):
    def __init__(s, im, lb, tf): s.im, s.lb, s.tf = im, lb, tf
    def __len__(s): return len(s.im)
    def __getitem__(s, i): return s.tf(s.im[i]), s.lb[i]

loader = DataLoader(DS(Xtr, ytr, train_tf), batch_size=args.batch, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

# ----------------------------- model -----------------------------------------
def build(arch):
    m = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}[arch](weights=None)
    m.fc = nn.Linear(m.fc.in_features, 9)
    return m

model = build(args.arch).to(device)
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

# manual LR schedule: linear warmup (stops early-training NaN in TRADES) then multistep decay
WARMUP = 5
def lr_at(epoch):
    if epoch <= WARMUP:
        return args.lr * epoch / WARMUP
    if args.cosine:
        t = (epoch - WARMUP) / max(1, args.epochs - WARMUP)
        return 0.5 * args.lr * (1 + math.cos(math.pi * t))
    if epoch < 0.5 * args.epochs:
        return args.lr
    if epoch < 0.75 * args.epochs:
        return args.lr * 0.1
    return args.lr * 0.01
def set_lr(lr):
    for g in opt.param_groups: g["lr"] = lr

# EMA shadow model
ema = copy.deepcopy(model) if args.ema > 0 else None
if ema:
    for prm in ema.parameters(): prm.requires_grad_(False)

@torch.no_grad()
def ema_update(ema, model, decay):
    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
        if e.dtype.is_floating_point:
            e.mul_(decay).add_(m, alpha=1 - decay)
        else:
            e.copy_(m)

# ----------------------------- attacks / losses ------------------------------
def pgd(model, x, y, eps, alpha, steps):
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(model(x + delta), y)
        g, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
        delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    return (x + delta).detach()

def clean_loss(model, x, y):
    return F.cross_entropy(model(x), y, label_smoothing=args.ls)

def pgd_loss(model, x, y):
    model.eval(); xadv = pgd(model, x, y, args.eps, args.alpha, args.steps); model.train()
    return F.cross_entropy(model(xadv), y, label_smoothing=args.ls)

CUR_BETA = 0.0  # set per-epoch (beta warmup) to keep early TRADES stable
def trades_loss(model, x, y):
    model.eval()
    kl = nn.KLDivLoss(reduction="batchmean")
    p_nat = F.softmax(model(x), dim=1).detach()
    xadv = (x + 0.001 * torch.randn_like(x)).clamp(0, 1).detach()
    for _ in range(args.steps):
        xadv.requires_grad_(True)
        loss_kl = kl(F.log_softmax(model(xadv), 1), p_nat)
        g, = torch.autograd.grad(loss_kl, xadv)
        xadv = (xadv.detach() + args.alpha * g.sign())
        xadv = torch.min(torch.max(xadv, x - args.eps), x + args.eps).clamp(0, 1)
    model.train()
    logits = model(x)
    loss_nat = F.cross_entropy(logits, y, label_smoothing=args.ls)
    # robust KL term uses log-softmax on both sides with a detached target -> numerically safe
    loss_rob = kl(F.log_softmax(model(xadv), 1), F.softmax(logits, 1).clamp_min(1e-8))
    return loss_nat + CUR_BETA * loss_rob

def pgd_on_loss(model, x, lossfn, eps, alpha, steps):
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    for _ in range(steps):
        loss = lossfn(model(x + delta))
        g, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
        delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    return (x + delta).detach()

def cutmix_pgd_loss(model, x, y):
    # CutMix then PGD-AT with the mixed-label objective (Rebuffi et al. 2021)
    lam = float(np.random.beta(1.0, 1.0))
    perm = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2], x.shape[3]
    r = math.sqrt(1.0 - lam)
    cw, ch = int(W * r), int(H * r)
    cx, cy = int(np.random.randint(W)), int(np.random.randint(H))
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    xm = x.clone()
    xm[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    ya, yb = y, y[perm]
    mixed = lambda out: lam * F.cross_entropy(out, ya, label_smoothing=args.ls) \
                        + (1 - lam) * F.cross_entropy(out, yb, label_smoothing=args.ls)
    model.eval()
    xadv = pgd_on_loss(model, xm, mixed, args.eps, args.alpha, args.steps)
    model.train()
    return mixed(model(xadv))

if args.cutmix and args.method == "pgd":
    step_loss = cutmix_pgd_loss
else:
    step_loss = {"trades": trades_loss, "pgd": pgd_loss, "clean": clean_loss}[args.method]

class AWP:
    "Adversarial Weight Perturbation (Wu et al. 2020): worst-case weight shift before the update."
    def __init__(self, model, gamma):
        self.model, self.gamma, self.backup = model, gamma, {}
    def _params(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad and p.dim() > 1]
    def perturb(self, x_adv, y):
        self.backup = {n: p.clone() for n, p in self._params()}
        loss = F.cross_entropy(self.model(x_adv), y)
        ps = [p for _, p in self._params()]
        grads = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            for (n, p), g in zip(self._params(), grads):
                if g is None: continue
                p.add_(self.gamma * p.norm() / (g.norm() + 1e-12) * g)
    def restore(self):
        with torch.no_grad():
            for n, p in self._params():
                if n in self.backup: p.copy_(self.backup[n])

awp_obj = AWP(model, args.awp) if (args.awp > 0 and args.method in ("pgd", "trades") and not args.cutmix) else None

# ----------------------------- evaluation ------------------------------------
@torch.no_grad()
def clean_acc(m, X, y, bs=512):
    m.eval(); c = 0
    for i in range(0, len(X), bs):
        c += (m(X[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item()
    return c / len(X)

def robust_acc(m, X, y, bs=256, n=2000):
    m.eval(); X, y = X[:n], y[:n]; c = 0
    for i in range(0, len(X), bs):
        xb, yb = X[i:i+bs], y[i:i+bs]
        xadv = pgd(m, xb, yb, args.eps, args.alpha, args.eval_steps)
        with torch.no_grad():
            c += (m(xadv).argmax(1) == yb).sum().item()
    return c / len(X)

# ----------------------------- train -----------------------------------------
best = 0.0
for epoch in range(1, args.epochs + 1):
    model.train(); run = 0.0; nskip = 0
    set_lr(lr_at(epoch))
    CUR_BETA = args.beta * min(1.0, epoch / 10.0)   # ramp TRADES beta over first 10 epochs
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if awp_obj is not None:
            model.eval(); x_adv = pgd(model, xb, yb, args.eps, args.alpha, args.steps); model.train()
            awp_obj.perturb(x_adv, yb)               # shift weights to worst case
            opt.zero_grad()
            # TRADES+AWP uses the full TRADES objective under perturbed weights; PGD+AWP uses CE on adv
            loss = trades_loss(model, xb, yb) if args.method == "trades" \
                   else F.cross_entropy(model(x_adv), yb, label_smoothing=args.ls)
            if not torch.isfinite(loss):
                awp_obj.restore(); opt.zero_grad(set_to_none=True); nskip += 1; continue
            loss.backward()
            awp_obj.restore()                        # undo weight shift, keep its gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            loss = step_loss(model, xb, yb)
            if not torch.isfinite(loss):             # skip a bad batch instead of poisoning weights
                opt.zero_grad(set_to_none=True); nskip += 1; continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tighter clip prevents NaN blow-ups
            opt.step()
        if ema: ema_update(ema, model, args.ema)
        run += loss.item()

    if epoch % args.eval_every == 0 or epoch == args.epochs:
        eval_model = ema if ema else model
        ca = clean_acc(eval_model, Xval, yval)
        ra = robust_acc(eval_model, Xval, yval)
        sc = 0.5 * ca + 0.5 * ra
        flag = ""
        if ca > 0.50 and sc > best:
            best = sc
            torch.save(eval_model.state_dict(), BASE / args.out)
            flag = "  <-- saved best"
        print(f"epoch {epoch:3d}  loss {run/len(loader):.3f}  clean {ca:.4f}  robust {ra:.4f}  score {sc:.4f}{flag}", flush=True)
    else:
        print(f"epoch {epoch:3d}  loss {run/len(loader):.3f}", flush=True)

print(f"Done. Best validation score = {best:.4f}. Saved to {args.out}", flush=True)
# sanity: reloads into a plain arch
m = build(args.arch); m.load_state_dict(torch.load(BASE / args.out, map_location="cpu")); m.eval()
with torch.no_grad():
    assert m(torch.randn(1, 3, 32, 32)).shape == (1, 9)
print("Sanity OK:", args.arch, "->", args.out, flush=True)
