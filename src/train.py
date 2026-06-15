"""Training / evaluation harness for the Sampling-NN study.

Provides: seed control, a generic train loop with early stopping on val acc,
metrics (accuracy, NLL, ECE, predictive entropy), MC-averaged evaluation,
gradient-variance probing, and robustness-under-input-noise evaluation.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MLP, SmallCNN


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(dataset, sample_pos, sl_kwargs, hidden=256):
    if dataset == "mnist":
        return MLP(in_dim=784, hidden=hidden, n_classes=10, depth=2,
                   sample_pos=sample_pos, **sl_kwargs)
    elif dataset == "cifar10":
        return SmallCNN(in_ch=3, n_classes=10, fc=hidden, **sl_kwargs)
    raise ValueError(dataset)


def iterate(x, y, bs, shuffle, device, gen=None):
    n = len(x)
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield x[j].to(device), y[j].to(device)


@torch.no_grad()
def evaluate(model, x, y, device, bs=1000, mc=1, sample_test=True):
    """Evaluate. If mc>1, average softmax probabilities over mc stochastic passes.
    sample_test toggles whether the SamplingLayer samples at eval time."""
    model.eval()
    model.sampler.sample_test = sample_test
    n = len(x)
    probs_sum = torch.zeros(n, 10, device=device)
    for _ in range(mc):
        ptr = 0
        for xb, yb in iterate(x, y, bs, False, device):
            logits = model(xb)
            probs_sum[ptr:ptr + len(xb)] += F.softmax(logits, dim=1)
            ptr += len(xb)
    probs = probs_sum / mc
    yb_all = y.to(device)
    preds = probs.argmax(1)
    acc = (preds == yb_all).float().mean().item()
    nll = F.nll_loss(torch.log(probs.clamp_min(1e-12)), yb_all).item()
    ece = expected_calibration_error(probs, yb_all)
    ent = (-(probs.clamp_min(1e-12).log() * probs).sum(1)).mean().item()
    return {"acc": acc, "nll": nll, "ece": ece, "entropy": ent}


def expected_calibration_error(probs, labels, n_bins=15):
    conf, preds = probs.max(1)
    acc = (preds == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return ece.item()


def measure_grad_variance(model, x, y, device, n_batches=20, bs=128):
    """Variance of the gradient of the loss wrt the sampler's INPUT-side parameters,
    estimated across stochastic forward passes on the SAME minibatch (isolates sampling
    noise, not data noise)."""
    model.train()
    # pick a parameter feeding the sampler: last backbone linear (or fc1 for CNN)
    if hasattr(model, "linears"):
        target = model.linears[model.sample_pos].weight
    else:
        target = model.fc1.weight
    xb, yb = next(iterate(x, y, bs, True, device,
                          gen=torch.Generator().manual_seed(123)))
    grads = []
    for _ in range(n_batches):
        model.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        g = torch.autograd.grad(loss, target, retain_graph=False)[0]
        grads.append(g.detach().flatten())
    G = torch.stack(grads)  # (n_batches, P)
    # mean per-parameter variance across stochastic passes
    return G.var(dim=0, unbiased=True).mean().item()


def train_model(model, data, device, epochs=15, lr=1e-3, bs=128, val_idx=None,
                tr_idx=None, kl_weight=0.0, patience=5, log=None, seed=0,
                weight_decay=0.0):
    xtr_full, ytr_full, xte, yte = data
    if tr_idx is not None:
        xtr, ytr = xtr_full[tr_idx], ytr_full[tr_idx]
        xval, yval = xtr_full[val_idx], ytr_full[val_idx]
    else:
        xtr, ytr = xtr_full, ytr_full
        xval, yval = xte, yte
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    best_val, best_state, wait = -1, None, 0
    history = []
    for ep in range(epochs):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in iterate(xtr, ytr, bs, True, device, gen=gen):
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if kl_weight > 0:
                loss = loss + kl_weight * model.kl()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(xb)
        train_acc = correct / tot
        val = evaluate(model, xval, yval, device, sample_test=True)
        history.append({"epoch": ep, "train_acc": train_acc,
                        "train_loss": loss_sum / tot, "val_acc": val["acc"]})
        if log:
            log(f"  ep{ep:02d} train_acc {train_acc:.4f} val_acc {val['acc']:.4f}")
        if val["acc"] > best_val:
            best_val = val["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, train_acc
