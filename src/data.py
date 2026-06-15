"""Data loaders that read the PRE-DOWNLOADED MNIST/CIFAR-10 files directly
(no torchvision download). Returns tensors and split indices.

MNIST: IDX .gz in datasets/mnist/.  CIFAR-10: pickled batches in
datasets/cifar-10-batches-py/.  All normalization stats are computed on the
TRAIN split only (no leakage); val/test are transformed with train stats.
"""
import gzip
import pickle
import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(n * rows * cols)
        return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        return np.frombuffer(f.read(n), dtype=np.uint8)


def load_mnist():
    d = DATA / "mnist"
    xtr = _read_idx_images(d / "train-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    ytr = _read_idx_labels(d / "train-labels-idx1-ubyte.gz").astype(np.int64)
    xte = _read_idx_images(d / "t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    yte = _read_idx_labels(d / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    # normalize with train mean/std (standard MNIST 0.1307/0.3081, but compute to be safe)
    mean, std = xtr.mean(), xtr.std()
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    # flatten for MLP; keep channel dim available via reshape later
    xtr = xtr.reshape(len(xtr), -1)
    xte = xte.reshape(len(xte), -1)
    return (torch.from_numpy(xtr), torch.from_numpy(ytr),
            torch.from_numpy(xte), torch.from_numpy(yte)), {"in_dim": 784, "n_classes": 10, "shape": (1, 28, 28)}


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10():
    d = DATA / "cifar-10-batches-py"
    xs, ys = [], []
    for i in range(1, 6):
        b = _unpickle(d / f"data_batch_{i}")
        xs.append(b[b"data"])
        ys.append(np.array(b[b"labels"]))
    xtr = np.concatenate(xs).astype(np.float32) / 255.0  # (50000, 3072)
    ytr = np.concatenate(ys).astype(np.int64)
    bt = _unpickle(d / "test_batch")
    xte = bt[b"data"].astype(np.float32) / 255.0
    yte = np.array(bt[b"labels"], dtype=np.int64)
    # reshape to (N,3,32,32)
    xtr = xtr.reshape(-1, 3, 32, 32)
    xte = xte.reshape(-1, 3, 32, 32)
    # per-channel normalization from train
    mean = xtr.mean(axis=(0, 2, 3), keepdims=True)
    std = xtr.std(axis=(0, 2, 3), keepdims=True)
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    return (torch.from_numpy(xtr), torch.from_numpy(ytr),
            torch.from_numpy(xte), torch.from_numpy(yte)), {"in_dim": 3072, "n_classes": 10, "shape": (3, 32, 32)}


def make_splits(xtr, ytr, val_frac=0.1, seed=0):
    """Carve a validation split out of train (deterministic given seed)."""
    g = torch.Generator().manual_seed(seed)
    n = len(xtr)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return tr_idx, val_idx


def get_dataset(name):
    if name == "mnist":
        return load_mnist()
    if name == "cifar10":
        return load_cifar10()
    raise ValueError(name)


if __name__ == "__main__":
    for name in ["mnist", "cifar10"]:
        (xtr, ytr, xte, yte), meta = get_dataset(name)
        print(f"{name}: xtr {tuple(xtr.shape)} ytr {tuple(ytr.shape)} "
              f"xte {tuple(xte.shape)} classes {ytr.unique().numel()} "
              f"mean {xtr.mean():.3f} std {xtr.std():.3f} meta {meta}")
