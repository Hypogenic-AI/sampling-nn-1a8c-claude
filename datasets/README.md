# Datasets

This directory contains the datasets for the **Sampling Neural Network** project
(sampling from the distribution of activations at an intermediate layer). Raw data
files are **NOT committed to git** (see `.gitignore`); follow the download
instructions below to reproduce them. A small `samples_summary.json` documenting
shapes/classes is committed for reference.

Both datasets are deliberately small, standard image-classification benchmarks.
They let the experiment runner train a compact classifier, insert a stochastic
"sampling layer" at an intermediate point, and measure the effect on
accuracy / calibration / robustness with fast iteration.

---

## Dataset 1: MNIST

### Overview
- **Source**: http://yann.lecun.com/exdb/mnist/ (mirror used: Google CVDF mirror)
- **Size**: 70,000 grayscale images, 28×28, ~12 MB total (gzipped IDX)
- **Format**: IDX ubyte (gzipped)
- **Task**: 10-class digit classification
- **Splits**: train = 60,000, test = 10,000
- **Pixel range**: 0–255 (uint8); normalize to [0,1] or standardize before use
- **License**: MNIST terms (free for research)

### Download Instructions
```bash
mkdir -p datasets/mnist && cd datasets/mnist
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sL "https://storage.googleapis.com/cvdf-datasets/mnist/$f" -o "$f"
done
cd -
```
Alternatively (recommended for experiments, auto-downloads):
```python
from torchvision import datasets, transforms
train = datasets.MNIST("datasets/mnist_tv", train=True,  download=True, transform=transforms.ToTensor())
test  = datasets.MNIST("datasets/mnist_tv", train=False, download=True, transform=transforms.ToTensor())
```

### Loading the raw IDX files (no torchvision)
```python
import gzip, struct, numpy as np
def read_images(path):
    with gzip.open(path,'rb') as f:
        _, n, r, c = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(n*r*c), np.uint8).reshape(n, r, c)
def read_labels(path):
    with gzip.open(path,'rb') as f:
        _, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(n), np.uint8)
X = read_images("datasets/mnist/train-images-idx3-ubyte.gz")   # (60000,28,28)
y = read_labels("datasets/mnist/train-labels-idx1-ubyte.gz")   # (60000,)
```

---

## Dataset 2: CIFAR-10

### Overview
- **Source**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Size**: 60,000 color images, 32×32×3, ~163 MB (tar.gz) / ~177 MB extracted
- **Format**: Python pickled batches (`data_batch_1..5`, `test_batch`)
- **Task**: 10-class natural-image classification
- **Splits**: train = 50,000 (5 batches × 10,000), test = 10,000
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Pixel range**: 0–255 (uint8); each image is a 3072-vector = 1024 R + 1024 G + 1024 B (row-major)
- **License**: MIT-style / free for research (Krizhevsky 2009)

### Download Instructions
```bash
cd datasets
curl -sL https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz   # -> datasets/cifar-10-batches-py/
cd -
```
Alternatively (recommended for experiments):
```python
from torchvision import datasets, transforms
train = datasets.CIFAR10("datasets/cifar_tv", train=True,  download=True, transform=transforms.ToTensor())
test  = datasets.CIFAR10("datasets/cifar_tv", train=False, download=True, transform=transforms.ToTensor())
```

### Loading the raw pickled batches (no torchvision)
```python
import pickle, numpy as np
def load_batch(path):
    with open(path,'rb') as f:
        d = pickle.load(f, encoding='bytes')
    X = d[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)  # (N,32,32,3)
    y = np.array(d[b'labels'])
    return X, y
Xtr, ytr = load_batch("datasets/cifar-10-batches-py/data_batch_1")  # repeat for 1..5
Xte, yte = load_batch("datasets/cifar-10-batches-py/test_batch")
```

---

## Why these datasets for this project
- **Fast iteration**: MLP/small-CNN training on MNIST runs in minutes on CPU/GPU,
  enabling many ablations over *where* to sample, *which distribution* to sample
  from, and *which gradient estimator* to use.
- **Difficulty gradient**: CIFAR-10 adds enough complexity (natural images, deeper
  nets) to test whether intermediate-layer sampling helps or hurts at higher
  capacity, and to measure regularization/robustness effects.
- **Direct precedent**: MNIST is the benchmark used in the foundational
  stochastic-neuron / VAE / Gumbel-Softmax papers (see `../literature_review.md`),
  so results are directly comparable to prior work; CIFAR-10 is used in VQ-VAE.

## Suggested preprocessing
- Normalize MNIST to [0,1] (`/255`) or standardize per-pixel.
- For CIFAR-10, per-channel normalize with mean `(0.4914,0.4822,0.4465)`,
  std `(0.2470,0.2435,0.2616)`; optional augmentation (random crop+flip).

See `samples_summary.json` for verified shapes/classes.
