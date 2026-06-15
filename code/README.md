# Cloned Code Repositories

## Repo 1: pytorch/examples

- **URL**: https://github.com/pytorch/examples
- **Location**: `code/pytorch-examples/`
- **Clone**: shallow (`--depth 1`)
- **License**: BSD-3-Clause
- **Purpose**: Clean, minimal, official reference implementations that serve as
  both **baselines** and **scaffolding** for the sampling-layer experiments.

### Most relevant subdirectories

| Path | What it gives us | Use in this project |
|------|------------------|---------------------|
| `vae/main.py` | Full VAE on MNIST: encoder→`reparameterize(mu, logvar)`→decoder, ELBO loss (BCE + KL) | **Canonical reference for the reparameterization trick** — exactly the "sample from a distribution at an intermediate (latent) layer" operation. Directly adaptable: move the sampling node from the bottleneck to an arbitrary hidden layer of a classifier. |
| `mnist/main.py` | Small CNN classifier on MNIST (conv-conv-fc-fc, dropout, NLL loss) | **Deterministic baseline** classifier; insert a stochastic sampling layer between its hidden layers and compare accuracy/calibration. |
| `mnist_forward_forward/` | Forward-Forward training (no backprop) | Alternative credit-assignment reference if sampling breaks differentiability. |
| `imagenet/`, `siamese_network/`, `dcgan/` | Larger training loops | Patterns for scaling to CIFAR-10 / heavier nets. |

### Key reusable snippet (the reparameterization node), from `vae/main.py`
```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std          # <-- sample an activation; gradients flow into mu, logvar
```
For this project the experiment runner can generalize this into a drop-in
`SamplingLayer` placed after any hidden layer:
- **Gaussian** intermediate sampling: layer outputs `(mu, logvar)`, sample `mu + eps*std` (continuous analogue).
- **Categorical / softmax** intermediate sampling: apply `F.gumbel_softmax(logits, tau, hard=...)` (the discrete analogue, directly mirroring final-softmax sampling).
- **Bernoulli** intermediate sampling: straight-through Bernoulli on `sigmoid(a)` (Bengio et al. 2013).

### Installation / requirements
Each example has its own `requirements.txt` (essentially `torch`, `torchvision`).
The experiment runner should install into the existing workspace venv:
```bash
source .venv/bin/activate
uv add torch torchvision        # CPU build is fine for MNIST; GPU build for CIFAR-10
```
The examples auto-download MNIST/CIFAR-10 via torchvision, OR point them at the
already-downloaded raw data in `../datasets/` (see `datasets/README.md`).

### Notes / status
- Not run end-to-end here (torch not yet installed — left to the experiment phase
  to choose the CPU/GPU build). Code was inspected; entry points and the
  reparameterization mechanism are confirmed and documented above.
- `torch.accelerator` API in `vae/main.py` requires a recent PyTorch (>=2.4); if
  an older torch is installed, replace the device block with the standard
  `torch.cuda.is_available()` check.

---

## Note on built-in primitives (no extra repo needed)
PyTorch ships the core sampling ops this project needs, so no further repos were
cloned:
- `torch.nn.functional.gumbel_softmax(logits, tau, hard)` — Gumbel-Softmax / ST Gumbel-Softmax (Jang et al. 2016 / Maddison et al. 2016).
- `torch.randn_like`, `torch.distributions.*` — Gaussian/categorical reparameterized sampling.
- Straight-through estimator is a 1-line custom autograd pattern: `y = (hard - soft).detach() + soft`.

VQ-VAE (van den Oord et al. 2017) has no single canonical repo cloned; if the
experiment runner wants discrete-codebook sampling, the vector-quantization layer
is ~30 lines (nearest-neighbor lookup + straight-through copy + commitment loss);
see `../literature_review.md` for the exact objective.
