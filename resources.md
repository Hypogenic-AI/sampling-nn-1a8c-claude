# Resources Catalog — Sampling Neural Network

## Summary
Resources gathered for the project *"sampling from the distribution of activations at
an intermediate neural-network layer, analogous to sampling from the final softmax."*
Includes 13 papers (4 deep-read), 2 datasets, and 1 code repository, plus a
self-contained, reproducible workspace (isolated `uv` venv + `pyproject.toml`).

| Resource type | Count |
|---------------|-------|
| Papers (PDF)  | 13 |
| Datasets      | 2 (MNIST, CIFAR-10) |
| Code repos    | 1 (pytorch/examples) |

---

## Papers
Total downloaded: **13** (all valid PDFs, from arXiv). Detail in `papers/README.md`;
notes in `literature_review.md`.

| Title | Authors | Year | File (papers/) | Key info |
|-------|---------|------|----------------|----------|
| Estimating/Propagating Gradients Through Stochastic Neurons ★ | Bengio et al. | 2013 | `1308.3432_...stochastic_neurons.pdf` | Bernoulli sampling at hidden units; ST / REINFORCE / STS estimators. **Foundational.** |
| Auto-Encoding Variational Bayes (VAE) ★ | Kingma, Welling | 2013 | `1312.6114_...auto_encoding_variational_bayes.pdf` | Reparameterization trick: `z=μ+σ⊙ε` at latent layer. |
| Categorical Reparameterization w/ Gumbel-Softmax ★ | Jang, Gu, Poole | 2016 | `1611.01144_...gumbel_softmax...pdf` | Differentiable **softmax/categorical** sampling at intermediate layers. **Most direct match.** |
| Neural Discrete Representation Learning (VQ-VAE) ★ | van den Oord et al. | 2017 | `1711.00937_...vqvae...pdf` | Discrete codebook + straight-through at intermediate layer. |
| The Concrete Distribution | Maddison et al. | 2016 | `1611.00712_...concrete_distribution.pdf` | Continuous relaxation density; theory behind Gumbel-Softmax. |
| Stochastic Backpropagation (DLGMs) | Rezende et al. | 2014 | `1401.4082_...stochastic_backprop...pdf` | Reparameterized gradients, concurrent with VAE. |
| ProbAct: Probabilistic Activation Function | Lee et al. | 2019 | `1905.10761_...probact...pdf` | Activation output **sampled** from learned (μ,σ) per layer. Closest drop-in. |
| Dropout as a Bayesian Approximation | Gal, Ghahramani | 2016 | `1506.02142_...dropout_bayesian...pdf` | Dropout = activation sampling = approx Bayes; MC-dropout. |
| Variational Dropout + Local Reparameterization | Kingma et al. | 2015 | `1506.02557_...variational_dropout...pdf` | Weight noise → per-activation local noise; learnable rates. |
| Weight Uncertainty (Bayes by Backprop) | Blundell et al. | 2015 | `1505.05424_...bayes_by_backprop.pdf` | Sampling of *weights* (contrast to activations). |
| Noisy Activation Functions | Gülçehre et al. | 2016 | `1603.00391_...noisy_activation_functions.pdf` | Noise injection into activations aids gradient flow. |
| Deep Variational Information Bottleneck | Alemi et al. | 2016 | `1612.00410_...deep_variational_information_bottleneck.pdf` | Stochastic bottleneck; robustness/generalization. |
| Stochastic Activation Pruning | Dhillon et al. | 2018 | `1803.01442_...stochastic_activation_pruning.pdf` | Test-time activation sampling for robustness (see erratum caveat). |

★ = deep-read in full (all chunks). Chunked PDFs in `papers/pages/`.

## Datasets
Total downloaded: **2**. Raw data git-ignored; download instructions + verified
shapes in `datasets/README.md`.

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MNIST | LeCun / CVDF mirror | 70k imgs, ~12 MB | 10-class digit cls | `datasets/mnist/` (IDX .gz) | 60k train / 10k test, 28×28 gray. Verified. |
| CIFAR-10 | Krizhevsky (Toronto) | 60k imgs, ~177 MB | 10-class image cls | `datasets/cifar-10-batches-py/` | 50k train / 10k test, 32×32×3, pickled batches. Verified. |

Verified shapes/classes recorded in `datasets/samples_summary.json`.

## Code Repositories
Total cloned: **1**. Detail in `code/README.md`.

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| pytorch/examples | github.com/pytorch/examples | VAE reference (reparameterization), MNIST classifier baseline | `code/pytorch-examples/` | shallow clone; `vae/main.py` has the canonical `reparameterize()` node to generalize into a `SamplingLayer`. |

Note: PyTorch ships `F.gumbel_softmax`, `torch.distributions`, and the straight-through
1-liner, so no further repos were required.

---

## Resource Gathering Notes

### Search strategy
- **paper-finder** service (localhost:8000) across 4 queries — *intermediate-layer
  activation sampling*, *VAE latent sampling*, *Gumbel-Softmax / discrete latent*,
  *stochastic neurons* — yielding **214 unique candidates** (`paper_search_results/`).
  Note: the diligent mode timed out; fast mode worked reliably.
- **arXiv API** fallback (43 candidates) to recover canonical IDs and ensure
  downloadable PDFs.
- Final 13 selected for relevance (score 3 + foundational status) and downloaded from
  arXiv for reliable PDFs.

### Selection criteria
- Direct match to "sampling at an intermediate layer" (Bengio, ProbAct, Gumbel-Softmax,
  VQ-VAE, VAE).
- Enabling techniques (reparameterization, Concrete, stochastic backprop).
- Mechanistic neighbors that explain *effects* (dropout/Bayesian, VIB, noisy
  activations, SAP) → inform baselines and metrics.
- Preferred foundational + highly-cited works for a solid experimental footing.

### Challenges encountered
- paper-finder **diligent** mode timed out (300s); switched to **fast** mode (success).
- The `arxiv` Python package (v4.0.0) lacks `Result.download_pdf`; downloaded PDFs
  directly via `curl` from `arxiv.org/pdf/<id>.pdf` instead (all 13 succeeded, verified `%PDF`).
- Most paper-finder URLs point to Semantic Scholar (not direct PDFs); mapped key
  titles to arXiv IDs for downloading.

### Gaps and workarounds
- No single canonical VQ-VAE repo cloned (the VQ layer is ~30 lines; objective is in
  the review). PyTorch built-ins cover Gumbel-Softmax and straight-through.
- Datasets kept small/standard by design for fast ablations; scaling datasets
  (ImageNet, audio) documented but not downloaded.

---

## Recommendations for Experiment Design

1. **Primary dataset(s)**: MNIST first (fast, matches foundational papers), then
   CIFAR-10. Both downloaded and verified.
2. **Baseline methods**: deterministic net; dropout / MC-dropout; additive Gaussian
   noise; sampling-off (τ→0) ablation to isolate the *act of sampling*.
3. **Core ablation**: insert a `SamplingLayer` after a hidden layer and compare
   **Gaussian (reparameterization)**, **categorical (Gumbel-Softmax / ST)**, and
   **Bernoulli (straight-through)** sampling — sweeping layer position, temperature /
   noise scale, and train-time vs test-time sampling.
4. **Evaluation metrics**: accuracy / NLL, calibration (ECE), robustness-to-noise,
   generalization gap, gradient variance; report single-sample and MC-averaged.
5. **Code to adapt/reuse**: `code/pytorch-examples/vae/main.py` (`reparameterize`) and
   `mnist/main.py` (baseline); `torch.nn.functional.gumbel_softmax` for the categorical
   case; the straight-through pattern `y = (hard - soft).detach() + soft`.
6. **Cautions**: a soft intermediate sample may be inverted by a strong decoder (use a
   hard/ST variant); anneal τ; guard against dead/indecisive units; add per-layer
   KL-to-prior when the distribution is learned.
