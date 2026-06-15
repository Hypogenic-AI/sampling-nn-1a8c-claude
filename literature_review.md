# Literature Review: Sampling Neural Network

**Research hypothesis.** Investigate the effects of *sampling from the distribution
of activations at an intermediate neural-network layer*, analogous to sampling from
the final softmax layer, to understand how this impacts model behavior and performance.

**Prepared:** 2026-06-15 · 13 papers reviewed (4 deep-read in full).

---

## 1. Research Area Overview

The hypothesis sits at the intersection of three established threads:

1. **Stochastic neurons / stochastic activations** — replacing a deterministic
   hidden activation `a` with a *sample* from a distribution parameterized by `a`
   (Bernoulli, Gaussian, or categorical). This is *literally* the project's idea.
2. **Reparameterization & differentiable sampling** — the machinery that makes
   training through a sampling node possible (VAE, Concrete/Gumbel-Softmax, stochastic
   backprop). Without it, sampling at an intermediate layer blocks gradients.
3. **Stochasticity as regularization / Bayesian inference / robustness** — dropout,
   variational dropout, information bottleneck, stochastic activation pruning — which
   show *why* sampling activations changes behavior (uncertainty, generalization,
   robustness, compression).

A key conceptual framing recurs across the literature: **sampling from the final
softmax is categorical sampling**; the project generalizes this operation to
intermediate layers. The literature already provides the exact tools (Gumbel-Softmax
for the categorical case, the reparameterization trick for the continuous case,
straight-through / REINFORCE for the discrete/non-differentiable case) — but no paper
isolates and systematically studies *intermediate-layer activation sampling as the
object of study itself*. That is the gap this project targets.

---

## 2. Key Papers

### 2.1 Foundational mechanism papers (deep-read)

#### Estimating or Propagating Gradients Through Stochastic Neurons (Bengio, Léonard, Courville 2013)
- **Contribution.** The canonical treatment of *sampling at hidden units*. A
  stochastic binary neuron computes `p_i = sigmoid(a_i)` then **Bernoulli-samples**
  `h_i ∈ {0,1}`. Because the sample is non-differentiable, the paper introduces and
  compares **four gradient estimators**:
  - **Straight-Through (ST)**: sample in forward pass, treat threshold as identity in
    backward pass (`∂L/∂a_i ≈ ∂L/∂h_i`). *Best empirically*, simplest, biased.
  - **Unbiased REINFORCE**: `ĝ_i = (h_i − sigmoid(a_i))·L`, with a per-unit
    minimum-variance baseline `L̄_i`. Unbiased, needs no backprop, higher variance.
  - **Stochastic-Times-Smooth (STS)**: `h_i = b_i·√p_i`, `b_i ~ Bernoulli(√p_i)` —
    keeps a differentiable smooth factor while staying sparse.
  - **Noisy rectifier**: additive noise inside a ReLU → stochastic exact-zeros.
- **Results.** On MNIST conditional computation, ST gave the best test error (1.39%);
  all four trained successfully. Noise helped even the *training* objective (exploration).
- **Relevance.** Directly defines "sample from the distribution of an activation at an
  intermediate layer" (the per-neuron Bernoulli analogue of softmax sampling) and
  supplies the menu of gradient estimators the project must choose among. **Default
  recommendation: straight-through.**

#### Auto-Encoding Variational Bayes / VAE (Kingma & Welling 2013)
- **Contribution.** The **reparameterization trick**: draw a latent (intermediate)
  activation as `z = μ(x) + σ(x) ⊙ ε`, `ε ~ N(0, I)`, where `μ, σ` are network
  outputs. Stochasticity is pushed into parameter-free noise `ε`, so the sampling
  node is a deterministic differentiable function → standard backprop works.
  Trained by maximizing the ELBO `= E_q[log p(x|z)] − KL(q(z|x) ‖ p(z))`.
- **Results.** Faster convergence and better likelihood than Wake-Sleep / MCEM on
  MNIST and Frey Faces; the KL term regularizes, so more latent units did not overfit;
  `L=1` sample/datapoint suffices with minibatch ≈ 100.
- **Relevance.** The latent layer **is** an intermediate activation-sampling layer
  (continuous case). Provides the enabling technique and a ready objective structure
  (per-layer KL-to-prior regularizer). Caveat: as stated it covers only *continuous*
  activations — the discrete/softmax analogue needs relaxations (next).

#### Categorical Reparameterization with Gumbel-Softmax (Jang, Gu, Poole 2016/17)
- **Contribution.** Differentiable **categorical (softmax) sampling**. Gumbel-Max:
  `z = one_hot(argmax_i [g_i + log π_i])`, `g_i ~ Gumbel(0,1)`. Relax the argmax to a
  temperature-τ softmax: `y_i = softmax((log π_i + g_i)/τ)`. As τ→0 → true categorical
  (one-hot); large τ → uniform/smooth. **Straight-Through Gumbel-Softmax** gives hard
  one-hot forward, soft backward.
- **Results.** Best among single-sample estimators on structured prediction and VAEs
  (SBN/VAE NLL); 2×–9.9× faster than marginalization in semi-supervised classification
  with matching accuracy.
- **Relevance.** **The most direct match to the hypothesis.** Sampling from a softmax
  *is* categorical sampling; this paper makes that operation differentiable *at
  intermediate layers*. τ is the central knob interpolating deterministic-softmax ↔
  true sampling; anneal τ (cap ~0.5) to trade bias vs gradient variance.

#### Neural Discrete Representation Learning / VQ-VAE (van den Oord et al. 2017)
- **Contribution.** A **discrete codebook** at the intermediate layer: encoder output
  `z_e(x)` is quantized by nearest-neighbor to a codebook entry `z_q(x)=e_k`
  (`k = argmin_j ‖z_e(x)−e_j‖`). Trained with reconstruction + codebook loss
  `‖sg[z_e]−e‖²` + commitment loss `β‖z_e−sg[e]‖²` and a **straight-through** gradient
  copy. A separate autoregressive prior is fit over codes to *sample/generate*.
- **Results.** First discrete-latent model to match continuous VAEs (CIFAR-10 4.67 vs
  4.51 bits/dim); avoids posterior collapse; learns phoneme-like / object-like codes.
- **Relevance.** Concrete, successful instance of a *categorical distribution over an
  intermediate activation*. **Important warning for this project:** a purely *soft*
  intermediate relaxation can be *inverted/ignored* by a powerful downstream decoder —
  hard discretization (or a collapse-prevention mechanism) may be needed. Forcing
  sampling at the intermediate layer pushed representations toward abstract, high-level
  factors.

### 2.2 Supporting / contrasting papers (abstract-level)

| Paper | Year | What it adds | Where stochasticity lives |
|-------|------|--------------|---------------------------|
| **The Concrete Distribution** (Maddison et al.) | 2016 | Independent derivation of the continuous discrete-relaxation density + bias/variance theory underpinning Gumbel-Softmax | Activations (categorical) |
| **Stochastic Backpropagation** (Rezende et al.) | 2014 | Concurrent reparameterized-gradient rules for deep latent Gaussian models | Activations (continuous) |
| **ProbAct** (Lee et al.) | 2019 | A *stochastic activation function*: output sampled from a per-element (mean, variance) distribution at every layer; learnable variance; reports generalization gains | **Activations (every layer)** — closest drop-in to the hypothesis |
| **Dropout as a Bayesian Approximation** (Gal & Ghahramani) | 2016 | Dropout = sampling a Bernoulli mask over activations = approx. Bayesian inference; MC-dropout for uncertainty | Activations (Bernoulli mask) |
| **Variational Dropout + Local Reparameterization** (Kingma et al.) | 2015 | Weight uncertainty → per-activation local noise, variance ∝ 1/batch; learnable rates | Activations (Gaussian, induced) |
| **Weight Uncertainty / Bayes by Backprop** (Blundell et al.) | 2015 | Reparameterized sampling of *weights* instead of activations | Weights (contrast) |
| **Noisy Activation Functions** (Gülçehre et al.) | 2016 | Inject noise into saturating activations so gradients flow; noise aids optimization+exploration | Activations (additive noise) |
| **Deep Variational Information Bottleneck** (Alemi et al.) | 2016 | Stochastic bottleneck trained to compress input while keeping label info; reparameterized sampling; improves robustness/generalization | Activations (bottleneck) |
| **Stochastic Activation Pruning** (Dhillon et al.) | 2018 | *Sample* which activations to keep (prob ∝ magnitude) at test time for adversarial robustness (note: later adversarial re-evaluation reduced its robustness — treat robustness claims cautiously) | Activations (sampled mask) |

---

## 3. Common Methodologies

- **Reparameterization (path-derivative) gradients** — continuous activations:
  `sample = f(params, noise)` with parameter-free noise (VAE, Rezende, Concrete).
  Low variance; the default when activations are continuous.
- **Gumbel-Softmax / Concrete relaxation** — categorical activations: temperature-τ
  softmax over `log π + Gumbel noise`. Anneal τ; ST variant for hard samples.
- **Straight-Through estimator** — non-differentiable (binary/quantized) activations:
  hard forward, identity (or soft) backward. Cheap, biased, empirically strong
  (Bengio 2013, VQ-VAE, ST-Gumbel).
- **Score-function / REINFORCE (+ baselines)** — fully general, unbiased, no backprop
  required, but high variance; needs variance reduction to compete.
- **Auxiliary objectives** — KL-to-prior (VAE/VIB), commitment/codebook loss (VQ-VAE),
  sparsity/firing-rate targets (Bengio), to keep the sampled distribution well-behaved.

## 4. Standard Baselines (for this project)

1. **Deterministic network** (no sampling) — the reference accuracy/calibration.
2. **Dropout / MC-dropout** — the most common "sampling over activations" baseline.
3. **Gaussian-noise injection** at the same layer (additive noise, no learned σ).
4. **Deterministic-softmax / temperature-only** at the chosen layer (sampling off,
   τ→0 limit) — isolates the effect of *sampling* vs the *distributional layer* itself.
5. **VAE-style continuous reparameterized sampling** at the layer (μ, σ heads).
6. **Gumbel-Softmax / ST-Gumbel** categorical sampling at the layer.

## 5. Evaluation Metrics

- **Task performance**: test accuracy / error (MNIST, CIFAR-10); NLL / cross-entropy.
- **Calibration & uncertainty**: ECE, NLL, predictive entropy; behavior under
  MC-averaging of multiple samples at test time.
- **Robustness**: accuracy under input noise / corruptions / (optionally) adversarial
  perturbations — motivated by SAP and VIB.
- **Generalization gap**: train vs test error as a function of *where* and *how much*
  sampling is applied.
- **Representation quality**: linear-probe accuracy on the sampled layer; for discrete
  variants, code-usage / cluster-purity (à la VQ-VAE phoneme mapping).
- **Training dynamics**: gradient variance, convergence speed, sensitivity to τ /
  noise scale / estimator choice.

## 6. Datasets in the Literature

- **MNIST** — used by Bengio 2013, VAE, Gumbel-Softmax (and most stochastic-neuron
  work). Primary benchmark here for fast, comparable iteration. *(downloaded)*
- **CIFAR-10** — used by VQ-VAE; secondary, higher-difficulty benchmark. *(downloaded)*
- Frey Faces, ImageNet, audio (VCTK/LibriSpeech), DeepMind Lab video — used by VAE /
  VQ-VAE; out of scope for an initial study but relevant if scaling.

## 7. Gaps and Opportunities

- **No systematic study of intermediate-layer sampling as the variable of interest.**
  Prior work either samples for a *purpose* (generative modeling, regularization,
  robustness) or at a *fixed* place (latent bottleneck, every layer). A controlled
  sweep over *which layer*, *which distribution* (Bernoulli / Gaussian / categorical),
  *sampling temperature/scale*, and *gradient estimator* — measuring the same metrics
  throughout — is missing and is exactly the hypothesis.
- **Train-time vs test-time sampling** is under-explored as a clean ablation
  (sample only in training? only at test like MC-dropout/SAP? both?).
- **Single-sample vs MC-averaged inference** trade-offs at intermediate layers.
- **The softmax-sampling analogy** is rarely made explicit: applying *categorical
  sampling* to a hidden layer (via Gumbel-Softmax) and comparing to the continuous
  (Gaussian/VAE) and binary (Bengio) cases under one protocol.

## 8. Recommendations for Our Experiment

- **Recommended datasets**: start with **MNIST** (fast, directly comparable to the
  foundational papers), then **CIFAR-10** for a harder test. Both already downloaded.
- **Recommended architecture**: a small MLP and a small CNN (reuse
  `code/pytorch-examples/mnist` and `vae`); insert a drop-in `SamplingLayer` after a
  chosen hidden layer.
- **Recommended sampling variants to compare** (the core ablation):
  1. Continuous **Gaussian** (μ, log σ heads) via reparameterization — VAE-style.
  2. **Categorical / Gumbel-Softmax** (+ ST variant) — the direct softmax analogue.
  3. **Bernoulli** stochastic neurons via straight-through — Bengio-style.
- **Recommended gradient estimators**: **straight-through** (default, cheap),
  **reparameterization** (continuous), with **REINFORCE+baseline** as an unbiased
  reference if variance/bias is in question.
- **Recommended baselines**: deterministic net, dropout/MC-dropout, additive Gaussian
  noise, and the τ→0 / sampling-off ablation (to separate the *distributional layer*
  from the *act of sampling*).
- **Recommended metrics**: accuracy/NLL + calibration (ECE) + robustness-to-noise +
  generalization gap + gradient variance; report single-sample and MC-averaged.
- **Methodological cautions** (from the deep reads):
  - A *soft* intermediate sample can be **inverted/ignored** by a strong downstream
    stack (VQ-VAE) — include a hard/ST variant.
  - **Anneal temperature** (Gumbel-Softmax) and watch the bias↔variance trade-off.
  - Watch for **degenerate units** (dead / indecisive) — use firing-rate/KL or
    commitment-style regularizers (Bengio, VQ-VAE).
  - Add a **per-layer KL-to-prior** term when the sampling distribution is learned, as
    a built-in regularizer (VAE/VIB).
  - Decide and **clearly separate train-time vs test-time** sampling in the protocol.
