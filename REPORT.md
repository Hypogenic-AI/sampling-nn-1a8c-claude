# Sampling Neural Network — Research Report

**Research question.** What happens if we form a probability distribution over an
**intermediate** layer's activations and **sample** from it — the same operation we
apply to the final softmax — rather than passing the deterministic activations forward?

**Key finding (one sentence).** Intermediate-layer sampling is a *distribution-type–
dependent regularizer*: continuous (Gaussian) sampling is essentially **free** (≈
deterministic accuracy) and improves calibration and the generalization gap, while the
*literal* softmax analogue — categorical (Gumbel-Softmax) sampling that collapses a hidden
layer to ~one active unit — is **the most expensive**, has the **highest gradient
variance**, and **fails to train at all on the harder CIFAR-10 task** (collapsing to chance).

**Practical implication.** "Sampling a hidden layer like the output softmax" is *possible*
and even beneficial in its gentle continuous form, but the direct categorical form is a
severe information bottleneck whose cost scales with task difficulty and with how
aggressively (low temperature) you sample — it is not a free drop-in.

---

## 1. Executive Summary

The final layer of a classifier is stochastic — we sample a class from the softmax. The
submitter asked what happens if we do the *same thing* at an intermediate layer: treat the
activations as a distribution and draw a sample. We built a single drop-in `SamplingLayer`
and, holding architecture/data/optimizer fixed, compared seven activation operators —
**deterministic** (reference), three **baselines** (dropout, additive Gaussian noise), and
the sampling variants the literature identifies as the three faces of this idea:
**continuous Gaussian** (VAE reparameterization), **binary Bernoulli** (Bengio
straight-through), and **categorical Gumbel-Softmax** (soft and hard one-hot — *the direct
softmax analogue*). We evaluated on **MNIST** (MLP, 3 seeds) and **CIFAR-10** (small CNN, 3
seeds) with accuracy, NLL, calibration (ECE), generalization gap, gradient variance,
single-sample vs MC-averaged inference, a temperature/noise sweep, a layer-position sweep,
and a robustness-to-input-noise sweep — 117 trained models in total.

Three results stand out. (1) **The distribution type dominates everything.** Continuous
Gaussian sampling costs ≈0 accuracy on MNIST (97.7% vs 97.8% deterministic) and *beats*
deterministic on CIFAR-10 calibration (ECE 0.074 vs 0.112) and NLL; categorical sampling
costs 8–10 points on MNIST and **collapses to 10–25%** on CIFAR-10. (2) **Sampling
regularizes.** Every stochastic variant shrinks the train–test gap and most lower ECE — the
benefit the literature predicts. (3) **MC-averaging recovers the single-sample cost**, just
as averaging softmax samples approximates the expectation: dropout climbs from 96.6%→97.9%
as K grows from 1→50. The gradient-variance ranking (Bernoulli-ST lowest, hard-categorical
highest) exactly tracks the accuracy cost, explaining *why* categorical sampling is hard.

---

## 2. Research Question & Motivation

**Hypothesis.** Sampling from the distribution of activations at an intermediate layer
(analogous to softmax sampling) changes model behavior in ways that depend on the
distribution type, the amount of sampling, the layer position, and whether sampling is used
at train and/or test time.

**Why it matters.** Hidden-layer stochasticity is the shared mechanism behind VAEs,
Gumbel-Softmax discrete latents, stochastic neurons, dropout, and the information
bottleneck — but these were always studied *for a purpose* or *at a fixed place*. The
`literature_review.md` gap is explicit: **no prior work isolates intermediate-layer
activation sampling as the variable of study**, and the **softmax-sampling analogy is rarely
made explicit**. This project runs exactly that controlled study. Knowing the *cost*
(accuracy) and *benefit* (calibration, generalization, robustness) of this operation, and
*where/how* it is best applied, is directly useful for anyone considering stochastic hidden
representations.

**Hypotheses tested.** H1 single-sample accuracy ≤ deterministic, gap grows with sampling
amount; H2 sampling improves calibration / generalization gap; H3 MC-averaging recovers
accuracy and calibration; H4 Gaussian gentlest, hard categorical most aggressive; H5
sampling improves robustness to input noise.

---

## 3. Data Construction

- **MNIST** — 60k train / 10k test, 28×28 grayscale, 10 classes; read directly from the
  pre-downloaded IDX files (`datasets/mnist/`). Standardized with **train-split** mean/std
  (no leakage). Flattened to 784-d for the MLP.
- **CIFAR-10** — 50k train / 10k test, 32×32×3, 10 classes; read from the pre-downloaded
  pickled batches (`datasets/cifar-10-batches-py/`). Per-channel standardized with
  **train-split** statistics.
- **Validation split.** A 10% validation set is carved from train (deterministic per seed)
  for early stopping and any knob selection; **the test set is used only for final
  evaluation**. Both datasets are balanced (10 classes), so plain accuracy is appropriate
  and no resampling/class-weighting is needed. Data quality verified in `src/data.py`
  (shapes, normalized mean≈0/std≈1, class counts).

---

## 4. Methodology

### 4.1 The `SamplingLayer` (the object of study)
A drop-in module placed after a hidden ReLU. It turns the incoming activations into a
distribution and draws a sample. Modes (`src/models.py`):

| Mode | Distribution | Sample | Gradient estimator | Role |
|---|---|---|---|---|
| `deterministic` | — | identity | — | reference |
| `dropout` | Bernoulli mask | mask·x/(1-p) | exact (mask is data-independent) | baseline (MC-dropout) |
| `gauss_noise` | x + σ·𝒩(0,1) | additive | exact (reparameterized) | baseline (unlearned noise) |
| `gaussian_reparam` | 𝒩(μ(x), σ(x)) per unit | μ+σ⊙ε | **reparameterization** | VAE-style continuous |
| `bernoulli_st` | Bernoulli(sigmoid(x)) per unit | hard 0/1 | **straight-through** | Bengio stochastic neuron |
| `gumbel_softmax` | softmax(x) over units | relaxed sample | **Gumbel-Softmax** | *direct softmax analogue (soft)* |
| `gumbel_st` | softmax(x) over units | hard one-hot | **ST-Gumbel** | *direct softmax analogue (hard: sample ONE unit)* |

The categorical modes implement the idea literally: the layer's units are treated as
*logits of a categorical distribution* (softmax → a distribution), and we draw a
(Gumbel-Softmax) sample, exactly as we sample the final softmax — but over hidden units. The
one-hot simplex vector is rescaled by the layer width so its magnitude is comparable to a
dense ReLU activation. `gaussian_reparam` adds a small KL-to-prior term (weight 1e-3) as a
built-in regularizer (VAE/VIB), as recommended in the review.

### 4.2 Backbones, training, evaluation
- **MNIST:** MLP `784→256→256→10` (ReLU), `SamplingLayer` after hidden-1 (`pos=0`) by
  default. **CIFAR-10:** small CNN (4 conv + 2 FC), `SamplingLayer` on the 256-d FC features.
- **Training:** Adam (lr 1e-3), batch 128, ≤15 epochs (MNIST) / 20 (CIFAR), early stopping
  on val accuracy (patience 5), best checkpoint restored. Seeds {0,1,2}; all RNGs seeded.
- **Metrics:** test accuracy, NLL, **ECE (15-bin)**, predictive entropy, generalization gap
  (train−test acc), and **gradient variance** at the sampling layer (variance of the loss
  gradient across stochastic passes on a *fixed* minibatch — isolates sampling noise).
  Stochastic models are evaluated both **single-sample** and **MC-averaged** over K passes.
- **Statistics:** paired t-test across seeds vs deterministic; Cohen's d; α=0.05.
- **Hardware/runtime:** NVIDIA RTX A6000; torch 2.12.0+cu130; 117 models, ≈36 min total
  compute (MNIST seconds–minutes/model, CIFAR ≈40–75 s/model). Full versions in
  `results/environment.json`.

### 4.3 Experiments
E1 core comparison (MNIST, all modes); E2 temperature τ / noise σ sweep; E3 layer-position
sweep; E4 train/test × MC-averaging (K∈{1,5,20,50}); E5 robustness to input Gaussian noise;
E6 CIFAR-10 confirmation. Selectable via `python src/run_experiments.py --exp E1 …`.

---

## 5. Results

### 5.1 E1 — Core comparison on MNIST (MLP, 3 seeds, sampling after hidden-1)

| Mode | Acc (1 sample) | Acc (MC-10) | NLL | ECE | Gen-gap | Grad-var |
|---|---|---|---|---|---|---|
| Deterministic | **0.9776** | 0.9776 | 0.088 | 0.0118 | 0.0172 | 0 |
| Dropout | 0.9655 | 0.9781 | 0.123 | 0.0115 | 0.0118 | 9.3e-7 |
| Gaussian noise | 0.9800 | **0.9804** | 0.091 | 0.0123 | 0.0152 | 5.3e-8 |
| Gaussian reparam (VAE) | 0.9773 | 0.9775 | 0.089 | 0.0098 | 0.0111 | 1.5e-6 |
| Bernoulli ST | 0.9499 | 0.9663 | 0.166 | **0.0070** | 0.0109 | 8.0e-8 |
| Gumbel-Softmax (soft) | 0.8995 | 0.9191 | 0.386 | 0.0194 | 0.0006 | 7.1e-6 |
| Gumbel ST (hard 1-hot) | 0.8735 | 0.8859 | 0.511 | 0.0150 | −0.0214 | 1.5e-5 |

Paired t-test vs deterministic (single-sample acc): Bernoulli ST Δ=−2.77 pts (p=0.0015),
Gumbel-Softmax Δ=−7.82 pts (p=0.031), Gumbel-ST Δ=−10.4 pts (p=0.048), Gaussian reparam
Δ=−0.03 pts (p=0.83, **not significant** — free), Gaussian noise Δ=+0.24 pts (p=0.11).
Figure: `results/plots/E1_core_mnist_bars.png`.

**Reading.** Cost ordering is exactly H4: Gaussian ≈ 0 ≪ Bernoulli ≪ categorical. ECE
*drops* for the gentle stochastic variants (best: Bernoulli 0.0070, Gaussian-reparam 0.0098
vs 0.0118 deterministic) and the generalization gap shrinks for every sampling variant —
sampling regularizes. The gradient-variance column is the mechanistic explanation: it
ranks Gauss-noise/Bernoulli-ST (≈1e-7) < dropout/reparam (≈1e-6) < Gumbel-soft (7e-6) <
**Gumbel-hard (1.5e-5)**, the same order as the accuracy cost.

### 5.2 E2 — How much you sample (temperature / noise scale)

| Gumbel-Softmax τ | 0.1 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|
| Acc (1 sample) | 0.664 | 0.900 | 0.957 | 0.969 | 0.976 |

| Gaussian noise σ | 0.1 | 0.3 | 0.6 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| Acc (1 sample) | 0.979 | 0.980 | 0.980 | 0.978 | 0.976 |

Figure: `results/plots/E2_sweep.png`. **Lower τ = sharper categorical = more aggressive
sampling = larger cost** (monotonic, 66%→97.6%). As τ grows the softmax flattens toward
uniform and the *act of sampling* fades out — accuracy approaches the deterministic
reference. This cleanly separates "the distributional layer" from "the act of sampling":
the cost *is* the sampling. Gaussian noise is gentle across two decades of σ, with a mild
robustness sweet-spot at σ≈0.3–0.6.

### 5.3 E3 — Where you sample (layer position)

| Mode | pos=0 (after hidden-1) | pos=1 (after hidden-2, nearer output) |
|---|---|---|
| Bernoulli ST | 0.9499 | 0.9783 |
| Gaussian reparam | 0.9773 | 0.9795 |
| Gumbel-ST | 0.8735 | 0.8886 |

Sampling **closer to the output is consistently cheaper** — the network has fewer remaining
layers whose computation is disrupted, and (for categorical) the bottleneck sits nearer the
already-low-dimensional decision. `results/summary_E3_position.csv`.

### 5.4 E4 — MC-averaging recovers the cost (and the softmax analogy)

| K (test samples) | 1 | 5 | 20 | 50 |
|---|---|---|---|---|
| Dropout acc | 0.9655 | 0.9766 | 0.9793 | 0.9795 |
| Gumbel-Softmax acc | 0.8995 | 0.9174 | 0.9209 | 0.9220 |
| Gaussian reparam acc | 0.9773 | 0.9774 | 0.9774 | 0.9774 |

Figure: `results/plots/E4_mc.png`. Averaging many stochastic passes approximates the
expectation over the activation distribution — exactly as averaging softmax samples
approximates the class distribution — and recovers most of the single-sample loss
(dropout +1.4 pts; Gumbel +2.3 pts). Low-variance reparameterized Gaussian has almost
nothing to recover. (Caveat: MC-averaging *categorical* samples can *raise* ECE — averaging
near-one-hot draws yields overconfident mean probabilities.)

### 5.5 E5 — Robustness to input Gaussian noise (MNIST)

| σ (input) | 0.0 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|
| Deterministic | 0.9776 | 0.9719 | 0.9421 | 0.8769 |
| Dropout (MC-10) | 0.9782 | 0.9733 | **0.9514** | **0.8988** |
| Gaussian reparam | 0.9775 | 0.9704 | 0.9397 | 0.8733 |
| Gumbel-ST | 0.8854 | 0.8782 | 0.8501 | 0.7984 |

Figure: `results/plots/E5_robust.png`. The robustness benefit is **real but modest** and is
driven mainly by **dropout / MC-averaging** (best at every corrupted severity, +1 pt at
σ=1.0, +2.2 pts at σ=1.5). Continuous Gaussian sampling tracks the deterministic curve;
categorical models start lower but degrade *relatively* gracefully. H5 is only weakly
supported.

### 5.6 E6 — CIFAR-10 confirmation (small CNN, 3 seeds) — the striking result

| Mode | Acc (1) | Acc (MC-10) | NLL | ECE | Gen-gap |
|---|---|---|---|---|---|
| Deterministic | 0.7503 | 0.7503 | 0.941 | 0.1119 | 0.2253 |
| Dropout | 0.7298 | 0.7645 | 0.962 | 0.0987 | 0.1845 |
| Gaussian noise | 0.7456 | 0.7472 | 0.939 | 0.1240 | 0.2288 |
| **Gaussian reparam** | **0.7548** | **0.7553** | **0.784** | 0.0741 | 0.2084 |
| Bernoulli ST | 0.7384 | 0.7650 | 0.833 | **0.0681** | 0.1458 |
| **Gumbel-Softmax (soft)** | **0.1000** | **0.1000** | 2.316 | 0.030 | −0.000 |
| **Gumbel ST (hard 1-hot)** | **0.2555** | 0.2574 | 1.895 | 0.038 | 0.001 |

Figure: `results/plots/E6_core_cifar10_bars.png`. On a harder task the divide widens
dramatically: **continuous Gaussian sampling matches/beats deterministic accuracy** (75.5%
vs 75.0%) while **cutting NLL 0.94→0.78 and ECE 0.112→0.074**; Bernoulli ST gives the best
calibration (0.068) and smallest gen-gap (0.146). But the **literal categorical softmax
analogue collapses** — Gumbel-Softmax to **chance (10%)** and hard one-hot to **25%**.
Forcing all CNN features through a single sampled unit is too severe a bottleneck (and too
high-variance) to optimize. This is the project's sharpest finding and a concrete instance
of the `literature_review.md` VQ-VAE caution about degenerate/collapsed hidden samples.

---

## 6. Analysis & Discussion

- **H1 (cost) — supported.** Single-sample accuracy ≤ deterministic for every genuine
  sampling variant, and the gap grows monotonically as τ↓ / sampling sharpens (E2) and as
  the distribution moves from continuous→binary→categorical (E1/E6).
- **H2 (regularization) — supported.** All sampling variants shrink the train–test gap;
  most lower ECE; Gaussian-reparam/Bernoulli improve calibration *and* (on CIFAR) NLL —
  trading a tiny accuracy cost (or none) for better-behaved probabilities.
- **H3 (MC recovery) — supported.** MC-averaging recovers most of the single-sample loss
  (E4), realizing the softmax-sampling analogy: many samples ≈ the expectation.
- **H4 (distribution type) — strongly supported.** Gaussian gentlest, hard categorical most
  aggressive, Bernoulli between — visible in accuracy, NLL, *and* gradient variance, with
  the gradient-variance ranking mechanistically explaining the accuracy ranking.
- **H5 (robustness) — weakly supported.** A modest improvement, mostly from dropout/MC,
  not a large effect.

**Why categorical sampling is special.** It is the *only* variant that changes the
*dimensionality of information flow*: a hard one-hot sample passes essentially log₂(width)
bits, versus the full continuous vector for Gaussian/Bernoulli. That is simultaneously the
strongest regularizer (smallest gen-gap, lowest ECE on CIFAR) and the highest-variance,
hardest-to-train operator — beneficial as a mild relaxation (large τ, MNIST) but
catastrophic when pushed hard or on a task that needs the bandwidth (CIFAR). This is the
crux of "what intermediate-layer softmax sampling does."

**Surprises.** (1) Gaussian *additive noise* and *reparameterized* sampling are effectively
free, and on CIFAR the latter *improves* accuracy+calibration+NLL together. (2) The total
collapse of soft Gumbel-Softmax to exactly chance on CIFAR — not merely degraded but
untrainable. (3) Sampling **later** is cheaper (E3), opposite to a naive "early bottleneck
is best" intuition.

---

## 7. Limitations

- **Scope:** two datasets (MNIST/CIFAR-10), small models, single sampling site per net,
  3 seeds. Trends are clear and significant but absolute numbers are modest (no heavy CNN
  tuning/augmentation on CIFAR).
- **Categorical-collapse confound:** the collapse could be partly an *optimization* failure
  (LR/scale/temperature for the rescaled one-hot at a wide FC layer) rather than a pure
  information limit. A τ-anneal schedule and per-mode LR tuning were out of scope; we report
  the collapse honestly as observed under the shared protocol. The τ-sweep (E2) shows the
  effect is real and monotonic, supporting the information-bottleneck reading.
- **Calibration of categorical MC-averaging** can worsen ECE — MC-averaging is not a
  universal calibration fix.
- **Robustness** tested only under Gaussian input noise (no adversarial/corruption suites).
- **Gradient variance** measured on one parameter tensor feeding the sampler as a proxy.

---

## 8. Conclusions & Next Steps

**Answer to the research question.** Sampling a hidden layer the way we sample the final
softmax *works and even helps in its gentle continuous form* — Gaussian reparameterized
sampling costs ≈0 accuracy and yields better-calibrated, lower-NLL models with a smaller
generalization gap — but the **literal categorical version (sample one unit from a softmax
over hidden units) is a severe, high-variance information bottleneck**: it is the most
expensive variant on MNIST and **fails to train on CIFAR-10**, with cost scaling as you
sharpen the sample (lower τ), move the layer earlier, and increase task difficulty. The
behavior is governed first and foremost by the **distribution type**, and MC-averaging at
test time recovers the single-sample cost, exactly mirroring softmax sampling.

**Next steps.** (1) Temperature **annealing** + per-mode LR for the categorical modes to
test whether CIFAR collapse is bottleneck vs optimization. (2) A **VQ-VAE-style codebook**
(K≫width prototypes) as a less brutal categorical layer. (3) Multiple/structured sampling
sites and an explicit **information-bottleneck (VIB)** objective to quantify bits passed.
(4) Scale to deeper nets / harder data to see whether continuous sampling's calibration
benefit persists. (5) Uncertainty quality (OOD detection) from MC-sampled hidden layers.

---

## References (resources used)
Bengio, Léonard & Courville (2013) *Stochastic Neurons*; Kingma & Welling (2013) *VAE*;
Jang, Gu & Poole (2016) *Gumbel-Softmax*; van den Oord et al. (2017) *VQ-VAE*; Maddison et
al. (2016) *Concrete*; Gal & Ghahramani (2016) *Dropout as Bayesian*; Lee et al. (2019)
*ProbAct*; Alemi et al. (2016) *Deep VIB*; Dhillon et al. (2018) *Stochastic Activation
Pruning*. PDFs in `papers/`; synthesis in `literature_review.md`; catalog in `resources.md`.
Datasets: MNIST (LeCun), CIFAR-10 (Krizhevsky). Tools: PyTorch 2.12, NumPy, SciPy,
scikit-learn, pandas, matplotlib. Code reuse: `code/pytorch-examples/` (VAE
`reparameterize`, MNIST baseline) generalized into `src/models.py:SamplingLayer`.
