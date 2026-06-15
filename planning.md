# Planning — Sampling Neural Network

## Motivation & Novelty Assessment

### Why This Research Matters
Modern networks are deterministic in their hidden layers but *stochastic at the very
end*: the final softmax defines a categorical distribution over classes and we
**sample** from it (or take its argmax). The submitter asks a sharp question: what if
we apply that *same operation* — form a distribution over a layer's activations and
sample a single realization — at an **intermediate** layer? Understanding this is
valuable because (a) it unifies a scattered literature (VAE latent sampling,
Gumbel-Softmax, stochastic neurons, dropout) under one controlled lens; (b) hidden-layer
sampling is the mechanism behind uncertainty estimation, regularization, and discrete
representation learning; and (c) practitioners need to know the *cost* (does sampling
hurt accuracy?) and the *benefit* (calibration, robustness, generalization) of injecting
this stochasticity, and *where* it is best placed.

### Gap in Existing Work
From `literature_review.md`: prior work either samples for a **purpose** (VAE generation,
VIB compression, MC-dropout uncertainty, SAP robustness) or at a **fixed place** (the
latent bottleneck, or every layer as in ProbAct). **No paper isolates intermediate-layer
activation sampling as the variable of study itself** and runs a controlled sweep over
*which distribution*, *how much* sampling, *where*, and *train- vs test-time* under one
protocol. Crucially, the **softmax-sampling analogy is rarely made explicit**: treating a
hidden layer exactly like the output layer — softmax over its units → a categorical
distribution → sample one — is the most literal reading of the idea and is essentially
unstudied as such.

### Our Novel Contribution
A single controlled framework — a drop-in `SamplingLayer` — under which we compare, on the
same architecture/data/metrics:
1. **Categorical / Gumbel-Softmax sampling** over a hidden layer's units (the *direct*
   softmax analogue — the centerpiece, reading the idea literally).
2. **Per-unit Gaussian** sampling (VAE-style reparameterization).
3. **Per-unit Bernoulli** sampling (Bengio straight-through stochastic neurons).
against deterministic, dropout, and additive-Gaussian-noise baselines — measuring
accuracy, calibration (ECE/NLL), robustness-to-noise, generalization gap, and gradient
variance, with **train-time vs test-time** and **single-sample vs MC-averaged** ablations.

### Experiment Justification
- **E1 — Distribution-type comparison (core).** Insert `SamplingLayer` at a fixed mid
  layer; compare all sampling variants + baselines on MNIST, multiple seeds. *Why:* the
  central question — what does each kind of intermediate sampling do to behavior.
- **E2 — Temperature / noise-scale sweep.** Vary τ (Gumbel) and σ (Gaussian) from
  near-deterministic to strong sampling. *Why:* isolates the *amount* of sampling and the
  bias↔variance / determinism↔stochasticity trade-off; locates the τ→0 "sampling-off"
  limit that separates the *distributional layer* from the *act of sampling*.
- **E3 — Layer-position sweep.** Move the categorical `SamplingLayer` to early/mid/late
  positions. *Why:* tests the hypothesis that *where* you sample matters (info bottleneck
  vs near-output collapse).
- **E4 — Train-time vs test-time × MC-averaging.** Toggle sampling on/off at train and
  test; average K samples at inference. *Why:* clean ablation of when stochasticity helps,
  and whether MC-averaging recovers accuracy and improves calibration/uncertainty.
- **E5 — Robustness & calibration under input corruption.** Evaluate best variants under
  Gaussian input noise at increasing severity. *Why:* tests the literature's claim that
  activation sampling buys robustness/uncertainty (VIB/SAP/MC-dropout).
- **E6 — CIFAR-10 confirmation (secondary, small CNN).** Re-run the core comparison on a
  harder dataset. *Why:* external validity beyond MNIST.

---

## Research Question
Does forming a probability distribution over an intermediate layer's activations and
**sampling** from it (as we do at the final softmax) change model behavior — and how does
the effect depend on the **distribution type** (categorical/Gaussian/Bernoulli), the
**amount** of sampling (τ/σ), the **layer position**, and **train- vs test-time** use —
measured by accuracy, calibration, robustness, generalization, and gradient variance?

## Background and Motivation
See "Motivation & Novelty" above. The enabling machinery exists (reparameterization,
Gumbel-Softmax/Concrete, straight-through), so the contribution is the *controlled study*,
not a new estimator.

## Hypothesis Decomposition
- **H1 (cost).** Intermediate sampling adds noise to the forward pass; single-sample test
  accuracy will be ≤ deterministic, with the gap growing as τ/σ grows and as the layer
  moves toward the output.
- **H2 (calibration/uncertainty).** Train-time sampling acts as a regularizer; sampling
  variants improve calibration (lower ECE) and/or generalization gap vs deterministic.
- **H3 (MC recovery).** MC-averaging many samples at test time recovers most of the
  single-sample accuracy loss and improves calibration — the same way averaging softmax
  samples approximates the expectation.
- **H4 (distribution type).** Continuous Gaussian sampling is the gentlest (lowest cost);
  hard categorical one-hot sampling is the most aggressive (highest cost, strongest
  bottleneck/representation effect), with Bernoulli in between.
- **H5 (robustness).** Models trained with intermediate sampling degrade more gracefully
  under input noise than the deterministic baseline.

Independent variables: distribution type, τ/σ, layer position, train/test sampling flag,
K (MC samples). Dependent variables: test acc, NLL, ECE, robust acc, train−test gap,
gradient variance.

## Proposed Methodology

### Approach
A drop-in `SamplingLayer(mode, ...)` inserted after a chosen hidden layer of a fixed
backbone (MLP for MNIST, small CNN for CIFAR-10). `mode ∈ {deterministic, dropout,
gauss_noise, gaussian_reparam, bernoulli_st, gumbel_softmax, gumbel_st}`. Everything else
(optimizer, schedule, width, depth) held constant so the sampling operator is the only
variable. The categorical modes implement the literal softmax analogue: `softmax` over the
layer's units (a distribution), then a Gumbel-Softmax sample (soft) or straight-through
one-hot (hard), scaled back so magnitude is comparable.

### Experimental Steps
1. Implement data loaders for the pre-downloaded MNIST/CIFAR (no torchvision download).
2. Implement `SamplingLayer` + backbones + train/eval harness with seeds, logging,
   checkpointing, gradient-variance probing.
3. E1 core comparison (MNIST, ≥3 seeds) → results table + plots.
4. E2 τ/σ sweep; E3 layer-position sweep; E4 train/test × MC; E5 robustness; E6 CIFAR-10.
5. Statistical tests (paired t-test across seeds), calibration curves, error analysis.

### Baselines
Deterministic net (reference); Dropout / MC-dropout (canonical activation sampling);
additive Gaussian noise (unlearned); "sampling-off" τ→0 / σ→0 limit (isolates the act of
sampling from the distributional layer).

### Evaluation Metrics
Test accuracy & error; NLL/cross-entropy; **ECE** (15-bin) for calibration; predictive
entropy; robust accuracy under input Gaussian noise (severity sweep); generalization gap
(train−test acc); gradient variance at the sampling layer. Single-sample **and**
MC-averaged (K samples) reported for stochastic models.

### Statistical Analysis Plan
Each config run with seeds {0,1,2} (more for the core table if time permits). Report
mean ± std. Compare each sampling variant to deterministic with a **paired t-test** across
seeds; significance α=0.05; also report effect size (Cohen's d) and 95% CIs. No test-set
peeking for tuning — τ/σ chosen on a validation split.

## Expected Outcomes
Supportive: stochastic variants trade a small single-sample accuracy cost for better
calibration / smaller generalization gap / better noise robustness; MC-averaging recovers
accuracy; effects scale with τ/σ and depend on layer position and distribution type.
Refuting: sampling only hurts with no compensating benefit on any metric, or is
indistinguishable from deterministic (e.g. downstream layers invert/ignore the noise — the
VQ-VAE caution).

## Timeline and Milestones
- Setup + data/EDA: ~15 min. Implementation + smoke test: ~40 min. E1–E6 runs: ~60–90 min
  (MNIST runs are seconds–minutes each on A6000; CIFAR a few min each). Analysis + figures:
  ~30 min. Report: ~25 min. Buffer ~25%.

## Potential Challenges
- **Gradient blocking** at hard samples → use straight-through / Gumbel relaxation.
- **Downstream inversion** of soft samples (VQ-VAE caution) → include hard/ST variants.
- **Dead/indecisive units** → monitor unit usage; keep runs short enough to iterate.
- **τ scale sensitivity** → sweep and anneal; cap τ≈0.5 per the literature.
- One GPU is busy; use a free A6000 (set CUDA_VISIBLE_DEVICES).

## Success Criteria
- A clean, reproducible framework + a results table covering all variants/baselines on
  MNIST with ≥3 seeds and statistical tests.
- At least the τ/σ sweep, train/test×MC ablation, and robustness eval completed, with
  figures.
- A clear, evidence-based answer to "what does intermediate-layer sampling do?", honestly
  reporting costs and any benefits — even if null.
