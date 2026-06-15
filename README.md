# Sampling Neural Network

What if we form a probability distribution over an **intermediate** layer's activations and
**sample** from it — the way we sample the final softmax — instead of forwarding the
deterministic activations? This project runs the controlled study that the literature is
missing, with a single drop-in `SamplingLayer` compared against deterministic and standard
baselines on MNIST and CIFAR-10.

## Key findings
- **Distribution type dominates.** Continuous **Gaussian** sampling (VAE reparameterization)
  is essentially **free** — MNIST 97.7% vs 97.8% deterministic — and on **CIFAR-10 it beats
  deterministic** on calibration (ECE 0.074 vs 0.112) and NLL (0.78 vs 0.94).
- **The literal softmax analogue is expensive.** Categorical **Gumbel-Softmax** sampling
  (collapse a hidden layer to ~one active unit) costs 8–10 pts on MNIST and **collapses to
  10–25% (chance) on CIFAR-10** — a severe, high-variance information bottleneck.
- **Sampling regularizes.** Every stochastic variant shrinks the train–test gap and most
  lower ECE.
- **MC-averaging recovers the cost.** Averaging K test-time samples mirrors softmax
  sampling: dropout 96.6%→97.9% as K:1→50.
- **Gradient variance explains it.** Bernoulli straight-through lowest, hard-categorical
  highest — the same order as the accuracy cost. Cost also scales with sampling sharpness
  (low τ), earlier layer position, and task difficulty.

See **[REPORT.md](REPORT.md)** for the full study (tables, statistics, figures, discussion).

## Reproduce
```bash
source .venv/bin/activate          # isolated uv venv (torch 2.12 + cu130)
# smoke test (~10s):
CUDA_VISIBLE_DEVICES=0 python src/run_experiments.py --exp smoke --seeds 0
# full experiments (~36 min on one A6000; E6/CIFAR can run on a second GPU):
CUDA_VISIBLE_DEVICES=0 python src/run_experiments.py --exp E1 --seeds 0 1 2   # also E2..E5
CUDA_VISIBLE_DEVICES=1 python src/run_experiments.py --exp E6 --seeds 0 1 2   # CIFAR-10
# aggregate, run stats, make figures:
python src/analyze.py
```
Results land in `results/metrics/*.jsonl`, summaries in `results/summary_*.csv`, figures in
`results/plots/`. All RNGs seeded; environment pinned in `results/environment.json`.

## File structure
```
planning.md            # Phase 0/1: motivation, novelty, hypothesis decomposition, plan
REPORT.md              # PRIMARY deliverable: full results + analysis
CODE_WALKTHROUGH.md    # code structure, functions, how-to-run
src/
  data.py              # load pre-downloaded MNIST/CIFAR (no torchvision download), train-stat norm
  models.py            # SamplingLayer (7 modes) + MLP / SmallCNN backbones
  train.py             # seeds, train loop (early stopping), metrics (ECE/NLL/grad-var), MC eval
  run_experiments.py   # E1..E6 orchestration -> results/metrics/*.jsonl
  analyze.py           # aggregate + paired t-tests + figures
results/
  metrics/*.jsonl      # raw per-run records      plots/*.png  figures
  summary_*.csv        # aggregated tables         stats_*.json paired t-tests
literature_review.md, resources.md, papers/, datasets/, code/   # pre-gathered resources
```

## What the `SamplingLayer` does
Drop-in after a hidden ReLU. Modes: `deterministic`, `dropout`, `gauss_noise`,
`gaussian_reparam` (𝒩(μ,σ) reparameterization), `bernoulli_st` (Bengio straight-through),
`gumbel_softmax` / `gumbel_st` (**the direct softmax analogue**: softmax over hidden units →
Gumbel-Softmax / hard one-hot sample). Train- vs test-time sampling and MC-averaging are
toggleable.
