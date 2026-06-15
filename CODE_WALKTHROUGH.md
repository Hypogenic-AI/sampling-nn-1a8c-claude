# Code Walkthrough — Sampling Neural Network

## Structure overview
```
src/data.py            # dataset loading + train-statistic normalization (no leakage)
src/models.py          # SamplingLayer (the object of study) + MLP / SmallCNN backbones
src/train.py           # seeds, training loop, metrics, MC-averaged & robustness evaluation
src/run_experiments.py # experiment definitions E1..E6, writes results/metrics/*.jsonl
src/analyze.py         # aggregation, paired t-tests, all figures
```
Data flow: `data.get_dataset` → `train.build_model` (inserts `SamplingLayer`) →
`train.train_model` (early stopping on val) → `train.evaluate` (single-sample + MC) →
`run_experiments` writes JSONL → `analyze` produces CSVs/PNGs/stats.

## Key components

### `models.SamplingLayer(dim, mode, tau, sigma, dropout_p, sample_test, hard, kl_weight, cat_scale)`
**Purpose.** Drop-in module placed after a hidden ReLU that converts activations into a
distribution and draws a sample — generalizing final-softmax sampling to a hidden layer.
**Modes.** `deterministic` (identity), `dropout` (Bernoulli mask), `gauss_noise` (additive
σ·𝒩), `gaussian_reparam` (per-unit 𝒩(μ,σ) via reparameterization `μ+σ⊙ε`, optional
KL-to-prior), `bernoulli_st` (per-unit Bernoulli with straight-through gradient),
`gumbel_softmax`/`gumbel_st` (treat units as categorical logits → `F.gumbel_softmax`, soft
or hard one-hot — *the direct softmax analogue*; one-hot rescaled by `dim`).
**Design rationale.** A single module with a `mode` switch keeps the backbone, optimizer,
and data identical across conditions, so the *sampling operator is the only variable*.
`sample_test` toggles eval-time sampling (train-time vs test-time ablation); `last_kl`
exposes the KL term to the loss. The categorical rescale by `dim` keeps downstream magnitude
comparable to a dense ReLU activation (a raw one-hot has tiny norm).
**Example.** `SamplingLayer(256, mode="gumbel_st", tau=0.5)`.

### `models.MLP` / `models.SmallCNN`
MNIST MLP `784→256→256→10` with the sampler after hidden `sample_pos` (0 or 1); CIFAR
4-conv + 2-FC CNN with the sampler on the 256-d FC features. Both expose `kl()` for the
optional VAE-style KL term.

### `train.train_model(...)`
Adam + cross-entropy (+ `kl_weight*kl`), minibatching, **early stopping on validation
accuracy** (patience 5) with best-checkpoint restore. Deterministic given `seed`.

### `train.evaluate(model, x, y, device, mc, sample_test)`
Averages softmax probabilities over `mc` stochastic passes (MC inference), then computes
accuracy, **NLL**, **ECE (15-bin)**, predictive entropy. `sample_test` controls whether the
sampler fires at eval time.

### `train.measure_grad_variance(...)`
Variance of the loss gradient w.r.t. the parameter feeding the sampler, across repeated
stochastic forward passes on a **fixed** minibatch — isolates *sampling* noise (not data
noise). Explains why categorical sampling is hard to train.

### `run_experiments.py`
`--exp {smoke,E1,E2,E3,E4,E5,E6,all}`, `--seeds`, `--epochs`. `CORE_MODES` lists the seven
operators; `run_one` trains+evaluates one config and returns a record. Each experiment dumps
a JSONL to `results/metrics/`.

## How to run
```bash
source .venv/bin/activate
python src/data.py                                   # verify data loads (shapes/stats)
CUDA_VISIBLE_DEVICES=0 python src/run_experiments.py --exp smoke --seeds 0   # ~10 s
CUDA_VISIBLE_DEVICES=0 python src/run_experiments.py --exp E1 --seeds 0 1 2  # +E2..E5
CUDA_VISIBLE_DEVICES=1 python src/run_experiments.py --exp E6 --seeds 0 1 2  # CIFAR-10
python src/analyze.py                                 # tables + figures
```
**Runtime.** MNIST runs: seconds–minutes each; CIFAR ≈40–75 s each; full suite ≈36 min of
compute (MNIST and CIFAR can run concurrently on two GPUs). **Resources.** ~2 GB GPU RAM;
datasets pre-downloaded in `datasets/`.

## Reproducibility & quality notes
- All RNGs seeded (`train.set_seed`); 3 seeds per config; paired t-tests across seeds.
- Normalization fit on the **train split only**; test set used only for final evaluation;
  a 10% validation split drives early stopping.
- Environment pinned in `results/environment.json` (torch 2.12.0+cu130, Python 3.12).
- **Known limitation.** Categorical modes can collapse on CIFAR-10 (see REPORT §6/§7); this
  is reported honestly under the shared protocol — a τ-anneal + per-mode LR is future work.
