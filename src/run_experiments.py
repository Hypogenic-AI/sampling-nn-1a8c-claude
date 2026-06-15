"""Orchestrate experiments E1-E6 for the Sampling-NN study.

Each config is trained with seeds and evaluated; results are appended to
results/metrics/<exp>.jsonl. Designed to be re-runnable; uses --exp to select.

Usage:
  python src/run_experiments.py --exp smoke
  python src/run_experiments.py --exp E1 --seeds 0 1 2
  python src/run_experiments.py --exp all --seeds 0 1 2
"""
import argparse
import json
import time
from pathlib import Path

import torch

from data import get_dataset, make_splits
from train import (set_seed, build_model, train_model, evaluate,
                   measure_grad_variance)

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
MET.mkdir(parents=True, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# default sampling-layer kwargs per mode (tuned mild defaults from the literature)
def default_sl(mode, **over):
    base = dict(mode=mode, tau=0.5, sigma=0.3, dropout_p=0.5,
                sample_test=True, hard=False, kl_weight=0.0, cat_scale=True)
    base.update(over)
    return base


CORE_MODES = [
    ("deterministic", {}),
    ("dropout", {}),
    ("gauss_noise", {}),
    ("gaussian_reparam", {"kl_weight": 1e-3}),
    ("bernoulli_st", {}),
    ("gumbel_softmax", {}),     # soft categorical (direct softmax analogue)
    ("gumbel_st", {}),          # hard one-hot sample of a single unit
]


def run_one(dataset, mode, sl_over, seed, device, epochs, sample_pos=0,
            mc_list=(1, 10), kl_weight=0.0, log=print, hidden=256):
    set_seed(seed)
    (xtr, ytr, xte, yte), meta = get_dataset(dataset)
    tr_idx, val_idx = make_splits(xtr, ytr, val_frac=0.1, seed=seed)
    sl = default_sl(mode, **sl_over)
    klw = sl.pop("kl_weight", 0.0)
    model = build_model(dataset, sample_pos, sl, hidden=hidden)
    t0 = time.time()
    model, hist, train_acc = train_model(
        model, (xtr, ytr, xte, yte), device, epochs=epochs,
        tr_idx=tr_idx, val_idx=val_idx, kl_weight=klw, seed=seed, log=None)
    dt = time.time() - t0

    rec = {"dataset": dataset, "mode": mode, "seed": seed,
           "sample_pos": sample_pos, "sl": {**sl, "kl_weight": klw},
           "train_acc": train_acc, "train_time_s": round(dt, 1),
           "epochs_run": len(hist)}

    # deterministic single-pass test (sampling OFF) -> "expectation" behavior
    rec["test_off"] = evaluate(model, xte, yte, device, mc=1, sample_test=False)
    # stochastic single-sample and MC-averaged
    if mode != "deterministic":
        for mc in mc_list:
            rec[f"test_mc{mc}"] = evaluate(model, xte, yte, device, mc=mc, sample_test=True)
        # gradient variance at the sampling layer
        try:
            rec["grad_var"] = measure_grad_variance(model, xtr[tr_idx], ytr[tr_idx], device)
        except Exception as e:
            rec["grad_var"] = None
            log(f"    grad_var failed: {e}")
    else:
        rec["test_mc1"] = rec["test_off"]
        rec["grad_var"] = measure_grad_variance(model, xtr[tr_idx], ytr[tr_idx], device)
    # generalization gap
    main = rec.get("test_mc1", rec["test_off"])
    rec["gen_gap"] = round(train_acc - main["acc"], 4)
    log(f"  [{dataset}/{mode}/seed{seed}/pos{sample_pos}] "
        f"off_acc {rec['test_off']['acc']:.4f} "
        f"mc1_acc {rec.get('test_mc1', {}).get('acc', float('nan')):.4f} "
        f"({dt:.0f}s)")
    return rec


def dump(exp, recs):
    path = MET / f"{exp}.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} records -> {path}")


def exp_smoke(device, seeds):
    recs = []
    for mode, over in [("deterministic", {}), ("gumbel_st", {}), ("gaussian_reparam", {"kl_weight": 1e-3})]:
        recs.append(run_one("mnist", mode, over, seeds[0], device, epochs=2))
    dump("smoke", recs)


def exp_E1(device, seeds, epochs=15):
    """Core distribution-type comparison on MNIST, mid layer (pos=0)."""
    recs = []
    for seed in seeds:
        for mode, over in CORE_MODES:
            recs.append(run_one("mnist", mode, over, seed, device, epochs=epochs,
                                 sample_pos=0, mc_list=(1, 10, 30)))
    dump("E1_core_mnist", recs)


def exp_E2(device, seeds, epochs=15):
    """Temperature / noise-scale sweep on the two continuous knobs."""
    recs = []
    for seed in seeds:
        for tau in [0.1, 0.5, 1.0, 2.0, 5.0]:
            recs.append(run_one("mnist", "gumbel_softmax", {"tau": tau}, seed, device,
                                 epochs=epochs, sample_pos=0))
        for sigma in [0.1, 0.3, 0.6, 1.0, 2.0]:
            recs.append(run_one("mnist", "gauss_noise", {"sigma": sigma}, seed, device,
                                 epochs=epochs, sample_pos=0))
    dump("E2_sweep_mnist", recs)


def exp_E3(device, seeds, epochs=15):
    """Layer-position sweep for the categorical (hard) sampler."""
    recs = []
    for seed in seeds:
        for pos in [0, 1]:  # after 1st or 2nd hidden layer
            for mode in ["gumbel_st", "gaussian_reparam", "bernoulli_st"]:
                over = {"kl_weight": 1e-3} if mode == "gaussian_reparam" else {}
                recs.append(run_one("mnist", mode, over, seed, device, epochs=epochs,
                                     sample_pos=pos))
    dump("E3_position_mnist", recs)


def exp_E4(device, seeds, epochs=15):
    """Train-time vs test-time sampling + MC averaging.
    Variant A: sample in train, configurable at test (covered by mc eval).
    Variant B: deterministic train, sample only at test (MC-dropout style)."""
    recs = []
    for seed in seeds:
        # B: train deterministic-ish then sample at test -> use dropout & gauss_noise
        #    (these don't change train objective much) plus gaussian_reparam
        for mode, over in [("dropout", {}), ("gauss_noise", {}),
                           ("gumbel_softmax", {}), ("gaussian_reparam", {"kl_weight": 1e-3})]:
            r = run_one("mnist", mode, over, seed, device, epochs=epochs,
                        sample_pos=0, mc_list=(1, 5, 20, 50))
            recs.append(r)
    dump("E4_traintest_mc_mnist", recs)


def exp_E5(device, seeds, epochs=15):
    """Robustness under input Gaussian noise. Train each model, evaluate across
    severities. Saves per-severity acc for deterministic vs sampling variants."""
    from train import evaluate as _eval
    sevs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    recs = []
    for seed in seeds:
        (xtr, ytr, xte, yte), meta = get_dataset("mnist")
        tr_idx, val_idx = make_splits(xtr, ytr, val_frac=0.1, seed=seed)
        for mode, over in [("deterministic", {}), ("dropout", {}),
                           ("gauss_noise", {}), ("gaussian_reparam", {"kl_weight": 1e-3}),
                           ("gumbel_softmax", {}), ("gumbel_st", {})]:
            set_seed(seed)
            sl = default_sl(mode, **over)
            klw = sl.pop("kl_weight", 0.0)
            model = build_model("mnist", 0, sl, hidden=256)
            model, hist, train_acc = train_model(
                model, (xtr, ytr, xte, yte), device, epochs=epochs,
                tr_idx=tr_idx, val_idx=val_idx, kl_weight=klw, seed=seed, log=None)
            g = torch.Generator().manual_seed(seed + 777)
            row = {"dataset": "mnist", "mode": mode, "seed": seed, "sev": {}}
            for s in sevs:
                xn = xte + s * torch.randn(xte.shape, generator=g)
                use_mc = 10 if mode != "deterministic" else 1
                m = _eval(model, xn, yte, device, mc=use_mc,
                          sample_test=(mode != "deterministic"))
                row["sev"][str(s)] = m
            recs.append(row)
            print(f"  [E5 {mode} seed{seed}] clean {row['sev']['0.0']['acc']:.4f} "
                  f"sev1.0 {row['sev']['1.0']['acc']:.4f}")
    dump("E5_robust_mnist", recs)


def exp_E6(device, seeds, epochs=20):
    """CIFAR-10 confirmation of the core comparison with a small CNN."""
    recs = []
    for seed in seeds:
        for mode, over in CORE_MODES:
            recs.append(run_one("cifar10", mode, over, seed, device, epochs=epochs,
                                 sample_pos=0, mc_list=(1, 10), hidden=256))
    dump("E6_core_cifar10", recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    device = get_device()
    print(f"device={device} exp={args.exp} seeds={args.seeds}")
    fns = {"smoke": lambda: exp_smoke(device, args.seeds),
           "E1": lambda: exp_E1(device, args.seeds, args.epochs),
           "E2": lambda: exp_E2(device, args.seeds, args.epochs),
           "E3": lambda: exp_E3(device, args.seeds, args.epochs),
           "E4": lambda: exp_E4(device, args.seeds, args.epochs),
           "E5": lambda: exp_E5(device, args.seeds, args.epochs),
           "E6": lambda: exp_E6(device, args.seeds, 20)}
    if args.exp == "all":
        for k in ["E1", "E2", "E3", "E4", "E5", "E6"]:
            print(f"\n===== {k} =====")
            fns[k]()
    else:
        fns[args.exp]()


if __name__ == "__main__":
    main()
