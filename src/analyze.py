"""Aggregate JSONL results, run statistics, and produce figures + summary tables.

Outputs:
  results/summary_*.csv           aggregated mean/std tables
  results/stats.json              paired t-tests vs deterministic
  results/plots/*.png             figures
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
OUT = ROOT / "results"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODE_LABEL = {
    "deterministic": "Deterministic",
    "dropout": "Dropout",
    "gauss_noise": "Gaussian noise",
    "gaussian_reparam": "Gaussian reparam (VAE)",
    "bernoulli_st": "Bernoulli ST",
    "gumbel_softmax": "Gumbel-Softmax (soft)",
    "gumbel_st": "Gumbel ST (hard 1-hot)",
}
ORDER = list(MODE_LABEL.keys())


def load(exp):
    path = MET / f"{exp}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


# ---------- E1 core table ----------
def analyze_core(exp, dataset_label, mc_key="test_mc1", mc_avg_key="test_mc10"):
    recs = load(exp)
    if not recs:
        print(f"[{exp}] no records")
        return None
    rows = []
    for r in recs:
        main = r.get(mc_key, r["test_off"])
        avg = r.get(mc_avg_key, main)
        rows.append({
            "mode": r["mode"], "seed": r["seed"],
            "acc_off": r["test_off"]["acc"],
            "acc_s1": main["acc"], "acc_mc": avg["acc"],
            "nll_s1": main["nll"], "ece_s1": main["ece"], "ece_mc": avg["ece"],
            "entropy": main.get("entropy", np.nan),
            "gen_gap": r.get("gen_gap", np.nan),
            "grad_var": r.get("grad_var", np.nan),
            "train_acc": r["train_acc"],
        })
    df = pd.DataFrame(rows)
    agg = df.groupby("mode").agg(["mean", "std"])
    # flatten
    summary = pd.DataFrame(index=[m for m in ORDER if m in df["mode"].unique()])
    for m in summary.index:
        sub = df[df["mode"] == m]
        summary.loc[m, "n"] = len(sub)
        for col in ["acc_off", "acc_s1", "acc_mc", "nll_s1", "ece_s1", "ece_mc",
                    "gen_gap", "grad_var", "train_acc"]:
            summary.loc[m, col + "_mean"] = sub[col].mean()
            summary.loc[m, col + "_std"] = sub[col].std()
    summary.index = [MODE_LABEL.get(m, m) for m in summary.index]
    summary.to_csv(OUT / f"summary_{exp}.csv")
    print(f"\n===== {exp} ({dataset_label}) =====")
    show = summary[["acc_off_mean", "acc_s1_mean", "acc_mc_mean", "nll_s1_mean",
                    "ece_s1_mean", "gen_gap_mean", "grad_var_mean"]].round(4)
    print(show.to_string())

    # paired t-test vs deterministic (acc_s1)
    stat = {}
    det = df[df["mode"] == "deterministic"].sort_values("seed")["acc_s1"].values
    for m in df["mode"].unique():
        if m == "deterministic":
            continue
        cur = df[df["mode"] == m].sort_values("seed")["acc_s1"].values
        if len(cur) == len(det) and len(det) > 1:
            t, p = stats.ttest_rel(cur, det)
            stat[m] = {"delta_acc": float(cur.mean() - det.mean()),
                       "t": float(t), "p": float(p),
                       "cohens_d": float(cohens_d(cur, det))}
    (OUT / f"stats_{exp}.json").write_text(json.dumps(stat, indent=2))
    return df, summary, stat


def plot_core_bars(df, exp, title):
    modes = [m for m in ORDER if m in df["mode"].unique()]
    labels = [MODE_LABEL[m] for m in modes]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col1, col2, name) in zip(axes, [
            ("acc_s1", "acc_mc", "Test accuracy"),
            ("ece_s1", None, "ECE (calibration error)"),
            ("gen_gap", None, "Generalization gap (train-test)")]):
        means1 = [df[df["mode"] == m][col1].mean() for m in modes]
        errs1 = [df[df["mode"] == m][col1].std() for m in modes]
        x = np.arange(len(modes))
        if col2:
            means2 = [df[df["mode"] == m][col2].mean() for m in modes]
            ax.bar(x - 0.2, means1, 0.4, yerr=errs1, label="single sample", capsize=3)
            ax.bar(x + 0.2, means2, 0.4, label="MC-averaged", capsize=3, alpha=0.8)
            ax.legend()
        else:
            ax.bar(x, means1, 0.6, yerr=errs1, capsize=3, color="C2")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{exp}_bars.png", dpi=120)
    plt.close(fig)


# ---------- E2 sweep ----------
def analyze_sweep():
    recs = load("E2_sweep_mnist")
    if not recs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (mode, knob, label) in zip(axes, [
            ("gumbel_softmax", "tau", "Gumbel-Softmax temperature τ"),
            ("gauss_noise", "sigma", "Gaussian noise scale σ")]):
        sub = [r for r in recs if r["mode"] == mode]
        knobs = sorted(set(r["sl"][knob] for r in sub))
        accs1, accmc, eces = [], [], []
        for k in knobs:
            rk = [r for r in sub if r["sl"][knob] == k]
            accs1.append(np.mean([r["test_mc1"]["acc"] for r in rk]))
            accmc.append(np.mean([r["test_mc10"]["acc"] for r in rk]))
            eces.append(np.mean([r["test_mc1"]["ece"] for r in rk]))
        ax.plot(knobs, accs1, "o-", label="acc (1 sample)")
        ax.plot(knobs, accmc, "s--", label="acc (MC-10)")
        ax2 = ax.twinx()
        ax2.plot(knobs, eces, "^:", color="C3", label="ECE")
        ax2.set_ylabel("ECE", color="C3")
        ax.set_xlabel(label)
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"{MODE_LABEL[mode]}: effect of sampling amount")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOTS / "E2_sweep.png", dpi=120)
    plt.close(fig)
    print("\n[E2] sweep plot saved")
    # table
    rows = []
    for r in recs:
        knob = "tau" if r["mode"] == "gumbel_softmax" else "sigma"
        rows.append({"mode": r["mode"], "knob_val": r["sl"][knob],
                     "acc_s1": r["test_mc1"]["acc"], "acc_mc": r["test_mc10"]["acc"],
                     "ece": r["test_mc1"]["ece"]})
    pd.DataFrame(rows).groupby(["mode", "knob_val"]).mean().round(4).to_csv(OUT / "summary_E2_sweep.csv")


# ---------- E4 MC averaging ----------
def analyze_mc():
    recs = load("E4_traintest_mc_mnist")
    if not recs:
        return
    mcs = [1, 5, 20, 50]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    modes = sorted(set(r["mode"] for r in recs))
    for r_mode in modes:
        sub = [r for r in recs if r["mode"] == r_mode]
        accs, eces = [], []
        for mc in mcs:
            key = f"test_mc{mc}"
            accs.append(np.mean([r[key]["acc"] for r in sub if key in r]))
            eces.append(np.mean([r[key]["ece"] for r in sub if key in r]))
        axes[0].plot(mcs, accs, "o-", label=MODE_LABEL.get(r_mode, r_mode))
        axes[1].plot(mcs, eces, "o-", label=MODE_LABEL.get(r_mode, r_mode))
    axes[0].set_title("MC-averaging recovers accuracy")
    axes[0].set_xlabel("# MC samples at test (K)"); axes[0].set_ylabel("Test accuracy")
    axes[1].set_title("MC-averaging improves calibration")
    axes[1].set_xlabel("# MC samples at test (K)"); axes[1].set_ylabel("ECE")
    for a in axes:
        a.set_xscale("log"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "E4_mc.png", dpi=120)
    plt.close(fig)
    print("[E4] MC plot saved")


# ---------- E5 robustness ----------
def analyze_robust():
    recs = load("E5_robust_mnist")
    if not recs:
        return
    sevs = sorted(set(float(s) for r in recs for s in r["sev"].keys()))
    modes = [m for m in ORDER if m in set(r["mode"] for r in recs)]
    fig, ax = plt.subplots(figsize=(8, 6))
    table = {}
    for m in modes:
        sub = [r for r in recs if r["mode"] == m]
        ys, es = [], []
        for s in sevs:
            vals = [r["sev"][str(s)]["acc"] for r in sub]
            ys.append(np.mean(vals)); es.append(np.std(vals))
        ax.errorbar(sevs, ys, yerr=es, marker="o", capsize=3, label=MODE_LABEL.get(m, m))
        table[m] = ys
    ax.set_xlabel("Input Gaussian noise σ (severity)")
    ax.set_ylabel("Test accuracy (MC-10 for stochastic)")
    ax.set_title("Robustness to input corruption")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "E5_robust.png", dpi=120)
    plt.close(fig)
    pd.DataFrame(table, index=[f"sev_{s}" for s in sevs]).T.round(4).to_csv(OUT / "summary_E5_robust.csv")
    print("[E5] robustness plot saved")


# ---------- E3 position ----------
def analyze_position():
    recs = load("E3_position_mnist")
    if not recs:
        return
    rows = []
    for r in recs:
        rows.append({"mode": r["mode"], "pos": r["sample_pos"],
                     "acc_s1": r["test_mc1"]["acc"], "acc_mc": r["test_mc10"]["acc"],
                     "ece": r["test_mc1"]["ece"]})
    df = pd.DataFrame(rows)
    df.groupby(["mode", "pos"]).mean().round(4).to_csv(OUT / "summary_E3_position.csv")
    print("\n[E3] position summary:")
    print(df.groupby(["mode", "pos"]).mean().round(4).to_string())


def main():
    e1 = analyze_core("E1_core_mnist", "MNIST")
    if e1:
        plot_core_bars(e1[0], "E1_core_mnist", "MNIST — intermediate-layer sampling (pos=mid)")
    analyze_sweep()
    analyze_position()
    analyze_mc()
    analyze_robust()
    e6 = analyze_core("E6_core_cifar10", "CIFAR-10")
    if e6:
        plot_core_bars(e6[0], "E6_core_cifar10", "CIFAR-10 — intermediate-layer sampling")
    print("\nAnalysis complete. Plots in results/plots/")


if __name__ == "__main__":
    main()
