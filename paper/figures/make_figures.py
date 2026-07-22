"""
paper/figures/make_figures.py — the two figures for the pivoted paper.

Reads frozen evaluation/results/faa_{model}_seed{n}.csv (4 models x 5 seeds)
and evaluation/results/attack_validation.csv. No torch import.

    python paper/figures/make_figures.py

Fig 1 (fig1_dcr_vs_fidelity): DCR rewards infidelity — dcr_ratio vs fidelity.
Fig 2 (fig2_c2st_vs_mia):     naive MIA cries wolf — C2ST vs DOMIAS + controls.
Fig 3 (fig3_leakage_dial):    KDE bandwidth sweep — where the DCR gate flips
                              relative to the MIA gate. Needs
                              evaluation/results/leakage_dial.csv
                              (python experiments/leakage_dial.py).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent
RES = ROOT / "evaluation" / "results"
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["vae", "ctgan", "tvae", "gc"]
LABEL = {"vae": "VAE", "ctgan": "CTGAN", "tvae": "TVAE", "gc": "GaussCopula"}
COLOR = {"vae": "tab:blue", "ctgan": "tab:orange", "tvae": "tab:green", "gc": "tab:red"}
MARK = {"vae": "o", "ctgan": "s", "tvae": "^", "gc": "D"}
W = 3.5

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300,
                     "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def _load():
    rows = [pd.read_csv(RES / f"faa_{m}_seed{s}.csv").iloc[0]
            for m in MODELS for s in SEEDS]
    df = pd.DataFrame(rows)
    assert len(df) == 20, f"expected 20 runs, got {len(df)}"
    return df


def _load_dial():
    p = RES / "leakage_dial.csv"
    assert p.exists(), f"{p} missing — run python experiments/leakage_dial.py"
    return pd.read_csv(p)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def fig1(df):
    """dcr_ratio (y) vs fidelity (x): high DCR pairs with low fidelity."""
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")  # DCR gate
    for m in MODELS:
        s = df[df.model == m]
        ax.scatter(s.fidelity_score, s.dcr_ratio, c=COLOR[m], marker=MARK[m],
                   s=38, label=LABEL[m], edgecolors="none")
    ax.text(ax.get_xlim()[0], 1.0, " DCR gate", fontsize=7, color="gray", va="bottom")
    ax.set_xlabel("Marginal fidelity (KSComplement / 1$-$TVD)")
    ax.set_ylabel("DCR ratio (syn/holdout)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, "fig1_dcr_vs_fidelity")


def fig2(df):
    """C2ST (distinguishability) vs DOMIAS MIA AUC, per model, + controls."""
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.axhline(0.55, color="gray", lw=0.8, ls="--")  # MIA gate
    ax.text(ax.get_xlim()[0] if False else 0.02, 0.56, "MIA gate 0.55",
            transform=ax.get_yaxis_transform(), fontsize=7, color="gray")
    g = df.groupby("model")
    x = range(len(MODELS))
    ax.bar([i - 0.2 for i in x], [g.get_group(m).c2st_acc.mean() for m in MODELS],
           0.4, label="C2ST (distinguishability)", color="lightgray")
    ax.bar([i + 0.2 for i in x], [g.get_group(m).mia_auc_domias.mean() for m in MODELS],
           0.4, yerr=[g.get_group(m).mia_auc_domias.std() for m in MODELS],
           label="DOMIAS MIA AUC", color="tab:purple", capsize=2)
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=7)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="center left")
    fig.tight_layout()
    _save(fig, "fig2_c2st_vs_mia")


def fig3(dial):
    """DCR ratio and DOMIAS AUC against KDE bandwidth: where the gates flip."""
    g = dial.groupby("bandwidth")
    bw = list(g.groups.keys())

    fig, ax = plt.subplots(figsize=(W, 2.8))
    # Shade where Axis 3 passes but Axis 4 still fails — the finding.
    gap = [b for b in bw if g.dcr_ratio.mean()[b] >= 1.0
           and g.mia_auc_domias.mean()[b] > 0.55]
    if gap:
        lo, hi = min(gap) * 0.87, max(bw) * 1.06
        ax.axvspan(lo, hi, color="0.85", zorder=0)
        # Label sits above the axes: the span is only ~0.7 of a decade wide, so
        # any in-plot placement collides with one of the two curves crossing it.
        ax.text((lo * hi) ** 0.5, 1.02, "DCR passes, MIA fires", fontsize=6.5,
                color="0.35", ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
    ax.errorbar(bw, g.dcr_ratio.mean(), yerr=g.dcr_ratio.std(), color="tab:red",
                marker="o", ms=3.5, lw=1, capsize=2, label="DCR ratio")
    ax.axhline(1.0, color="tab:red", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("KDE bandwidth (leakage $\\leftarrow$ | $\\rightarrow$ smoothing)")
    ax.set_ylabel("DCR ratio (syn/holdout)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.errorbar(bw, g.mia_auc_domias.mean(), yerr=g.mia_auc_domias.std(),
                 color="tab:purple", marker="s", ms=3.5, lw=1, capsize=2,
                 label="DOMIAS AUC")
    ax2.plot(bw, g.mia_auc_domias_clf.mean(), color="tab:purple", marker="^",
             ms=3.5, lw=1, ls=":", label="DOMIAS AUC (clf variant)")
    ax2.axhline(0.55, color="tab:purple", lw=0.7, ls="--")
    ax2.axhline(0.5, color="black", lw=0.5)
    ax2.set_ylabel("MIA AUC", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="center left")
    fig.tight_layout()
    _save(fig, "fig3_leakage_dial")


if __name__ == "__main__":
    df = _load()
    fig1(df)
    fig2(df)
    dial = _load_dial()
    fig3(dial)
    av = RES / "attack_validation.csv"
    print("controls:", pd.read_csv(av).to_dict("records") if av.exists()
          else "MISSING — run evaluation/attack_validation.py")
    # self-check: every figure exists and is non-trivial
    for stem in ("fig1_dcr_vs_fidelity", "fig2_c2st_vs_mia", "fig3_leakage_dial"):
        p = FIG / f"{stem}.png"
        assert p.exists() and p.stat().st_size > 8000, stem
    print("OK")
