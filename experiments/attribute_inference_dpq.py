"""
experiments/attribute_inference_dpq.py

"Removed for safety, leaks anyway" — attribute-inference attack on the single
most sensitive item in depression screening: DPQ090, suicidal ideation
("Suicidal_Thoughts"), and whether the leak scales with generator fidelity.

The main models DROP all DPQ items. Here we KEEP suicidal ideation as a column,
train a generator on [demographics/lifestyle QIDs + suicidal-ideation flag],
release the synthetic data, then ask:

    Can an attacker who sees only the synthetic data reconstruct which REAL
    individuals reported suicidal ideation, from their demographics alone?

Attack (standard synthetic-data AIA): fit RF on synthetic (QIDs -> suicidal),
score ROC AUC on real test individuals. Compared to a marginal baseline (0.5)
and a real-data upper bound. For each (model, seed) we also record the
generator's marginal fidelity, to test: does AIA leakage rise with fidelity?

Models: VAE (from-scratch, unconditional) + SDV TVAE + SDV GaussianCopula.
CTGAN is skipped (conditional GAN needs bespoke unconditional retraining).

    python experiments/attribute_inference_dpq.py            # full sweep
    python experiments/attribute_inference_dpq.py --seed 42 --model vae
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from preprocess import build_dataset, DPQ_ITEMS
from models.vae.model import VAE, vae_loss
from utils.seed_utils import set_all_seeds

warnings.filterwarnings("ignore")
SENSITIVE = "Suicidal_Thoughts"   # DPQ090
MODELS = ["vae", "gc", "tvae"]


def _prep(seed: int):
    """Item-inclusive split. Returns scaled QID matrices, binary target, and
    the continuous-column mask over qid_cols (for fidelity)."""
    df = build_dataset(drop_leakage=False)
    assert SENSITIVE in df.columns
    y = (df[SENSITIVE].values >= 1).astype(int)
    qid_cols = [c for c in df.columns
                if c not in DPQ_ITEMS and c != config.TARGET_COL]
    X = df[qid_cols].astype(float).values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    scaler = MinMaxScaler().fit(Xtr)
    cont_mask = np.array([c in config.CONTINUOUS_COLS for c in qid_cols])
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte, qid_cols, cont_mask


def _gen_vae(Xtr, ytr, binary_idx, seed, epochs=120):
    set_all_seeds(seed)
    dev = torch.device(config.DEVICE)
    mat = np.hstack([Xtr, ytr.reshape(-1, 1).astype(np.float32)]).astype(np.float32)
    model = VAE(input_dim=mat.shape[1], binary_col_indices=binary_idx,
                hidden_dims=config.VAE_HIDDEN_DIMS,
                latent_dim=config.VAE_LATENT_DIM).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.VAE_LR)
    loader = DataLoader(TensorDataset(torch.tensor(mat)), batch_size=256,
                        shuffle=True, drop_last=True)
    for ep in range(1, epochs + 1):
        beta = min(1.0, ep / 50)
        model.train()
        for (b,) in loader:
            b = b.to(dev)
            recon, mu, lv = model(b)
            loss, _, _ = vae_loss(recon, b, mu, lv, kl_weight=beta)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        syn = model.generate(len(Xtr), dev).cpu().numpy()
    n = Xtr.shape[1]
    rng = np.random.default_rng(seed)
    syn_y = (rng.random(len(syn)) < np.clip(syn[:, n], 0, 1)).astype(int)
    return syn[:, :n], syn_y


def _gen_sdv(kind, Xtr, ytr, qid_cols, cont_mask, seed):
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer, GaussianCopulaSynthesizer
    set_all_seeds(seed)
    df = pd.DataFrame(Xtr, columns=qid_cols)
    df[SENSITIVE] = ytr
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    for c, is_cont in zip(qid_cols, cont_mask):
        meta.update_column(c, sdtype="numerical" if is_cont else "categorical")
    meta.update_column(SENSITIVE, sdtype="categorical")
    if kind == "tvae":
        synth = TVAESynthesizer(meta, epochs=config.TVAE_EPOCHS, verbose=False)
    else:
        synth = GaussianCopulaSynthesizer(meta)
    synth.fit(df)
    out = synth.sample(len(Xtr))
    return out[qid_cols].astype(float).values, out[SENSITIVE].astype(int).values


def _fidelity(real_X, syn_X, real_y, syn_y, cont_mask):
    """Mean per-column effect-size fidelity (KSComplement / 1-TVD)."""
    fids = []
    for j in range(real_X.shape[1]):
        r, s = real_X[:, j], syn_X[:, j]
        if cont_mask[j]:
            fids.append(1.0 - ks_2samp(r, s).statistic)
        else:
            lv = np.union1d(np.unique(r), np.unique(s))
            tvd = 0.5 * sum(abs((r == v).mean() - (s == v).mean()) for v in lv)
            fids.append(1.0 - tvd)
    tvd_y = 0.5 * sum(abs((real_y == v).mean() - (syn_y == v).mean())
                      for v in (0, 1))
    fids.append(1.0 - tvd_y)
    return float(np.mean(fids))


def _snap(syn_X, real_X, cont_mask):
    """Snap non-continuous synthetic QID cols to nearest valid real level.
    No-op for SDV (already valid); fixes VAE soft sigmoid outputs."""
    out = syn_X.copy()
    for j in range(syn_X.shape[1]):
        if cont_mask[j]:
            continue
        levels = np.unique(real_X[:, j])
        if len(levels) == 1:
            out[:, j] = levels[0]; continue
        idx = np.searchsorted(levels, syn_X[:, j]).clip(1, len(levels) - 1)
        left, right = levels[idx - 1], levels[idx]
        out[:, j] = np.where(np.abs(syn_X[:, j] - left) <= np.abs(right - syn_X[:, j]),
                             left, right)
    return out


def _auc(Xtr, ytr, Xte, yte, seed):
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def run_one(model, seed):
    Xtr, Xte, ytr, yte, qid_cols, cont_mask = _prep(seed)
    n = Xtr.shape[1]
    if model == "vae":
        binary_idx = [i for i in range(n) if not cont_mask[i]] + [n]
        syn_X, syn_y = _gen_vae(Xtr, ytr, binary_idx, seed)
    else:
        syn_X, syn_y = _gen_sdv(model, Xtr, ytr, qid_cols, cont_mask, seed)

    syn_X = _snap(syn_X, Xtr, cont_mask)
    fidelity = _fidelity(Xtr, syn_X, ytr, syn_y, cont_mask)
    syn_auc = 0.5 if syn_y.sum() in (0, len(syn_y)) else _auc(syn_X, syn_y, Xte, yte, seed)
    real_auc = _auc(Xtr, ytr, Xte, yte, seed)
    lift = (syn_auc - 0.5) / (real_auc - 0.5) if real_auc > 0.5 else 0.0
    return {"model": model, "seed": seed, "prevalence": round(float(ytr.mean()), 4),
            "fidelity": round(fidelity, 4), "syn_auc": round(syn_auc, 4),
            "real_auc": round(real_auc, 4), "recovered_frac": round(lift, 4),
            "syn_rate": round(float(syn_y.mean()), 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model", choices=MODELS)
    a = ap.parse_args()
    seeds = [a.seed] if a.seed else config.SEEDS
    models = [a.model] if a.model else MODELS

    rows = [run_one(m, s) for m in models for s in seeds]
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "aia_results.csv"
    df.to_csv(out, index=False)

    print("\n=== per (model, seed) ===")
    print(df.to_string(index=False))
    print("\n=== per-model mean ===")
    g = df.groupby("model")[["fidelity", "syn_auc", "real_auc", "recovered_frac"]].mean()
    print(g.round(4).to_string())
    if len(df) > 2 and df["fidelity"].std() > 0:
        r, p = pearsonr(df["fidelity"], df["syn_auc"])
        print(f"\nPearson(fidelity, syn_auc) = {r:.3f}  (p={p:.3f}, n={len(df)})")
    print(f"\nsaved -> {out}")
    # self-check
    assert df["syn_auc"].between(0, 1).all() and df["real_auc"].between(0, 1).all()
    print("OK")
