"""Basic multimodal disease-progression demo on SIMULATED data.

Five small experiments, one per focus area of the PhD project description:
  1. Multimodal fusion          2. Sparse longitudinal progression
  3. Causal treatment effect    4. Missing modalities
  5. Fairness across subgroups

Run:  python progression_demo.py        (about 1 minute on a laptop)
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODS = {"imaging": 4, "wearable": 6, "tabular": 3}      # features per modality
SIGMA = {"imaging": 1.0, "wearable": 1.0, "tabular": 1.2}  # measurement noise


def simulate(n, seed, wear_noise_minority=1.0):
    """Cohort with a latent severity z. Every modality is a noisy view of z.
    y = 1 if the disease progresses within 12 months. g = 1 marks a minority subgroup."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    g = (rng.random(n) < 0.3).astype(int)
    X = {}
    for m, d in MODS.items():
        sigma = np.full(n, SIGMA[m])
        if m == "wearable":
            sigma = np.where(g == 1, SIGMA[m] * wear_noise_minority, SIGMA[m])
        X[m] = z[:, None] * np.linspace(0.8, 1.2, d) + rng.normal(size=(n, d)) * sigma[:, None]
    y = (rng.random(n) < 1 / (1 + np.exp(-(3.0 * z - 0.5)))).astype(int)
    return {"z": z, "g": g, "X": X, "y": y}


def stack(X):
    return np.hstack([X[m] for m in MODS])


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def auc_ci(y, s, seed=0, B=200):
    """AUC with a 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# 1. Multimodal fusion --------------------------------------------------------
def exp_fusion(seed):
    D = simulate(3000, seed)
    tr, te = train_test_split(np.arange(3000), test_size=0.3, random_state=seed, stratify=D["y"])
    y = D["y"]
    probs = {m: lr().fit(D["X"][m][tr], y[tr]).predict_proba(D["X"][m][te])[:, 1] for m in MODS}
    A = stack(D["X"])
    out = {m + " only": probs[m] for m in MODS}
    out["late fusion (average of 3 models)"] = np.mean(list(probs.values()), axis=0)
    out["early fusion (one model, all features)"] = lr().fit(A[tr], y[tr]).predict_proba(A[te])[:, 1]
    out["reference: true severity (ceiling)"] = D["z"][te]
    return {k: auc_ci(y[te], v, seed) for k, v in out.items()}


# 2. Sparse longitudinal data -------------------------------------------------
def exp_longitudinal(seed, visits, n=600, noise=0.6):
    """Each patient is seen `visits` times at random times in [0, 4] years.
    Stage = offset a_i + rate_i * t + noise. Pooled within-patient slope estimates the
    population rate; each patient's offset is then recovered and visits are aligned."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1.2, n)
    rate = 0.6 * (1 + 0.35 * a)
    t = rng.uniform(0, 4, (n, visits))
    y = a[:, None] + rate[:, None] * t + noise * rng.normal(size=(n, visits))
    tc, yc = t - t.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    slope = (tc * yc).sum() / (tc ** 2).sum()
    offset = (y - slope * t).mean(1)
    return {"true_rate": rate.mean(), "est_rate": slope, "offset_corr": np.corrcoef(offset, a)[0, 1]}


# 3. Causal treatment effect --------------------------------------------------
def causal_once(seed, n=2000, conf=1.0):
    """Sicker patients are treated more often AND progress faster. True effect = -0.7."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    zh = z + 0.5 * rng.normal(size=n)                       # noisy baseline measurement
    t = (rng.random(n) < 1 / (1 + np.exp(-conf * z))).astype(int)
    y = 1.0 + 0.8 * z - 0.7 * t + 0.5 * rng.normal(size=n)  # change in disease score
    ps = LogisticRegression().fit(zh[:, None], t).predict_proba(zh[:, None])[:, 1]
    return {
        "naive difference": y[t == 1].mean() - y[t == 0].mean(),
        "regression adjusted (noisy baseline)": LinearRegression().fit(np.c_[t, zh], y).coef_[0],
        "inverse-probability weighting": np.mean(t * y / ps) - np.mean((1 - t) * y / (1 - ps)),
        "regression adjusted (true z, oracle)": LinearRegression().fit(np.c_[t, z], y).coef_[0],
    }


def exp_causal(reps=200):
    runs = [causal_once(s) for s in range(reps)]
    return {k: (np.mean([r[k] for r in runs]) + 0.7, np.sqrt(np.mean([(r[k] + 0.7) ** 2 for r in runs])))
            for k in runs[0]}                                # (bias, RMSE) against truth -0.7


# 4. Missing modalities -------------------------------------------------------
def mask_blocks(X, p, rng):
    """Drop whole modalities at random (set to NaN) with probability p each."""
    X = X.copy()
    col = 0
    for d in MODS.values():
        X[rng.random(len(X)) < p, col:col + d] = np.nan
        col += d
    return X


def exp_missing(seed, rates=(0, .2, .4, .6, .8)):
    D = simulate(4000, seed)
    A, y = stack(D["X"]), D["y"]
    tr, te = train_test_split(np.arange(4000), test_size=0.3, random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    a = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000)).fit(A[tr], y[tr])
    Xaug = np.vstack([A[tr]] + [mask_blocks(A[tr], p, rng) for p in (.2, .4, .6)])
    b = HistGradientBoostingClassifier(max_iter=150).fit(Xaug, np.tile(y[tr], 4))
    c = HistGradientBoostingClassifier(max_iter=150).fit(A[tr], y[tr])
    out = {"logistic regression, mean-imputed": [], "boosted trees, trained on complete data": [],
           "boosted trees, trained with modality dropout": []}
    for p in rates:
        Xm = mask_blocks(A[te], p, np.random.default_rng(100 + int(p * 10)))
        out["logistic regression, mean-imputed"].append(roc_auc_score(y[te], a.predict_proba(Xm)[:, 1]))
        out["boosted trees, trained on complete data"].append(roc_auc_score(y[te], c.predict_proba(Xm)[:, 1]))
        out["boosted trees, trained with modality dropout"].append(roc_auc_score(y[te], b.predict_proba(Xm)[:, 1]))
    return list(rates), out


# 5. Fairness -----------------------------------------------------------------
def fairness_once(seed, wear_noise=4.0, n=8000):
    D = simulate(n, seed, wear_noise_minority=wear_noise)
    A, y, g = stack(D["X"]), D["y"], D["g"]
    tr, te = train_test_split(np.arange(n), test_size=0.4, random_state=seed, stratify=y * 2 + g)
    w = np.where(g[tr] == 1, 0.5 / g[tr].mean(), 0.5 / (1 - g[tr].mean()))
    pooled = lr().fit(A[tr], y[tr]).predict_proba(A[te])[:, 1]
    rew = lr().fit(A[tr], y[tr], logisticregression__sample_weight=w).predict_proba(A[te])[:, 1]
    sep = np.zeros(len(te))
    for k in (0, 1):
        m = lr().fit(A[tr][g[tr] == k], y[tr][g[tr] == k])
        sep[g[te] == k] = m.predict_proba(A[te][g[te] == k])[:, 1]
    res = {}
    for name, p in {"pooled model": pooled, "group-reweighted training": rew, "separate model per group": sep}.items():
        a0, a1 = (roc_auc_score(y[te][g[te] == k], p[g[te] == k]) for k in (0, 1))
        tpr = [((p[g[te] == k] > .5) & (y[te][g[te] == k] == 1)).sum() / (y[te][g[te] == k] == 1).sum() for k in (0, 1)]
        res[name] = {"AUC majority": a0, "AUC minority": a1, "AUC gap": a0 - a1, "TPR gap": tpr[0] - tpr[1]}
    return res


def exp_fairness(seeds=range(10)):
    runs = [fairness_once(s) for s in seeds]
    return {k: {m: (np.mean([r[k][m] for r in runs]), np.std([r[k][m] for r in runs])) for m in runs[0][k]}
            for k in runs[0]}


# Report ----------------------------------------------------------------------
def main(seed=0, outdir="."):
    md = ["# Results (simulated data, seed %d)\n" % seed]
    f = exp_fusion(seed)
    md += ["## 1. Multimodal fusion (AUC, 95% bootstrap interval)\n", "| Model | AUC | 95% CI |", "|---|---|---|"]
    md += [f"| {k} | {v[0]:.3f} | {v[1]:.3f} to {v[2]:.3f} |" for k, v in f.items()]
    md += ["\n## 2. Sparse longitudinal data (mean of 10 seeds)\n",
           "| Visits per patient | True rate | Estimated rate | Offset correlation |", "|---|---|---|---|"]
    for k in (2, 3, 4, 6):
        r = [exp_longitudinal(s, k) for s in range(10)]
        md.append(f"| {k} | {np.mean([x['true_rate'] for x in r]):.2f} | {np.mean([x['est_rate'] for x in r]):.2f} | "
                  f"{np.mean([x['offset_corr'] for x in r]):.2f} |")
    c = exp_causal()
    md += ["\n## 3. Causal effect (true effect -0.7; 200 simulated studies)\n",
           "| Estimator | Bias | RMSE |", "|---|---|---|"]
    md += [f"| {k} | {v[0]:+.3f} | {v[1]:.3f} |" for k, v in c.items()]
    rates, mres = exp_missing(seed)
    md += ["\n## 4. Missing modalities (AUC as the share of missing modalities grows)\n",
           "| Model | " + " | ".join(f"{p:.0%}" for p in rates) + " |", "|---|" + "---|" * len(rates)]
    md += [f"| {k} | " + " | ".join(f"{x:.3f}" for x in v) + " |" for k, v in mres.items()]
    fr = exp_fairness()
    md += ["\n## 5. Fairness (minority group has a 4x noisier wearable sensor; mean and SD over 10 seeds)\n",
           "| Training | AUC majority | AUC minority | AUC gap | TPR gap |", "|---|---|---|---|---|"]
    md += [f"| {k} | " + " | ".join(f"{v[m][0]:{'+' if 'gap' in m else ''}.3f} ({v[m][1]:.3f})" for m in
           ("AUC majority", "AUC minority", "AUC gap", "TPR gap")) + " |" for k, v in fr.items()]
    text = "\n".join(md) + "\n"
    open(f"{outdir}/results.md", "w").write(text)
    print(text)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, v in mres.items():
        ax[0].plot([p * 100 for p in rates], v, marker="o", label=k)
    ax[0].set(xlabel="Missing probability per modality (%)", ylabel="AUC", title="Missing modalities")
    ax[0].legend(fontsize=7)
    ax[1].bar(range(len(c)), [v[0] for v in c.values()])
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xticks(range(len(c)), ["naive", "adjusted\n(noisy)", "IPW", "oracle"], fontsize=8)
    ax[1].set(ylabel="Bias vs true effect", title="Causal estimators")
    fig.tight_layout()
    fig.savefig(f"{outdir}/figures.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    main(a.seed, a.outdir)
