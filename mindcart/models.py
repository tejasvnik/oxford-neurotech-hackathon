"""Model comparison: A (product + prior preferences) vs B (EEG) vs C (A + EEG).

    python -m mindcart.models sessions/P01 sessions/P02 --items stimuli/items.csv

Within-participant evaluation (EEG differs a lot between people):
  - repeated stratified 5-fold CV (10 repeats); one feature vector per trial, trials never
    overlap in time, and scaling is fitted inside each training fold -> no leakage
  - metrics: balanced accuracy, F1, vs majority-class baseline
  - permutation test (labels shuffled) -> p-value and 95th-percentile chance level
  - C - A: per-repeat difference = does EEG add anything on top of product data?
Outputs results.csv + results.png next to the first session.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META_NUM = ["price", "luminance", "contrast", "colourfulness", "prior_category", "prior_colour"]
META_CAT = ["category", "colour"]


def load_prefs(sdir: Path):
    """prefs.csv: type,value,score  e.g.  category,jacket,4  /  colour,black,5  (asked BEFORE the session)."""
    p = sdir / "prefs.csv"
    if not p.exists():
        return {}, {}
    df = pd.read_csv(p)
    get = lambda t: dict(zip(df[df.type == t].value.astype(str).str.lower(), df[df.type == t].score))
    return get("category"), get("colour")


def build_table(sdir: Path, items: pd.DataFrame, use_full=False):
    tr = pd.read_csv(sdir / "trials.csv")
    tr = tr[tr["ok"] & tr["label"].notna()].copy()
    tr["item_id"] = tr["item_id"].astype(str)
    it = items.copy(); it["item_id"] = it["item_id"].astype(str)
    df = tr.merge(it, on="item_id", how="left")
    pc, pcol = load_prefs(sdir)
    df["prior_category"] = df.get("category", pd.Series("", index=df.index)).astype(str).str.lower().map(pc)
    df["prior_colour"] = df.get("colour", pd.Series("", index=df.index)).astype(str).str.lower().map(pcol)
    meta_cols = [c for c in META_NUM if c in df and df[c].notna().any()]
    Xa = df[meta_cols].astype(float).fillna(df[meta_cols].astype(float).mean())
    for c in META_CAT:
        if c in df:
            Xa = pd.concat([Xa, pd.get_dummies(df[c].astype(str), prefix=c, dtype=float)], axis=1)
    eeg_cols = [c for c in df if c.startswith("full_" if use_full else "f_")]
    Xb = df[eeg_cols].astype(float)
    return Xa.to_numpy(), Xb.to_numpy(), df["label"].astype(int).to_numpy(), list(Xa.columns), eeg_cols


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def cv_scores(X, y, model, n_rep=10, seed=0):
    k = int(min(5, np.bincount(y).min()))
    if k < 2:
        return None
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_rep, random_state=seed)
    bacc, f1 = np.zeros(n_rep), np.zeros(n_rep)
    for r in range(n_rep):
        pred = np.empty_like(y)
        splits = list(rskf.split(X, y))[r * k:(r + 1) * k]
        for tr, te in splits:
            m = model().fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        bacc[r] = balanced_accuracy_score(y, pred); f1[r] = f1_score(y, pred, average="macro")
    return bacc, f1


def evaluate_participant(sdir, items, n_perm=500, use_full=False):
    Xa, Xb, y, a_cols, b_cols = build_table(Path(sdir), items, use_full)
    sets = {"A_product": Xa, "B_EEG": Xb, "C_combined": np.hstack([Xa, Xb])}
    res = {"participant": Path(sdir).name, "n_trials": len(y), "n_like": int(y.sum()),
           "majority_bacc": 0.5, "majority_acc": float(max(y.mean(), 1 - y.mean()))}
    per_rep = {}
    for name, X in sets.items():
        if X.shape[1] == 0:
            continue
        sc = cv_scores(X, y, clf)
        if sc is None:
            continue
        per_rep[name] = sc[0]
        k = int(min(5, np.bincount(y).min()))
        score, perm, p = permutation_test_score(clf(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0),
                                                scoring="balanced_accuracy", n_permutations=n_perm, random_state=0)
        res.update({f"{name}_bacc": sc[0].mean(), f"{name}_bacc_sd": sc[0].std(), f"{name}_f1": sc[1].mean(),
                    f"{name}_p": p, f"{name}_chance95": float(np.percentile(perm, 95)), f"{name}_nfeat": X.shape[1]})
    if "A_product" in per_rep and "C_combined" in per_rep:
        d = per_rep["C_combined"] - per_rep["A_product"]
        res["C_minus_A"] = float(d.mean()); res["C_beats_A_frac"] = float((d > 0).mean())
    return res


def plot(results, path):
    models = ["A_product", "B_EEG", "C_combined"]
    labels = ["A: product", "B: EEG", "C: combined"]
    fig, ax = plt.subplots(figsize=(1.8 + 2.2 * len(results), 4))
    w = 0.25
    for j, r in enumerate(results):
        for i, m in enumerate(models):
            if f"{m}_bacc" not in r:
                continue
            x = j + (i - 1) * w
            ax.bar(x, r[f"{m}_bacc"], w * 0.9, yerr=r[f"{m}_bacc_sd"], color=["#3b82f6", "#22c55e", "#f59e0b"][i],
                   label=labels[i] if j == 0 else None)
            ax.plot([x - w / 2, x + w / 2], [r[f"{m}_chance95"]] * 2, "k--", lw=1)
            if r[f"{m}_p"] < 0.05:
                ax.text(x, r[f"{m}_bacc"] + r[f"{m}_bacc_sd"] + 0.02, "*", ha="center", fontsize=14)
    ax.axhline(0.5, color="grey", lw=1)
    ax.set_xticks(range(len(results)), [r["participant"] for r in results])
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("Balanced accuracy (CV)")
    ax.set_title("MindCart: does EEG add to product data?\n(dashed = 95% chance from permutations, * p<.05)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dirs", nargs="+")
    ap.add_argument("--items", default="stimuli/items.csv")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--full", action="store_true", help="use all 24 EEG features instead of the reduced 8")
    a = ap.parse_args()
    items = pd.read_csv(a.items)
    results = [evaluate_participant(s, items, a.n_perm, a.full) for s in a.session_dirs]
    df = pd.DataFrame(results)
    out = Path(a.session_dirs[0]).parent
    df.to_csv(out / "results.csv", index=False)
    plot(results, out / "results.png")
    pd.set_option("display.width", 200)
    cols = ["participant", "n_trials"] + [c for c in df if c.endswith("_bacc") or c.endswith("_p")] + \
           [c for c in ("C_minus_A", "C_beats_A_frac") if c in df]
    print(df[cols].round(3).to_string(index=False))
    print(f"\nSaved {out / 'results.csv'} and {out / 'results.png'}")


if __name__ == "__main__":
    main()
