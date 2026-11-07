"""Epoch a MindCart session and extract one EEG feature vector per product trial.

    python -m mindcart.features sessions/P01 [--like extreme|median]

Per trial:
  baseline window = fixation, -0.9 .. -0.1 s before image onset
  analysis window = 0.5 .. 4.0 s after onset (skips visual onset response / first saccades)
  band power (Welch) in theta 4-8, alpha 8-13, beta 13-30 Hz, as log change vs baseline
Trials whose analysis window exceeds the artefact threshold (blinks/movement) are flagged.

Writes sessions/P01/trials.csv: trial, item_id, rating, label, ok, <reduced feats>, <full feats>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_pipeline import PipelineConfig, filter_offline  # noqa: E402

FS = 250.0
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
BASE_WIN = (-0.9, -0.1)
STIM_WIN = (0.5, 4.0)
POSTERIOR = ["Pz", "PO7", "Oz", "PO8"]
CENTRAL = ["C3", "Cz", "C4"]


def load_session(sdir: Path):
    eeg = pd.read_csv(sdir / "eeg.csv")
    mk = pd.read_csv(sdir / "markers.csv", dtype={"marker": str})
    t = eeg["lsl_time"].to_numpy()
    x = eeg.drop(columns="lsl_time").to_numpy(float)
    fs = 1.0 / np.median(np.diff(t))
    if abs(fs - FS) > 5:
        print(f"WARNING: estimated sample rate {fs:.1f} Hz (expected {FS})")
    # trials from markers
    rows = {}
    for lt, m in zip(mk["lsl_time"], mk["marker"]):
        p = m.split("|")
        if p[0] == "img":
            rows.setdefault(int(p[1]), {})["onset"] = lt
            rows[int(p[1])]["item_id"] = p[2]
        elif p[0] == "rate":
            rows.setdefault(int(p[1]), {})["rating"] = float(p[3]) if p[3] not in ("", "None") else np.nan
    trials = pd.DataFrame([{"trial": k, **v} for k, v in sorted(rows.items())])
    return t, x, trials


def band_logpower(seg, fs):
    f, p = welch(seg.T, fs=fs, nperseg=min(len(seg), int(fs)), noverlap=int(fs) // 2)
    return {b: np.log(p[:, (f >= lo) & (f < hi)].mean(axis=1) + 1e-12) for b, (lo, hi) in BANDS.items()}


def extract(sdir, like="extreme", ptp_uv=200.0):
    sdir = Path(sdir)
    cfg = PipelineConfig()
    t, x, trials = load_session(sdir)
    y = filter_offline(x, cfg)  # zero-phase OK offline
    ch = cfg.channel_names
    idx = {c: i for i, c in enumerate(ch)}
    out = []
    for _, tr in trials.iterrows():
        if "onset" not in tr or np.isnan(tr.get("onset", np.nan)):
            continue
        i0 = np.searchsorted(t, tr["onset"])
        sl = lambda w: slice(i0 + int(w[0] * FS), i0 + int(w[1] * FS))
        base, stim = y[sl(BASE_WIN)], y[sl(STIM_WIN)]
        if len(base) < 0.7 * FS or len(stim) < 3 * FS:
            continue
        bp_b, bp_s = band_logpower(base, FS), band_logpower(stim, FS)
        d = {b: bp_s[b] - bp_b[b] for b in BANDS}  # log change vs fixation
        rec = {"trial": tr["trial"], "item_id": tr["item_id"], "rating": tr.get("rating", np.nan),
               "max_ptp_uv": float(np.ptp(stim, axis=0).max())}
        rec["ok"] = rec["max_ptp_uv"] < ptp_uv
        # reduced, hypothesis-driven set (~8 features; safe for ~60 trials)
        mean = lambda b, chans: float(np.mean([d[b][idx[c]] for c in chans]))
        rec.update({
            "f_fz_theta": float(d["theta"][idx["Fz"]]),
            "f_fz_beta": float(d["beta"][idx["Fz"]]),
            "f_post_alpha": mean("alpha", POSTERIOR),
            "f_central_alpha": mean("alpha", CENTRAL),
            "f_asym_c_alpha": float(bp_s["alpha"][idx["C4"]] - bp_s["alpha"][idx["C3"]]),
            "f_asym_po_alpha": float(bp_s["alpha"][idx["PO8"]] - bp_s["alpha"][idx["PO7"]]),
            "f_beta_alpha": float(np.mean(d["beta"] - d["alpha"])),
            "f_post_theta": mean("theta", POSTERIOR),
        })
        for b in BANDS:  # full 24-feature set
            for c in ch:
                rec[f"full_{c}_{b}"] = float(d[b][idx[c]])
        out.append(rec)
    df = pd.DataFrame(out)
    df["label"] = make_labels(df["rating"], like)
    df.to_csv(sdir / "trials.csv", index=False)
    return df


def make_labels(r, mode="extreme"):
    """1 = like, 0 = dislike, NaN = excluded."""
    if mode == "extreme":
        return np.where(r >= 4, 1.0, np.where(r <= 2, 0.0, np.nan))
    med = np.nanmedian(r)
    return np.where(r > med, 1.0, np.where(r < med, 0.0, np.nan))  # ties at median dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dirs", nargs="+")
    ap.add_argument("--like", choices=["extreme", "median"], default="extreme")
    ap.add_argument("--ptp", type=float, default=200.0, help="artefact threshold, µV peak-to-peak")
    a = ap.parse_args()
    for s in a.session_dirs:
        df = extract(s, a.like, a.ptp)
        lab = df["label"]
        print(f"{s}: {len(df)} trials | artefact-rejected {(~df['ok']).sum()} | "
              f"like {int((lab == 1).sum())} / dislike {int((lab == 0).sum())} / excluded {int(lab.isna().sum())}")


if __name__ == "__main__":
    main()
