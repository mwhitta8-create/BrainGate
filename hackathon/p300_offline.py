#!/usr/bin/env python3
"""
Offline P300 analysis for Unicorn-style .mat files.

Expected file contents:
    fs    sampling rate (Hz)
    y     EEG, shape (n_samples, n_channels)
    trig  event channel, shape (n_samples, 1) or (n_samples,)
          here: 1 = target flash, -1 = non-target flash, 0 = no event

    python p300_offline.py yourfile.mat
    python p300_offline.py yourfile.mat --reject 120 --channels Fz C3 Cz C4 Pz PO7 Oz PO8

What it does:
  1. band-pass filters the EEG (0.5-20 Hz, zero-phase)
  2. cuts an epoch around every flash (-100 ms .. +800 ms), baseline-corrects it,
     drops epochs with huge artifacts
  3. plots the average target vs non-target response for every channel
  4. trains / cross-validates a shrinkage LDA classifier (target vs non-target)
     and reports AUC for each fold

Needs: numpy, scipy, matplotlib, scikit-learn
"""
import argparse

import numpy as np
from scipy import signal
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

DEFAULT_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]  # ASSUMED Unicorn order


def load(path):
    d = loadmat(path)
    fs = float(np.squeeze(d["fs"]))
    y = np.asarray(d["y"], dtype=float)
    trig = np.asarray(d["trig"]).squeeze()
    if y.shape[0] != trig.shape[0] and y.shape[1] == trig.shape[0]:
        y = y.T
    return fs, y, trig


def bandpass(y, fs, lo=0.5, hi=20.0):
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, y, axis=0)       # zero-phase (offline only)


def find_onsets(trig):
    """Sample index where each event starts, and whether it is a target (trig == 1)."""
    nz = trig != 0
    starts = np.flatnonzero(nz & ~np.r_[False, nz[:-1]] | (nz & (trig != np.r_[0, trig[:-1]])))
    return starts, trig[starts] == 1


def make_epochs(y, fs, onsets, tmin=-0.1, tmax=0.8, reject_uv=100.0):
    n0, n1 = int(round(tmin * fs)), int(round(tmax * fs))
    keep, epochs = [], []
    for k, s in enumerate(onsets):
        if s + n0 < 0 or s + n1 > len(y):
            continue
        ep = y[s + n0: s + n1].copy()                # (time, channels)
        ep -= ep[: -n0].mean(axis=0)                 # baseline = pre-stimulus mean
        if np.ptp(ep, axis=0).max() > reject_uv:
            continue
        epochs.append(ep)
        keep.append(k)
    t = np.arange(n0, n1) / fs * 1000.0              # ms
    return np.array(epochs), np.array(keep), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matfile")
    ap.add_argument("--reject", type=float, default=100.0,
                    help="drop epochs whose peak-to-peak exceeds this many uV in any channel")
    ap.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    ap.add_argument("--out", default="p300_erp.png", help="where to save the ERP figure")
    args = ap.parse_args()

    fs, y, trig = load(args.matfile)
    n_ch = y.shape[1]
    names = (args.channels + [f"CH{i+1}" for i in range(n_ch)])[:n_ch]
    print(f"{args.matfile}: {y.shape[0]} samples, {n_ch} channels, fs={fs:g} Hz, "
          f"{y.shape[0]/fs:.0f} s")

    yf = bandpass(y, fs)
    onsets, is_target = find_onsets(trig)
    epochs, keep, t = make_epochs(yf, fs, onsets, reject_uv=args.reject)
    labels = is_target[keep]
    print(f"events: {len(onsets)} (targets {is_target.sum()}, non-targets {(~is_target).sum()})")
    print(f"epochs kept after artifact rejection: {len(epochs)} "
          f"(targets {labels.sum()}, non-targets {(~labels).sum()}); "
          f"dropped {len(onsets) - len(epochs)}")
    if labels.sum() < 10:
        raise SystemExit("Too few target epochs left. Try a larger --reject value.")

    # ---- ERP plot: average target vs non-target per channel ----
    tgt = epochs[labels].mean(axis=0)                # (time, channels)
    non = epochs[~labels].mean(axis=0)
    cols = 4
    rows = int(np.ceil(n_ch / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), sharex=True, sharey=True)
    for c, ax in enumerate(np.atleast_1d(axes).ravel()):
        if c >= n_ch:
            ax.axis("off")
            continue
        ax.plot(t, tgt[:, c], color="tab:red", label=f"target (n={labels.sum()})")
        ax.plot(t, non[:, c], color="tab:blue", label=f"non-target (n={(~labels).sum()})")
        ax.axvspan(300, 500, color="gold", alpha=0.15)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(names[c])
        ax.set_xlabel("ms after flash")
        ax.set_ylabel("uV")
    np.atleast_1d(axes).ravel()[0].legend(fontsize=8)
    fig.suptitle("Average response: target vs non-target (yellow = 300-500 ms)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved ERP figure to {args.out}")

    # strongest difference
    diff = tgt - non
    win = (t >= 250) & (t <= 600)
    best = int(np.argmax(diff[win].max(axis=0)))
    peak_t = t[win][np.argmax(diff[win, best])]
    print(f"largest target-minus-non-target bump: {names[best]} at {peak_t:.0f} ms "
          f"({diff[win, best].max():.1f} uV)")

    # ---- classifier: shrinkage LDA, contiguous-block cross-validation ----
    # features: 0..800 ms after the flash, downsampled to ~50 Hz, all channels
    post = t >= 0
    step = max(1, int(round(fs / 50)))
    X = epochs[:, post, :][:, ::step, :].reshape(len(epochs), -1)
    aucs = []
    for train, test in KFold(n_splits=5, shuffle=False).split(X):   # no shuffling: avoids leakage
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X[train], labels[train])
        aucs.append(roc_auc_score(labels[test], clf.decision_function(X[test])))
    print("per-fold AUC:", np.round(aucs, 3), f"| mean {np.mean(aucs):.3f}")
    print("(0.5 = chance, 1.0 = perfect single-flash detection)")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
