import numpy as np
from scipy.io import loadmat

d = loadmat("s1.mat")
fs = float(d["fs"].squeeze())
trig = d["trig"].squeeze()
y = d["y"]

print("fs:", fs, "| duration (s):", round(len(trig) / fs, 1))
vals, counts = np.unique(trig, return_counts=True)
print("trigger values and counts:", dict(zip(vals.tolist(), counts.tolist())))

# first 15 event onsets: sample index, value, and gap since previous onset (ms)
onsets = np.flatnonzero((trig[1:] != trig[:-1]) & (trig[1:] != 0)) + 1
print("number of onsets:", len(onsets))
for i in onsets[:15]:
    print(i, trig[i])
print("gaps between onsets (ms):", (np.diff(onsets[:15]) / fs * 1000).round(0))

print("channel std (uV?):", y.std(axis=0).round(1))