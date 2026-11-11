"""Phase 7: plot my capacity sweep (SPR C in 2,4,6,8,12 ...) from runs/spr-c*-... dirs, with the same-N random control as a dotted line."""
import _common  # noqa
import argparse, glob, json, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cbus import data, metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="runs/spr-c*-n300-s0", help="run dirs of ONE model/N (glob)")
ap.add_argument("--control", default=None, help="random-control run dir at the same N")
ap.add_argument("--baseline", default=None, help="persona baseline run dir (optional)")
ap.add_argument("--data", default=data.DEFAULT_DATA)
ap.add_argument("--out", default="results/capacity_sweep.png")
a = ap.parse_args()
ht, hh = data.human_stats("test", a.data), data.human_hist("test", a.data)
ev = lambda d: M.evaluate(pd.read_csv(f"{d}/question_stats.csv"), pd.read_csv(f"{d}/score_hist.csv"), ht, hh)
rows = []
for d in glob.glob(a.glob):
    m = re.search(r"-c(\d+)-", d)
    if m and os.path.exists(f"{d}/question_stats.csv"):
        rows.append(dict(C=int(m.group(1)), **ev(d)))
df = pd.DataFrame(rows).sort_values("C")
print(df.round(3).to_string(index=False))
df.to_csv(a.out.replace(".png", ".csv"), index=False)
metrics = [("gap", "abs gap (lower better)"), ("jsd", "JSD (lower better)"), ("pearson", "Pearson (higher better)"), ("score_sd", "score SD (human ~0.17)")]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
ctrl = ev(a.control) if a.control else None
base = ev(a.baseline) if a.baseline else None
for ax, (k, t) in zip(axs, metrics):
    ax.plot(df.C, df[k], "o-", label="SPR (mine)")
    if ctrl: ax.axhline(ctrl[k], ls=":", c="gray", label="random control")
    if base: ax.axhline(base[k], ls="--", c="red", label="persona baseline")
    ax.set_title(t, fontsize=9); ax.set_xlabel("capacity C")
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(a.out, dpi=150); print("wrote", a.out)
