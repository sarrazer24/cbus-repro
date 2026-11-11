"""Phase 1/4: metrics table (mean acc, score SD, gap, JSD, ECE, Pearson, Spearman) for my runs next to
the authors' precomputed rows, plus a histogram plot and CSV.  JSD of MY runs is only comparable between
my own runs (synthetic exposure); item-level metrics are comparable to the paper (read relative to a same-N random control)."""
import _common  # noqa
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cbus import data, metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*", help="run dirs (default: runs/*)")
ap.add_argument("--data", default=data.DEFAULT_DATA)
ap.add_argument("--authors", nargs="*", default=[
    "random", "llm_baseline-llama-3.3-70b-instruct", "cbus_ts-c2-llama-3.3-70b-instruct",
    "cbus_spr-c4-llama-3.3-70b-instruct", "cbus_spr_dropout-c4-llama-3.3-70b-instruct",
    "llm_baseline-gemini-3.5-flash-lite", "cbus_spr-c4-gemini-3.5-flash-lite"])
ap.add_argument("--no-authors", action="store_true")
ap.add_argument("--out", default="results/table.csv")
ap.add_argument("--plot", default="results/score_hist.png")
a = ap.parse_args()

ht, hh = data.human_stats("test", a.data), data.human_hist("test", a.data)
rows, hists = [], {}
mu, sd = M.hist_stats(hh)
rows.append(dict(source="authors", name="Humans (test, ground truth)", mean_acc=mu, score_sd=sd, n_students=int(hh.n_students.sum())))
hists["Humans (test)"] = hh
tr, trh = data.human_stats("train", a.data), data.human_hist("train", a.data)
rows.append(dict(source="authors", name="Humans (held-out)", n_students=int(trh.n_students.sum()), **M.evaluate(tr, trh, ht, hh)))
if not a.no_authors:
    for m in a.authors:
        res = [M.evaluate(*data.load_authors_run(m, r, a.data), ht, hh) for r in (1, 2, 3)]
        row = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
        rows.append(dict(source="authors (mean of 3)", name=m, n_students=1198, **row))
        hists[m] = data.load_authors_run(m, 1, a.data)[1]
for d in (a.runs or sorted(glob.glob("runs/*"))):
    if not os.path.exists(os.path.join(d, "question_stats.csv")):
        continue
    qs, h = pd.read_csv(os.path.join(d, "question_stats.csv")), pd.read_csv(os.path.join(d, "score_hist.csv"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    ev = M.evaluate(qs, h, ht, hh)
    tot = meta["n_answered"] + meta["n_dropped_answers_or_units"]
    rows.append(dict(source="mine", name=meta["name"], n_students=meta["n_students_scored"],
                     dropped_pct=100 * meta["n_dropped_answers_or_units"] / max(tot, 1), **ev))
    hists[meta["name"]] = h
df = pd.DataFrame(rows)
cols = ["source", "name", "n_students", "mean_acc", "score_sd", "gap", "jsd", "ece", "pearson", "spearman", "n_responses", "dropped_pct"]
df = df[[c for c in cols if c in df.columns]]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 48)
print(df.round(3).to_string(index=False))
print("\nNOTE: 'mine' JSD uses synthetic exposure -> compare only among my own runs. Item metrics: compare to a same-N random control.")
fig, ax = plt.subplots(figsize=(8, 4))
for k, h in hists.items():
    p = h.sort_values("bin_index").n_students.to_numpy(float); p = 100 * p / p.sum()
    ax.plot(((h.bin_lo + h.bin_hi) / 2).sort_values(), p, label=k[:40], lw=2 if k.startswith("Humans") else 1)
ax.set_xlabel("fraction correct (per student)"); ax.set_ylabel("% of students"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(a.plot, dpi=150)
print("wrote", a.out, a.plot)
