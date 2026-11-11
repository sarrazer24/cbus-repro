"""Phase 1: reproduce Table 4 rows from the authors' precomputed aggregates with MY metric code,
plus sanity analyses (what the metrics reward). Offline, no LLM, no API key."""
import _common  # noqa
import argparse
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from cbus import data, metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=data.DEFAULT_DATA)
a = ap.parse_args()
root = a.data

ht, hh = data.human_stats("test", root), data.human_hist("test", root)
tr, trh = data.human_stats("train", root), data.human_hist("train", root)

ROWS = [("Random", "random"),
        ("Llama-3.3-70B baseline", "llm_baseline-llama-3.3-70b-instruct"),
        ("Llama CBUS-TS (C=2)", "cbus_ts-c2-llama-3.3-70b-instruct"),
        ("Llama CBUS-SPR (C=4)", "cbus_spr-c4-llama-3.3-70b-instruct"),
        ("Llama answer-noising", "answer_noising-llama-3.3-70b-instruct"),
        ("Llama random dropout", "cbus_spr_dropout-c4-llama-3.3-70b-instruct"),
        ("Gemini-3.5-FL CBUS-SPR", "cbus_spr-c4-gemini-3.5-flash-lite")]
KEYS = ["gap", "jsd", "ece", "pearson", "spearman"]
print("== Table 4 check (mean of 3 runs) ==")
print(f"{'row':28s} " + " ".join(f"{k:>8s}" for k in KEYS))
hr = M.evaluate(tr, trh, ht, hh)
print(f"{'Held-out humans':28s} " + " ".join(f"{hr[k]:8.3f}" for k in KEYS), "  (paper: .026 .006 .027 .895 .845)")
for label, m in ROWS:
    vals = []
    for r in (1, 2, 3):
        qs, h = data.load_authors_run(m, r, root)
        vals.append(M.evaluate(qs, h, ht, hh))
    print(f"{label:28s} " + " ".join(f"{np.mean([v[k] for v in vals]):8.3f}" for k in KEYS))
print("paper: Llama baseline .279 .679 .285 .241 .179 | TS .238 .475 .244 .455 .304 | SPR .037 .093 .069 .301 .297")

print("\n== Sanity 1: per-student score SD from histogram midpoints (approximate) ==")
for label, m in [("human test", None), ("random", "random"), ("Llama baseline", "llm_baseline-llama-3.3-70b-instruct"),
                 ("Llama CBUS-SPR", "cbus_spr-c4-llama-3.3-70b-instruct"), ("Llama CBUS-TS", "cbus_ts-c2-llama-3.3-70b-instruct")]:
    h = hh if m is None else data.load_authors_run(m, 1, root)[1]
    mu, sd = M.hist_stats(h)
    print(f"{label:18s} mean~{mu:.3f}  SD~{sd:.3f}")

print("\n== Sanity 2: how much of item-level correlation is just question format? ==")
corpus = data.load_corpus(root)
typ = {k: it.qtype for k, it in corpus.items.items()}
chance = {}
for k, it in corpus.items.items():
    chance[k] = 0.5 if it.qtype == "trueOrFalse" else (1 / len(it.options) if it.qtype == "multiChoice" else 0.5 ** len(it.options))
dt, dtr = M.difficulty(ht), M.difficulty(tr)
mean_by_type = pd.Series({k: dtr[k] for k in dtr.index}).groupby(lambda k: typ[k]).mean()
type_pred = pd.Series({k: mean_by_type[typ[k]] for k in dt.index})
chance_pred = pd.Series({k: chance[k] for k in dt.index})
for name, pred in [("type-only (fit on train, score on test)", type_pred), ("chance-level (1/options)", chance_pred)]:
    print(f"{name:42s} Pearson {pearsonr(dt, pred.loc[dt.index])[0]:.3f}  Spearman {spearmanr(dt, pred.loc[dt.index])[0]:.3f}")
q, h = data.load_authors_run("cbus_ts-c2-llama-3.3-70b-instruct", 1, root)
sd_ = M.difficulty(q); i = dt.index.intersection(sd_.index)
print(f"{'Llama CBUS-TS run 1 (for reference)':42s} Pearson {pearsonr(dt.loc[i], sd_.loc[i])[0]:.3f}  Spearman {spearmanr(dt.loc[i], sd_.loc[i])[0]:.3f}")
print("by type, mean human difficulty:", {k: round(float(v), 3) for k, v in dt.groupby(lambda k: typ[k]).mean().items()})

print("\n== Sanity 3: JSD noise floor for an IDENTICAL population of n students (resampling the test histogram) ==")
p = hh.sort_values("bin_index").n_students.to_numpy(float); p /= p.sum()
rng = np.random.default_rng(0)
for n in (50, 100, 200, 500, 1198):
    v = []
    for _ in range(300):
        h2 = hh.copy(); h2["n_students"] = rng.multinomial(n, p)
        h1 = hh.copy(); h1["n_students"] = p * 1e6   # reference = the observed distribution itself
        v.append(M.jsd(h1, h2))
    print(f"n={n:5d}: mean JSD {np.mean(v):.3f}")
print("(n resampled students vs the observed 1,198-student distribution; Monte Carlo, 300 reps)")
