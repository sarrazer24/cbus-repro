"""Evaluation metrics, implemented from the paper's Appendix E + the conventions I verified against Table 4:
 - abs gap  = |response-weighted mean accuracy difference|
 - JSD      = Jensen-Shannon DIVERGENCE (log base 2, not the sqrt distance) between 20-bin per-student-score histograms
 - ECE      = 6 equal-width bins over HUMAN item difficulty, weight = share of questions, bin value = mean item difficulty
 - Pearson / Spearman between per-question difficulties n_correct/(n_correct+n_incorrect)"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

NBINS = 20


def mean_acc(qs: pd.DataFrame) -> float:
    return float(qs.n_correct.sum() / (qs.n_correct.sum() + qs.n_incorrect.sum()))


def difficulty(qs: pd.DataFrame) -> pd.Series:
    n = qs.n_correct + qs.n_incorrect
    d = qs.n_correct / n.where(n > 0)
    d.index = list(zip(qs.sanity_text_id, qs.question_id))
    return d.dropna()


def _p(h: pd.DataFrame) -> np.ndarray:
    c = h.sort_values("bin_index").n_students.to_numpy(float)
    return c / c.sum()


def jsd(h1: pd.DataFrame, h2: pd.DataFrame) -> float:
    p, q = _p(h1), _p(h2)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def hist_stats(h: pd.DataFrame):
    """(mean, SD) of per-student score estimated from bin midpoints (approximate)."""
    h = h.sort_values("bin_index")
    mid = ((h.bin_lo + h.bin_hi) / 2).to_numpy(float)
    p = _p(h)
    mu = float((p * mid).sum())
    return mu, float(np.sqrt((p * (mid - mu) ** 2).sum()))


def student_hist(scores) -> pd.DataFrame:
    """20-bin histogram of per-student accuracy over [0,1] (last bin closed)."""
    cnt, edges = np.histogram(np.asarray(scores, float), bins=NBINS, range=(0, 1))
    return pd.DataFrame(dict(bin_index=range(NBINS), bin_lo=edges[:-1], bin_hi=edges[1:], n_students=cnt))


def ece(hum_d: pd.Series, sim_d: pd.Series, m: int = 6) -> float:
    idx = hum_d.index.intersection(sim_d.index)       # questions with >=1 simulated response
    h, s = hum_d.loc[idx].to_numpy(), sim_d.loc[idx].to_numpy()
    b = np.minimum((h * m).astype(int), m - 1)         # equal-width bins over human difficulty
    out = 0.0
    for i in range(m):
        k = b == i
        if k.any():
            out += (k.sum() / len(h)) * abs(h[k].mean() - s[k].mean())
    return float(out)


def evaluate(sim_qs: pd.DataFrame, sim_hist: pd.DataFrame, hum_qs: pd.DataFrame, hum_hist: pd.DataFrame) -> dict:
    hd, sd = difficulty(hum_qs), difficulty(sim_qs)
    idx = hd.index.intersection(sd.index)
    r = dict(
        mean_acc=mean_acc(sim_qs),
        score_sd=hist_stats(sim_hist)[1],
        gap=abs(mean_acc(hum_qs) - mean_acc(sim_qs)),
        jsd=jsd(hum_hist, sim_hist),
        ece=ece(hd, sd),
        pearson=float(pearsonr(hd.loc[idx], sd.loc[idx])[0]) if len(idx) > 2 else float("nan"),
        spearman=float(spearmanr(hd.loc[idx], sd.loc[idx])[0]) if len(idx) > 2 else float("nan"),
        n_items=len(idx),
        n_responses=int(sim_qs.n_correct.sum() + sim_qs.n_incorrect.sum()),
    )
    return r
