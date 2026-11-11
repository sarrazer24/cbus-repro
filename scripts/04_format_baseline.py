"""Phase 3: print the EXACT prompts (one item per question type, every mode) + a rough token/cost estimate,
so you can eyeball the option formatting and budget before spending anything. No network."""
import _common  # noqa
import argparse
from cbus import data, prompts as P

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=data.DEFAULT_DATA)
ap.add_argument("--students", type=int, default=300)
ap.add_argument("--texts-per-student", type=int, default=7)
ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (look it up on OpenRouter)")
ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
ap.add_argument("--show-text-chars", type=int, default=120)
a = ap.parse_args()
c = data.load_corpus(a.data)
tid = next(t for t in c.by_text if {i.qtype for i in c.by_text[t]} >= {"trueOrFalse", "multiChoice"})
text = c.texts[tid]
items = {}
for it in c.by_text[tid]:
    items.setdefault(it.qtype, it)
short = lambda s: s.replace(text, text[:a.show_text_chars] + " [...text truncated...]")
for qt, it in items.items():
    print(f"\n######## baseline / {qt} ########\n" + short(P.baseline_prompt(text, it)))
it = items["multiChoice"]
print("\n######## spr stage 1 ########\n" + short(P.spr_stage1(text, 4)))
print("\n######## ts stage 1 ########\n" + short(P.ts_stage1(text, it, 2)))
print("\n######## stage 2 (spr/ts/dropout) ########\n" + P.stage2_prompt(["<proposition 1>", "<proposition 2>"], it))

# rough input-token estimate (chars/4), average over texts weighted uniformly; qs/text ~4.8
import numpy as np
ids, p = data.text_weights(c, a.data)
L = np.array([len(c.texts[t]) / 4 for t in ids]); Q = np.array([len(c.by_text[t]) for t in ids])
N = a.students * a.texts_per_student
exp_len, exp_q = float((p * L).sum()), float((p * Q).sum())
base_in = N * exp_q * (exp_len + 120)                    # text re-sent for every question
spr_in = N * (exp_len + 150) + N * exp_q * 250           # one reading + short stage 2 per question
ts_in = N * exp_q * (exp_len + 250) + N * exp_q * 250    # reading per question + stage 2
drop_in = spr_in
out_tok = dict(baseline=N * exp_q * 15, spr=N * 80 + N * exp_q * 15, ts=N * exp_q * 95, spr_dropout=N * 200 + N * exp_q * 15)
ins = dict(baseline=base_in, spr=spr_in, ts=ts_in, spr_dropout=drop_in)
print(f"\n== rough estimate for {a.students} students x {a.texts_per_student} texts ({N} student-texts, ~{exp_q:.1f} q/text, ~{exp_len:.0f} tokens/text) ==")
tot_i = tot_o = 0
for k in ins:
    cost = "" if a.price_in is None else f"  ~${ins[k]/1e6*a.price_in + out_tok[k]/1e6*(a.price_out or 0):.2f}"
    print(f"{k:12s} in {ins[k]/1e6:6.1f}M  out {out_tok[k]/1e6:5.2f}M{cost}")
    tot_i += ins[k]; tot_o += out_tok[k]
if a.price_in is not None:
    print(f"all four: ~${tot_i/1e6*a.price_in + tot_o/1e6*(a.price_out or 0):.2f}  (+/- 30%; Norwegian text tokenizes worse than English)")
print("(estimate is rough; a 20-student pilot gives you the real tokens from meta.json -> usage)")
