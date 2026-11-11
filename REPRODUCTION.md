# REPRODUCTION.md: step-by-step protocol

All commands run from the repo root. `authors_data/` = clone of the authors' repo (see README). Write the pass/fail criteria (Phase 5) **before** the main runs and do not change them afterwards.

## Phase 0. Setup
```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/iai-group/emnlp2026-simreading authors_data
```
Paths can be overridden with `--data` or the env var `SIMREADING_DATA`.

## Phase 1. Offline reproduction of Table 4 (free)
```bash
python scripts/01_check_metrics.py
```
Expected (mean of the authors' 3 runs, my metric code): Llama-3.3-70B baseline gap .279 / JSD .679 / ECE .285 / Pearson .241 / Spearman .179; CBUS-SPR (C=4) .037 / .093 / .069 / .301 / .297; CBUS-TS (C=2) .238 / .475 / .244 / .455 / .304; held-out humans .026 / .006 / .027 / .895 / .845. If these do not match, stop: your metric code is wrong.
Metric conventions that matter: abs. gap on response-weighted accuracy; JSD = divergence, log base 2 (not the square root); ECE = 6 equal-width bins over *human* item difficulty, weighted by share of questions; item metrics on per-question n_correct/(n_correct+n_incorrect).

## Phase 2. Metric sanity analyses (same script)
The script also prints: (a) per-student score SD from histogram midpoints (humans about .17; Llama baseline about .04; CBUS-SPR about .11); (b) a type-only predictor fitted on the train half and scored on test (Pearson .523 / Spearman .339) and a chance-level predictor (.458 / .370), i.e. item correlation largely reflects question format; (c) the JSD noise floor for an identical population of n students (n=50: .068, 100: .034, 200: .018, 500: .007, 1198: .003).

## Phase 3. Mock smoke test and prompt check (free)
```bash
python scripts/02_run_simulation.py --mode spr --mock --students 20      # also try baseline / ts / spr_dropout
python scripts/02_run_simulation.py --mode random --students 300
python scripts/03_evaluate.py
python scripts/04_format_baseline.py --students 300 --price-in <usd/M> --price-out <usd/M>
```
The mock LLM answers randomly, so its metrics must look like the random control. Read the printed prompts: the options format and the text-then-question layout are yours to confirm.

## Phase 4. Pilot: 20 students, real model
```bash
export OPENROUTER_API_KEY=...
python scripts/02_run_simulation.py --mode baseline --model meta-llama/llama-3.3-70b-instruct --students 20
python scripts/02_run_simulation.py --mode spr      --model meta-llama/llama-3.3-70b-instruct --students 20
python scripts/02_run_simulation.py --mode ts       --model meta-llama/llama-3.3-70b-instruct --students 20
```
Check in `runs/*/meta.json`: drop rate (target: under 2%; if TS drops much more, see pitfalls), real token usage (rescale the cost estimate), and open the cache or add a debug print to read a few raw stage-1 and stage-2 replies. Do not draw conclusions from 20 students.

## Phase 5. Pre-registered criteria (write before Phase 6)
| id | criterion | needs |
|---|---|---|
| C1 | baseline mean accuracy at least 0.90 and score SD at most 0.06 | baseline run |
| C2 | SPR abs. gap at most 0.15 and SPR JSD at most half of the baseline JSD (my runs only) | baseline + SPR |
| C3 | TS abs. gap stays above 0.15 and TS item Pearson is above the baseline Pearson (both versus the same-N random control) | baseline + TS + control |
| C4 | random-dropout JSD within 0.03 of SPR JSD | SPR + dropout |
| C5 | a bigger/frontier backbone is not closer than the small one (gap and JSD) | 2 backbones |

## Phase 6. Main runs (about 300 students per condition, plus the random control)
```bash
M=meta-llama/llama-3.3-70b-instruct
python scripts/02_run_simulation.py --mode random --students 300
for m in baseline spr ts spr_dropout; do
  python scripts/02_run_simulation.py --mode $m --model $M --students 300 --workers 8
done
python scripts/03_evaluate.py runs/*-n300-s0
```
Re-running a command resumes from `cache.jsonl`. Read Pearson/Spearman relative to the random control at N=300 (a random guesser already scores about 0.35 Pearson at this N, versus about 0.41 at the authors' N).
For C5, repeat with a second backbone (e.g. `google/gemini-3.5-flash-lite`).

## Phase 7. Capacity sweep and extension
```bash
for c in 2 4 6 8 12; do
  python scripts/02_run_simulation.py --mode spr --model $M --capacity $c --students 300
done
python scripts/05_capacity_sweep_plot.py --glob "runs/spr-c*-llama-3.3-70b-instruct-n300-s0" \
   --control runs/random-c4-none-n300-s0 --baseline runs/baseline-c4-llama-3.3-70b-instruct-n300-s0
# extension: per-student capacity drawn uniformly from 2..6
python scripts/02_run_simulation.py --mode spr --model $M --capacity-range 2 6 --students 300
```
Question for the extension: does the score SD move from about .11 toward the human .17 without hurting gap and ECE? Compare `score_sd`, `gap`, `ece` against fixed C=4.

## Phase 8. Freeze results
Commit `runs/*/{question_stats.csv,score_hist.csv,student_scores.csv,meta.json}` and `results/`; tag the commit (`git tag v1-results`). Record model slugs, date, temperature and N in the post. Never commit `cache.jsonl` or `authors_data/`. Delete the Kaggle cache (the notebook does).

## Phase 9. Pitfalls
* **Compare JSD only between my own runs.** Synthetic exposure inflates it (a random guesser gets about 0.61 here versus 0.518 in the paper).
* **Same-N control.** Item metrics grow with N even for random guessing.
* **TS stage 1 prompt.** In the paper's template the question block includes the answer-format instruction, while the prompt says "do NOT answer yet". Some models will reply with an answer JSON and be dropped. Check the drop rate in the pilot; report it, do not silently change the prompt.
* **Reasoning models** (e.g. Gemini 3.5 family) may spend output tokens on hidden reasoning; check real cost in the pilot.
* **Cache key includes a per-student nonce**, otherwise identical prompts would give identical answers to every synthetic student.
* **Data quirks:** 3 of 750 questions have no category in the CSV (I default them to Locate; only matters for category analyses); one multiple-choice question flags two correct options (either counts); the human test histogram sums to 1,197 students (paper: 1,198).
* **Temperature:** 1.4 (the authors' config). A backend capping at 1.0 changes diversity; say so.
* **Checkbox scoring** is all-or-nothing; unparseable replies are dropped and counted.
