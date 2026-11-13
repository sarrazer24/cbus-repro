# REPRODUCTION.md: step-by-step protocol

All commands run from the repository root. `authors_data/` is a clone of the authors' repository (see the README). The pass/fail criteria in Phase 5 should be specified before the main runs and kept unchanged during evaluation.

## Phase 0. Setup

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/iai-group/emnlp2026-simreading authors_data
```

Paths can be overridden with `--data` or the `SIMREADING_DATA` environment variable.

## Phase 1. Offline reproduction of Table 4 (free)

```bash
python scripts/01_check_metrics.py
```

Expected values (mean of the authors' three runs, using my metric implementation):

| Condition | Abs. gap | JSD | ECE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Llama-3.3-70B persona baseline | 0.279 | 0.679 | 0.285 | 0.241 | 0.179 |
| CBUS-SPR (C=4) | 0.037 | 0.093 | 0.069 | 0.301 | 0.297 |
| CBUS-TS (C=2) | 0.238 | 0.475 | 0.244 | 0.455 | 0.304 |
| Held-out real students | 0.026 | 0.006 | 0.027 | 0.895 | 0.845 |

If these values do not match, investigate the metric implementation before proceeding.

Metric conventions that matter:

- Absolute gap is computed on response-weighted accuracy.
- JSD is the divergence, with logarithm base 2—not the square root of the divergence.
- ECE uses six equal-width bins over **human** item difficulty, weighted by the share of questions.
- Item-level metrics use per-question accuracy: `n_correct / (n_correct + n_incorrect)`.

## Phase 2. Metric sanity analyses (same script)

The script also prints:

1. **Per-student score standard deviation**, estimated from histogram midpoints: humans about 0.17, Llama persona baseline about 0.04, and CBUS-SPR about 0.11.
2. **Item-level correlation baselines:** a question-type-only predictor, fitted on the training half and evaluated on the test half, obtains Pearson 0.523 / Spearman 0.339. A chance-level predictor obtains 0.458 / 0.370. This illustrates that item-level correlation can substantially reflect question format.
3. **JSD sampling noise floor** for samples drawn from an identical population: n=50: 0.068; n=100: 0.034; n=200: 0.018; n=500: 0.007; n=1198: 0.003.

## Phase 3. Mock smoke test and prompt check (free)

```bash
python scripts/02_run_simulation.py --mode spr --mock --students 20
# Also try baseline, ts, and spr_dropout.
python scripts/02_run_simulation.py --mode random --students 300
python scripts/03_evaluate.py
python scripts/04_format_baseline.py --students 300 --price-in <usd/M> --price-out <usd/M>
```

The mock LLM answers randomly, so its metrics should resemble the random control. Inspect the printed prompts and confirm the answer-option format and text-then-question layout. These are implementation choices to check, not evidence of model behavior.

## Phase 4. Pilot: 20 students, real model

```bash
export OPENROUTER_API_KEY=...

python scripts/02_run_simulation.py --mode baseline --model meta-llama/llama-3.3-70b-instruct --students 20
python scripts/02_run_simulation.py --mode spr --model meta-llama/llama-3.3-70b-instruct --students 20
python scripts/02_run_simulation.py --mode ts --model meta-llama/llama-3.3-70b-instruct --students 20
```

Inspect `runs/*/meta.json` for:

- **Drop rate:** target below 2%. If TS drops substantially more replies, investigate the prompt behavior described in Phase 9.
- **Actual token usage:** use it to revise the cost estimate.
- **Raw model replies:** inspect the cache or add a debug print to review representative stage-1 and stage-2 responses.

Do not draw performance conclusions from 20 students.

## Phase 5. Evaluation criteria

The following criteria define the planned comparisons. They should be specified before the main runs and evaluated without changing the thresholds afterwards.

| ID | Criterion | Required runs |
|---|---|---|
| C1 | Baseline mean accuracy at least 0.90 and score SD at most 0.06 | Baseline |
| C2 | SPR absolute gap at most 0.15 and SPR JSD at most half the baseline JSD (compare JSD only within this reproduction) | Baseline + SPR |
| C3 | TS absolute gap remains above 0.15 and TS item Pearson exceeds baseline Pearson, with both interpreted against the same-N random control | Baseline + TS + control |
| C4 | Random-dropout JSD is within 0.03 of SPR JSD | SPR + dropout |
| C5 | A larger/frontier backbone is not closer to the human distribution than the smaller backbone, considering both gap and JSD | Two backbones |

These are evaluation thresholds, not guarantees that a condition will pass. Report failures and borderline results rather than adjusting criteria after seeing the outcomes.

## Phase 6. Main runs (about 300 students per condition, plus the random control)

```bash
M=meta-llama/llama-3.3-70b-instruct

python scripts/02_run_simulation.py --mode random --students 300

for m in baseline spr ts spr_dropout; do
  python scripts/02_run_simulation.py --mode $m --model $M --students 300 --workers 8
done

python scripts/03_evaluate.py runs/*-n300-s0
```

Re-running a command resumes from `cache.jsonl`.

Interpret Pearson and Spearman relative to the random control at N=300. A random guesser can already obtain substantial item-level correlation at this sample size (about 0.35 Pearson in this setup), compared with about 0.41 at the authors' sample size.

For C5, repeat the relevant conditions with a second backbone, for example `google/gemini-3.5-flash-lite`. Record the exact model identifier and any backend-specific settings.

## Phase 7. Capacity sweep and extension

```bash
for c in 2 4 6 8 12; do
  python scripts/02_run_simulation.py --mode spr --model $M --capacity $c --students 300
done

python scripts/05_capacity_sweep_plot.py --glob "runs/spr-c*-llama-3.3-70b-instruct-n300-s0" \
  --control runs/random-c4-none-n300-s0 \
  --baseline runs/baseline-c4-llama-3.3-70b-instruct-n300-s0

# Extension: per-student capacity drawn uniformly from 2 to 6.
python scripts/02_run_simulation.py --mode spr --model $M --capacity-range 2 6 --students 300
```

Extension question: does variable capacity move score SD from about 0.11 toward the human value of 0.17 without worsening the accuracy gap and ECE? Compare `score_sd`, `gap`, and `ece` against fixed C=4.

## Phase 8. Freeze results

Commit the intended aggregate outputs:

```text
runs/*/question_stats.csv
runs/*/score_hist.csv
runs/*/student_scores.csv
runs/*/meta.json
results/
```

Then tag the results commit:

```bash
git tag v1-results
```

Record model identifiers, date, temperature, sample size, and relevant generation settings in the accompanying report or blog post.

Never commit `cache.jsonl` or `authors_data/`. The Kaggle notebook deletes its cache after execution.

## Phase 9. Pitfalls and implementation notes

- **Compare JSD only between runs in this reproduction.** Synthetic exposure inflates JSD: a random guesser obtains about 0.61 here versus 0.518 in the paper.
- **Use a same-N control.** Item-level metrics can increase with sample size even for random guessing.
- **TS stage-1 prompt:** In the paper's template, the question block includes the answer-format instruction, while the prompt says "do NOT answer yet". Some models may return answer JSON during stage 1, causing those replies to be dropped. Check the pilot drop rate, report it, and do not silently change the prompt.
- **Reasoning models:** Some models, including models in the Gemini 3.5 family, may spend output tokens on internal reasoning. Check actual token usage and cost during the pilot.
- **Cache key:** It includes a per-student nonce. Without it, identical prompts could produce identical answers for every synthetic student.
- **Data quirks:** Three of 750 questions have no category in the CSV (the implementation defaults them to Locate; this matters only for category analyses). One multiple-choice question flags two correct options (either counts). The human test histogram sums to 1,197 students, while the paper reports 1,198.
- **Temperature:** The authors' configuration uses 1.4. A backend that caps temperature at 1.0 changes response diversity; report this difference.
- **Checkbox scoring:** Checkbox questions are scored all-or-nothing. Unparseable replies are dropped and counted rather than automatically scored as incorrect.
