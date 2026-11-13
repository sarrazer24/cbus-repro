# cbus-repro: a small-scale, from-scratch reproduction of CBUS

My own implementation of the main experiments from **["Act Like a 5th Grader" is Not Enough: Bounding Knowledge in LLM-Based User Simulators](https://arxiv.org/abs/2608.30033)** (Balog & Bakken, EMNLP Findings 2026). The authors' code and data are available at <https://github.com/iai-group/emnlp2026-simreading>. This repository is an independent implementation, not a copy of their code.

**The claim being tested.** Persona prompting ("act like a 5th grader") can produce near-perfect, near-identical simulated students. Forcing an LLM through a working-memory bottleneck—extracting at most C propositions, then answering questions with the original text removed—can produce score distributions closer to those of real students.

## What is here

| Part | What it does |
|---|---|
| `cbus/data.py` | Loads the authors' released item bank and aggregates (not shipped here) |
| `cbus/prompts.py` | Prompts transcribed from the paper (Figs. 2, 4, 5, 6, 11–13), plus answer parsing and grading |
| `cbus/llm.py` | OpenAI-compatible client (OpenRouter / Ollama / vLLM), disk cache, retries, and offline `MockLLM` |
| `cbus/simulate.py` | Persona baseline, CBUS-SPR, CBUS-TS, SPR with random dropout, no-LLM random control, and synthetic students |
| `cbus/metrics.py` | Absolute accuracy gap, JSD, ECE, Pearson, and Spearman (verified against the paper's Table 4) |
| `scripts/01..05` | Metric check, simulation runner, evaluation table, prompt/cost preview, and capacity-sweep plot |
| `notebooks/kaggle_runner.ipynb` | Kaggle runner; the main experiments were run using vLLM on a free Kaggle GPU environment with two T4 GPUs |
| `REPRODUCTION.md` | The phased protocol, pre-specified pass/fail criteria, and implementation pitfalls |

## Quick start (no API key, no GPU)

```bash
git clone https://github.com/sarrazer24/cbus-repro.git
cd cbus-repro
pip install -r requirements.txt

# The authors' data is not redistributed in this repository.
git clone --depth 1 https://github.com/iai-group/emnlp2026-simreading authors_data

# Reproduce the paper's Table 4 metrics offline.
python scripts/01_check_metrics.py

# Run a small smoke test with the mock LLM.
python scripts/02_run_simulation.py --mode spr --mock --students 20

# Evaluate available runs.
python scripts/03_evaluate.py
```

The mock LLM is intended for testing the pipeline, not for reproducing meaningful model results.

## Running real experiments

### OpenRouter

Set your `OPENROUTER_API_KEY` environment variable before running the commands below.

First, inspect the prompts and estimated cost:

```bash
python scripts/04_format_baseline.py --students 300 --price-in 0.10 --price-out 0.32
```

Then run a small pilot, followed by the full experiments:

```bash
# Small pilot
python scripts/02_run_simulation.py --mode spr --model meta-llama/llama-3.3-70b-instruct --students 20

# Same-N random control
python scripts/02_run_simulation.py --mode random --students 300

# Main simulation
python scripts/02_run_simulation.py --mode spr --model meta-llama/llama-3.3-70b-instruct --students 300
```

### Local inference server

The client also supports OpenAI-compatible local endpoints, including Ollama and vLLM. For example:

```bash
python scripts/02_run_simulation.py --mode spr --base-url http://localhost:11434/v1 --model <ollama-tag>
```

Running a 70B model locally on a typical laptop is not realistic; this option is intended for smaller models or machines with suitable hardware.

## Design choices that differ from the paper

These differences matter when interpreting the results.

* **Synthetic students.** Individual students' reading exposure is not publicly available. Each synthetic student therefore reads about seven texts sampled by text popularity, using the released reader counts. Item-level metrics (accuracy gap, ECE, Pearson, and Spearman) can be compared with the paper with sampling limitations in mind. **JSD values from my synthetic runs should be compared only against other runs in this reproduction**, not directly against the paper's JSD values.
* **Small sample size.** Item-level metrics depend strongly on the number of simulated students. Interpret them alongside a random control with the same sample size.
* **Answer-option format.** The `A. ...` option format is my implementation choice; the paper does not specify this exact format.
* **Temperature.** Temperature 1.4 follows the authors' configuration, described as 0.7 on a 0–1 scale. OpenRouter accepts values from 0 to 2. If a backend caps temperature at 1.0, use `--temperature 1.0` and report that difference.
* **Unparseable replies.** Replies that cannot be parsed are dropped and counted in `meta.json`; they are not automatically scored as incorrect.

## Licensing and data hygiene

My implementation is released under the MIT License (see `LICENSE`).

The authors' questions, categories, and human aggregates are distributed under CC BY-NC-ND 4.0; their simulation aggregates are under CC BY 4.0. The reading texts are third-party material and have their own licensing conditions. Refer to the original repository for the applicable terms.

**Do not commit `authors_data/` or `cache.jsonl`.** The cache can contain paraphrases of licensed reading texts. Both are excluded through `.gitignore`.

Only the intended aggregate outputs—such as `runs/*/question_stats.csv`, `score_hist.csv`, `meta.json`, and files in `results/`—should be committed. Do not redistribute the authors' source data or cached text through this repository.
