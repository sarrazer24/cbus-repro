# cbus-repro: a small-scale, from-scratch reproduction of CBUS

My own implementation of the main experiments of **"Act Like a 5th Grader" is Not Enough: Bounding Knowledge in LLM-Based User Simulators** (Balog & Bakken, EMNLP Findings 2026, [arXiv 2608.30033](https://arxiv.org/abs/2608.30033)). The authors' code and data: <https://github.com/iai-group/emnlp2026-simreading>. This repo is **not** a copy of their code.

**The claim being tested.** Persona prompting ("act like a 5th grader") yields near-perfect, near-identical simulated students. Forcing the LLM through a working-memory bottleneck (extract at most C propositions, then answer with the text removed) gives score distributions much closer to real students.

## What is here

| Part | What it does |
|---|---|
| `cbus/data.py` | loads the authors' released item bank and aggregates (not shipped here) |
| `cbus/prompts.py` | prompts transcribed from the paper (Figs. 2, 4, 5, 6, 11-13) + answer parsing/grading |
| `cbus/llm.py` | OpenAI-compatible client (OpenRouter / Ollama / vLLM), disk cache, retries, offline `MockLLM` |
| `cbus/simulate.py` | baseline, CBUS-SPR, CBUS-TS, SPR with random dropout, no-LLM random control, synthetic students |
| `cbus/metrics.py` | abs. gap, JSD, ECE, Pearson, Spearman (verified against the paper's Table 4) |
| `scripts/01..05` | metric check, simulation runner, evaluation table, prompt/cost preview, capacity-sweep plot |
| `notebooks/kaggle_runner.ipynb` | CPU-only Kaggle runner (API key as a Secret) |
| `REPRODUCTION.md` | the phased protocol, pre-registered pass/fail criteria, pitfalls |

## Quick start (no API key, no GPU)

```bash
git clone https://github.com/YOUR-USERNAME/cbus-repro && cd cbus-repro
pip install -r requirements.txt
git clone --depth 1 https://github.com/iai-group/emnlp2026-simreading authors_data   # their data: not redistributed here
python scripts/01_check_metrics.py                                   # reproduces Table 4 rows offline
python scripts/02_run_simulation.py --mode spr --mock --students 20  # smoke test, random mock LLM
python scripts/03_evaluate.py
```

Real runs (OpenRouter; set `OPENROUTER_API_KEY`):

```bash
python scripts/04_format_baseline.py --students 300 --price-in 0.10 --price-out 0.32   # look at prompts + rough cost first
python scripts/02_run_simulation.py --mode spr --model meta-llama/llama-3.3-70b-instruct --students 20    # pilot
python scripts/02_run_simulation.py --mode random --students 300                                         # same-N control
python scripts/02_run_simulation.py --mode spr --model meta-llama/llama-3.3-70b-instruct --students 300
```

Local server instead of OpenRouter: `--base-url http://localhost:11434/v1 --model <ollama tag>` (CPU inference of a 70B model is not realistic on a laptop; this path exists for small models only).

## Design choices that differ from the paper (be explicit about them)

* **Synthetic students.** Per-student exposure is not public, so each synthetic student reads about 7 texts sampled by text popularity (from the released reader counts). Item-level metrics (gap, ECE, Pearson, Spearman) are comparable with the paper up to sampling noise; **JSD of my runs is only comparable between my own runs.**
* **Small N.** Item metrics depend strongly on the number of simulated students; always read them against a random control at the same N.
* **Option format** (`A. ...` lines) is my choice; the paper does not specify it.
* **Temperature 1.4** is the authors' config (described as 0.7 on a 0-1 scale); OpenRouter accepts 0-2. If a backend caps at 1.0, use `--temperature 1.0` and say so.
* Unparseable replies are dropped and counted (`meta.json`), not scored as wrong.

## Licensing and data hygiene

My code: MIT (see `LICENSE`). The authors' questions, categories and human aggregates are CC BY-NC-ND 4.0, their simulation aggregates CC BY 4.0, and the reading texts are third-party material. **Do not commit `authors_data/` or `cache.jsonl`** (the cache contains paraphrases of licensed texts); `.gitignore` excludes them. Only my small aggregate outputs (`runs/*/question_stats.csv`, `score_hist.csv`, `meta.json`, `results/`) are meant to be committed.
