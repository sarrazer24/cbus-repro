"""Phase 3-6: run a simulated population. Writes runs/<name>/{question_stats.csv,score_hist.csv,student_scores.csv,meta.json}.
Examples:
  python scripts/02_run_simulation.py --mode spr --mock --students 20           # offline smoke test
  python scripts/02_run_simulation.py --mode random --students 300              # no-LLM random control
  python scripts/02_run_simulation.py --mode spr --model meta-llama/llama-3.3-70b-instruct --students 300
"""
import _common  # noqa
import argparse, json, os, time
from cbus import data
from cbus.llm import LLM, MockLLM, OPENROUTER
from cbus.simulate import Simulator, make_students, MODES

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=MODES)
ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
ap.add_argument("--capacity", type=int, default=None, help="C (default: 4 for spr/spr_dropout, 2 for ts)")
ap.add_argument("--capacity-range", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                help="extension: each student draws C uniformly from [LO,HI]")
ap.add_argument("--students", type=int, default=300)
ap.add_argument("--texts-per-student", type=int, default=7)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--mock", action="store_true", help="use the offline random MockLLM (no network)")
ap.add_argument("--temperature", type=float, default=1.4, help="authors' config: 1.4 (= 0.7 on a 0-1 scale); use 1.0 if a backend caps at 1")
ap.add_argument("--max-tokens", type=int, default=4000)
ap.add_argument("--base-url", default=OPENROUTER, help="OpenAI-compatible; Ollama: http://localhost:11434/v1")
ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
ap.add_argument("--cache", default="cache.jsonl")
ap.add_argument("--data", default=data.DEFAULT_DATA)
ap.add_argument("--name", default=None)
ap.add_argument("--out-root", default="runs")
a = ap.parse_args()

cap = a.capacity or (2 if a.mode == "ts" else 4)
corpus = data.load_corpus(a.data)
if a.mode == "random":
    llm = None
elif a.mock:
    llm = MockLLM(seed=a.seed)
else:
    key = os.environ.get(a.api_key_env, "")
    if not key and "localhost" not in a.base_url:
        raise SystemExit(f"Set {a.api_key_env} (env var / Kaggle secret) or use --mock.")
    llm = LLM(a.model, a.base_url, key, a.temperature, a.max_tokens, a.cache)

students = make_students(corpus, a.students, a.texts_per_student, a.seed, a.data,
                         tuple(a.capacity_range) if a.capacity_range else None)
tag = "mock" if a.mock else ("none" if a.mode == "random" else a.model.split("/")[-1])
capname = f"c{a.capacity_range[0]}-{a.capacity_range[1]}" if a.capacity_range else f"c{cap}"
name = a.name or f"{a.mode}-{capname}-{tag}-n{a.students}-s{a.seed}"
out = os.path.join(a.out_root, name)
os.makedirs(out, exist_ok=True)

sim = Simulator(llm, corpus, a.mode, cap, a.seed)
t0 = time.time()
print(f"[run] {name}: {a.students} students x ~{a.texts_per_student} texts")
r = sim.run(students, a.workers)
r["qs"].to_csv(os.path.join(out, "question_stats.csv"), index=False)
r["hist"].to_csv(os.path.join(out, "score_hist.csv"), index=False)
with open(os.path.join(out, "student_scores.csv"), "w") as f:
    f.write("score\n" + "\n".join(f"{x:.6f}" for x in r["scores"]) + "\n")
meta = dict(vars(a), name=name, capacity_used=cap, n_students_scored=len(r["scores"]), n_answered=r["n_answered"],
            n_dropped_answers_or_units=r["n_dropped"], dropped_stage1_calls=sim.dropped_stage1,
            dropped_answer_replies=sim.dropped_answers, seconds=round(time.time() - t0, 1),
            usage=llm.usage.dict() if llm else None,
            option_format="A. <option> lines (my choice; paper does not specify)")
with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1, default=str)
tot = r["n_answered"] + r["n_dropped"]
print(f"[done] answered={r['n_answered']} dropped={r['n_dropped']} ({100 * r['n_dropped'] / max(tot, 1):.2f}%) "
      f"usage={meta['usage']} -> {out}")
