"""Simulators: baseline persona, CBUS-SPR, CBUS-TS, SPR with random dropout, and a no-LLM random control.
A synthetic student reads `texts_per_student` texts sampled by popularity (no per-student exposure is public)."""
from __future__ import annotations
import hashlib, random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from . import prompts as P
from .data import Corpus, Item, text_weights
from .metrics import student_hist

MODES = ("baseline", "spr", "ts", "spr_dropout", "random")


def make_students(corpus: Corpus, n: int, texts_per_student: int, seed: int, data_root: str,
                  cap_range: Optional[Tuple[int, int]] = None) -> List[dict]:
    ids, p = text_weights(corpus, data_root)
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n):
        k = min(texts_per_student, len(ids))
        texts = list(rng.choice(ids, size=k, replace=False, p=p))
        cap = int(rng.integers(cap_range[0], cap_range[1] + 1)) if cap_range else None
        out.append(dict(sid=s, texts=texts, capacity=cap))
    return out


def random_answer(it: Item, rng: random.Random) -> int:
    if it.qtype == "trueOrFalse":
        return int((rng.random() < 0.5) == it.gold)
    if it.qtype == "multiChoice":
        return int(rng.randrange(len(it.options)) in it.gold)
    return int([rng.random() < 0.5 for _ in it.options] == it.gold)


class Simulator:
    def __init__(self, llm, corpus: Corpus, mode: str, capacity: int, seed: int = 0):
        assert mode in MODES
        self.llm, self.corpus, self.mode, self.capacity, self.seed = llm, corpus, mode, capacity, seed
        self.dropped_stage1 = 0
        self.dropped_answers = 0

    def _nonce(self, *parts):
        return "|".join(str(x) for x in (self.seed,) + parts)

    def run_unit(self, sid: int, tid: str, cap: int) -> List[Tuple[str, Optional[int]]]:
        """One student reading one text and answering all its gradable questions."""
        text, items = self.corpus.texts[tid], self.corpus.by_text[tid]
        res: List[Tuple[str, Optional[int]]] = []
        if self.mode == "random":
            rng = random.Random(self._nonce("rnd", sid, tid))
            return [(it.qid, random_answer(it, rng)) for it in items]
        props_spr = None
        if self.mode in ("spr", "spr_dropout"):
            pool = cap * 3 if self.mode == "spr_dropout" else cap
            reply = self.llm.complete(P.spr_stage1(text, pool), self._nonce("s1", sid, tid))
            props_spr = P.parse_propositions(reply, limit=pool)
            if props_spr is None:                      # whole unit dropped and counted
                self.dropped_stage1 += 1
                return [(it.qid, None) for it in items]
            if self.mode == "spr_dropout":
                rng = random.Random(self._nonce("drop", sid, tid))
                props_spr = rng.sample(props_spr, min(cap, len(props_spr)))
        for it in items:
            if self.mode == "baseline":
                reply = self.llm.complete(P.baseline_prompt(text, it), self._nonce("q", sid, it.qid))
            else:
                props = props_spr
                if self.mode == "ts":
                    r1 = self.llm.complete(P.ts_stage1(text, it, cap), self._nonce("s1", sid, it.qid))
                    props = P.parse_propositions(r1, limit=cap)
                    if props is None:
                        self.dropped_stage1 += 1
                        res.append((it.qid, None))
                        continue
                reply = self.llm.complete(P.stage2_prompt(props, it), self._nonce("q", sid, it.qid))
            g = P.grade(reply, it)
            if g is None:
                self.dropped_answers += 1
            res.append((it.qid, g))
        return res

    def run(self, students: List[dict], workers: int = 8, progress: bool = True) -> dict:
        units = [(s["sid"], t, s["capacity"] or self.capacity) for s in students for t in s["texts"]]
        done = 0
        results: Dict[Tuple[int, str], list] = {}
        def work(u):
            return u, self.run_unit(*u)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for u, r in ex.map(work, units):
                results[(u[0], u[1])] = r
                done += 1
                if progress and done % max(1, len(units) // 10) == 0:
                    print(f"  {done}/{len(units)} student-texts", flush=True)
        return self.aggregate(students, results)

    @staticmethod
    def aggregate(students, results) -> dict:
        qc: Dict[Tuple[str, str], List[int]] = {}
        scores, n_ans, n_drop = [], 0, 0
        for s in students:
            c = n = 0
            for t in s["texts"]:
                for qid, g in results[(s["sid"], t)]:
                    if g is None:
                        n_drop += 1
                        continue
                    a = qc.setdefault((t, qid), [0, 0])
                    a[0 if g else 1] += 1
                    c += g; n += 1
            if n:
                scores.append(c / n)
            n_ans += n
        qs = pd.DataFrame([dict(sanity_text_id=k[0], question_id=k[1], n_correct=v[0], n_incorrect=v[1]) for k, v in sorted(qc.items())])
        return dict(qs=qs, hist=student_hist(scores), scores=scores, n_answered=n_ans, n_dropped=n_drop)
