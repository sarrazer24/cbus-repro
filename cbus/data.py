"""Loading the authors' released data (NOT shipped in this repo; clone it separately)."""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

DEFAULT_DATA = os.environ.get("SIMREADING_DATA", "authors_data")  # path to the cloned authors' repo root


@dataclass
class Item:
    text_id: str
    qid: str
    qtype: str                      # multiChoice | trueOrFalse | checkboxes
    question: str
    options: List[str] = field(default_factory=list)
    gold: object = None             # list[int] (MC) | bool (TF) | list[bool] (checkbox)
    category: str = "Locate"        # 3 questions lack a category in the CSV -> Locate (my choice)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.text_id, self.qid)


@dataclass
class Corpus:
    texts: Dict[str, str]                 # text_id -> text (licensed third-party: never commit)
    items: Dict[Tuple[str, str], Item]
    by_text: Dict[str, List[Item]]


def load_corpus(root: str = DEFAULT_DATA) -> Corpus:
    with open(os.path.join(root, "data", "questions.json"), encoding="utf-8") as f:
        raw = json.load(f)
    cats = pd.read_csv(os.path.join(root, "data", "question_categories.csv"))
    catmap = {(r.sanity_text_id, r.question_id): r.category for r in cats.itertuples()}
    texts, items, by_text = {}, {}, {}
    for t in raw:
        tid = t["sanity_text_id"]
        texts[tid] = t["text"]
        lst = []
        for q in t["questions"]:
            ty = q["type"]
            if ty == "freeText":
                continue
            it = Item(tid, q["question_id"], ty, q["question"].strip(), category=catmap.get((tid, q["question_id"]), "Locate"))
            if ty == "multiChoice":
                it.options = [o["option"].strip() for o in q["options"]]
                it.gold = [i for i, o in enumerate(q["options"]) if o["is_correct"]]  # one item flags 2 options: any counts
            elif ty == "checkboxes":
                it.options = [o["option"].strip() for o in q["options"]]
                it.gold = [bool(o["is_correct"]) for o in q["options"]]
            elif ty == "trueOrFalse":
                it.gold = bool(q["is_correct"])
            items[it.key] = it
            lst.append(it)
        by_text[tid] = lst
    return Corpus(texts, items, by_text)


def _h(root, name):
    return pd.read_csv(os.path.join(root, "data", "aggregates", "human", name))


def human_stats(split: str = "test", root: str = DEFAULT_DATA) -> pd.DataFrame:
    return _h(root, f"question_stats_{split}.csv")


def human_hist(split: str = "test", root: str = DEFAULT_DATA, category: Optional[str] = None) -> pd.DataFrame:
    suffix = f"_{category}" if category else ""
    return _h(root, f"score_hist_{split}{suffix}.csv")


def reader_counts(split: str = "test", root: str = DEFAULT_DATA) -> pd.DataFrame:
    return _h(root, f"text_reader_counts_{split}.csv")


def authors_sim_dir(root: str = DEFAULT_DATA) -> str:
    return os.path.join(root, "data", "aggregates", "simulation")


def load_authors_run(method: str, run: int, root: str = DEFAULT_DATA):
    d = os.path.join(authors_sim_dir(root), method)
    qs = pd.read_csv(os.path.join(d, f"r{run}_question_stats.csv"))
    h = pd.read_csv(os.path.join(d, f"r{run}_score_hist.csv"))
    return qs, h


def text_weights(corpus: Corpus, root: str = DEFAULT_DATA) -> Tuple[List[str], np.ndarray]:
    """Sampling weights per text = test reader counts (texts without gradable questions or counts are dropped)."""
    rc = reader_counts("test", root)
    w = dict(zip(rc.sanity_text_id, rc.n_readers))
    ids = [t for t in corpus.texts if corpus.by_text.get(t) and w.get(t, 0) > 0]
    p = np.array([w[t] for t in ids], float)
    return ids, p / p.sum()
