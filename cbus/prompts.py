"""Prompts transcribed from the paper (Figs. 2, 4, 5, 6, 11-13) + answer parsing.
Option formatting ("A. ...", one per line) is MY choice: the paper does not specify it."""
from __future__ import annotations
import json, re
from typing import List, Optional
from .data import Item

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

BASELINE = ("You are simulating a 5th-grade student in Norway taking a reading comprehension test. "
            "Read the following text and answer the question as a typical 10-11 year old student would. "
            "The student may not always answer correctly.\n"
            "<TEXT> {text_content} </TEXT>\n<QUESTION> {question_text} </QUESTION>")

SPR_STAGE1 = ("You are simulating a 5th-grade student in Norway taking a reading comprehension test. "
              "You are about to read a text, and you'll be asked questions about this text. As a young reader, "
              "you cannot remember everything from the text \u2014 only a few key facts.\n"
              "Read the text and identify at most {capacity} distinct propositions (paraphrased atomic facts from the text) "
              "that you think are most relevant to the questions you'll be asked. Do not answer any questions yet.\n"
              "<TEXT> {text_content} </TEXT>\n"
              "Respond with ONLY a JSON object in the following format, with no other text: "
              '{{"propositions": ["proposition 1", "proposition 2", ...]}}')

TS_STAGE1 = ("You are simulating a 5th-grade student in Norway taking a reading comprehension test. "
             "You have just read the question below and are now scanning the text specifically to find the answer. "
             "As a young reader, you cannot process the whole text at once \u2014 you only focus on a few key details that seem to match the question.\n"
             "Scan the text and extract at most {capacity} distinct propositions (paraphrased atomic facts from the text) "
             "that share keywords or seem most directly related to what the question is asking. Do NOT answer the question yet.\n"
             "<TEXT> {text_content} </TEXT>\n<QUESTION> {question_text} </QUESTION>\n"
             "Respond with ONLY a JSON object in the following format, with no other text: "
             '{{"propositions": ["proposition 1", "proposition 2", ...]}}')

STAGE2 = ("You are simulating a 5th-grade student in Norway taking a reading comprehension test. "
          "You no longer have access to the full text \u2014 you can only rely on the propositions you remembered from reading. "
          "Answer the question as a typical 10-11 year old student would, based ONLY on the propositions below. "
          "The student may not always answer correctly.\n"
          "<PROPOSITIONS> {propositions} </PROPOSITIONS>\n<QUESTION> {question_text} </QUESTION>")

TF_FMT = ("This is a true or false question. Decide whether the following statement is true or false based on the text.\n"
          "Statement: {statement}\n"
          'Respond with ONLY a JSON object in the following format, with no other text: {{"answer": true}} or {{"answer": false}}')
MC_FMT = ("This is a multiple choice question. Choose exactly ONE answer from the options below.\n"
          "Options: {options_text}\n"
          'Respond with ONLY a JSON object in the following format, with no other text: {{"answer": "<letter>"}} '
          "where <letter> is the letter (A, B, C, ...) of your chosen option.")
CB_FMT = ("This is a checkbox question. For each option below, decide whether it is correct or not.\n"
          "Options: {options_text}\n"
          'Respond with ONLY a JSON object in the following format, with no other text: {{"answer": [true, false, ...]}}\n'
          "The list must contain one true/false value for each option, in the same order as listed above.")


def options_text(it: Item) -> str:
    return "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it.options))


def question_text(it: Item) -> str:
    if it.qtype == "trueOrFalse":
        return TF_FMT.format(statement=it.question)
    fmt = MC_FMT if it.qtype == "multiChoice" else CB_FMT
    return f"{it.question}\n" + fmt.format(options_text=options_text(it))  # stem before format block: my choice


def baseline_prompt(text: str, it: Item) -> str:
    return BASELINE.format(text_content=text, question_text=question_text(it))


def spr_stage1(text: str, capacity: int) -> str:
    return SPR_STAGE1.format(text_content=text, capacity=capacity)


def ts_stage1(text: str, it: Item, capacity: int) -> str:
    return TS_STAGE1.format(text_content=text, question_text=question_text(it), capacity=capacity)


def stage2_prompt(props: List[str], it: Item) -> str:
    body = "\n".join(f"- {p}" for p in props) if props else "(nothing remembered)"
    return STAGE2.format(propositions="\n" + body + "\n", question_text=question_text(it))


def _json(reply):
    if not reply:
        return None
    s = re.sub(r"```(?:json)?", "", reply).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def parse_propositions(reply: str, limit: Optional[int] = None) -> Optional[List[str]]:
    d = _json(reply)
    if not isinstance(d, dict) or not isinstance(d.get("propositions"), list):
        return None
    props = [str(p).strip() for p in d["propositions"] if str(p).strip()]
    return props[:limit] if limit else props


def grade(reply: str, it: Item) -> Optional[int]:
    """1/0, or None if unparseable (dropped and counted, never scored as wrong)."""
    d = _json(reply)
    if not isinstance(d, dict) or "answer" not in d:
        return None
    a = d["answer"]
    if it.qtype == "trueOrFalse":
        if isinstance(a, bool):
            return int(a == it.gold)
        if isinstance(a, str) and a.strip().lower() in ("true", "false"):
            return int((a.strip().lower() == "true") == it.gold)
        return None
    if it.qtype == "multiChoice":
        if not isinstance(a, str):
            return None
        a = a.strip().strip(".").upper()
        if len(a) != 1 or a not in LETTERS[: len(it.options)]:
            return None
        return int(LETTERS.index(a) in it.gold)
    if it.qtype == "checkboxes":  # all-or-nothing
        if not isinstance(a, list) or len(a) != len(it.options) or not all(isinstance(x, bool) for x in a):
            return None
        return int(a == it.gold)
    return None