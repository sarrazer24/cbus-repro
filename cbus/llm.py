"""OpenAI-compatible chat client (OpenRouter, Ollama, vLLM...) with disk cache, retries and a mock LLM.
The cache key includes a per-call `nonce` so identical prompts for different simulated students are NOT
collapsed into one sample (at temperature 1.4 each student needs its own draw). Needs only `requests`."""
from __future__ import annotations
import hashlib, json, os, random, re, threading, time
from typing import Optional
import requests

OPENROUTER = "https://openrouter.ai/api/v1"
OLLAMA = "http://localhost:11434/v1"


class Usage:
    def __init__(self):
        self.calls = self.cache_hits = self.tin = self.tout = 0
        self.lock = threading.Lock()

    def add(self, tin, tout, hit):
        with self.lock:
            self.calls += 1
            self.cache_hits += int(hit)
            if not hit:
                self.tin += tin
                self.tout += tout

    def dict(self):
        return dict(calls=self.calls, cache_hits=self.cache_hits, input_tokens=self.tin, output_tokens=self.tout)


class LLM:
    def __init__(self, model: str, base_url: str = OPENROUTER, api_key: Optional[str] = None, temperature: float = 1.4,
                 max_tokens: int = 4000, cache_path: Optional[str] = "cache.jsonl", retries: int = 5, timeout: int = 120):
        self.model, self.base_url, self.temperature, self.max_tokens = model, base_url.rstrip("/"), temperature, max_tokens
        self.api_key, self.retries, self.timeout = api_key, retries, timeout
        self.usage, self._lock, self.cache, self.cache_path = Usage(), threading.Lock(), {}, cache_path
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[r["k"]] = r["v"]
                    except Exception:
                        pass

    def _key(self, prompt, nonce):
        return hashlib.sha256(json.dumps([self.model, self.temperature, self.max_tokens, nonce, prompt]).encode()).hexdigest()

    def complete(self, prompt: str, nonce: str = "") -> Optional[str]:
        k = self._key(prompt, nonce)
        if k in self.cache:
            self.usage.add(0, 0, True)
            return self.cache[k]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}])
        for attempt in range(self.retries):
            try:
                r = requests.post(self.base_url + "/chat/completions", headers=headers, json=body, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                j = r.json()
                out = j["choices"][0]["message"]["content"] or ""
                u = j.get("usage", {})
                self.usage.add(u.get("prompt_tokens", len(prompt) // 4), u.get("completion_tokens", len(out) // 4), False)
                self._store(k, out)
                return out
            except Exception as e:  # noqa
                if attempt == self.retries - 1:
                    print(f"[llm] giving up: {e}")
                    return None
                time.sleep(min(60, 2 ** attempt + random.random()))

    def _store(self, k, v):
        self.cache[k] = v
        if self.cache_path:
            with self._lock, open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


class MockLLM:
    """Offline stand-in: valid-format random replies (= random-guess behaviour); occasionally garbage
    (garbage_rate) so the 'dropped' path is exercised. Deterministic given (prompt, nonce)."""
    def __init__(self, garbage_rate: float = 0.01, seed: int = 0):
        self.model, self.garbage_rate, self.seed, self.usage = "mock", garbage_rate, seed, Usage()

    def complete(self, prompt: str, nonce: str = "") -> str:
        rng = random.Random(hashlib.md5(f"{self.seed}|{nonce}|{prompt}".encode()).hexdigest())
        self.usage.add(len(prompt) // 4, 20, False)
        if rng.random() < self.garbage_rate:
            return "Sorry, I cannot answer that."
        if '"propositions"' in prompt and "Respond with ONLY" in prompt:
            m = re.search(r"at most (\d+) distinct", prompt)
            n = int(m.group(1)) if m else 4
            return json.dumps({"propositions": [f"fact {i}" for i in range(n)]})
        if "true or false question" in prompt:
            return json.dumps({"answer": rng.random() < 0.5})
        n_opt = len(re.findall(r"^[A-Z]\. ", prompt, flags=re.M))
        if "multiple choice question" in prompt:
            return json.dumps({"answer": "ABCDEFGHIJ"[rng.randrange(max(n_opt, 2))]})
        if "checkbox question" in prompt:
            return json.dumps({"answer": [rng.random() < 0.5 for _ in range(max(n_opt, 1))]})
        return "{}"
