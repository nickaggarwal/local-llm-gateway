"""Pure scoring functions per task. Each returns a float in [0, 1]."""

import re
import subprocess
import sys
import tempfile
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def _lev(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def score_ocr(output: str, ex: dict) -> float:
    """Composite of (1 - char error rate) and key-token coverage."""
    no, nt = _norm(output), _norm(ex["truth"])
    cer = _lev(no, nt) / max(1, len(nt))
    char_acc = max(0.0, 1.0 - cer)
    keys = ex.get("key_tokens", [])
    tok_acc = (sum(1 for k in keys if _norm(k) in no) / len(keys)) if keys else char_acc
    return round(0.5 * char_acc + 0.5 * tok_acc, 4)


def _accept_match(output: str, accept: list[str]) -> float:
    o = _norm(output).lower()
    return 1.0 if any(_norm(a).lower() in o for a in accept) else 0.0


def score_vision(output: str, ex: dict) -> float:
    return _accept_match(output, ex["accept"])


def score_chat(output: str, ex: dict) -> float:
    return _accept_match(output, ex["accept"])


def score_summarize(output: str, ex: dict) -> float:
    """Key-point coverage with a soft length penalty."""
    o = _norm(output).lower()
    pts = ex["must_include"]
    cover = sum(1 for p in pts if _norm(p).lower() in o) / len(pts)
    words = len(output.split())
    cap = ex.get("max_words", 80)
    penalty = 0.0 if words <= cap * 1.5 else 0.2
    return round(max(0.0, cover - penalty), 4)


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def score_code(output: str, ex: dict) -> float:
    """Functional correctness: run the solution against asserts (all-or-nothing)."""
    code = _extract_code(output)
    script = code + "\n\n" + "\n".join(ex["tests"]) + "\nprint('OK')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        return 1.0 if r.returncode == 0 and "OK" in r.stdout else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        import os
        os.unlink(path)


def cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# Note: embed is scored in run_eval (needs multiple embedding calls per example).
SCORERS = {
    "ocr": score_ocr,
    "vision": score_vision,
    "chat": score_chat,
    "summarize": score_summarize,
    "code": score_code,
}
