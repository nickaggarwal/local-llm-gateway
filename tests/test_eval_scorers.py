"""Eval scorers: pure objective scoring per task kind."""

import scorers
from scorers import (
    SCORERS, cosine, score_code, score_ocr, score_reasoning, score_summary, score_vision,
)


def test_scorer_keys():
    assert set(SCORERS) == {"ocr", "vision", "reasoning", "summary", "code"}


# --- OCR ---

def test_ocr_perfect_and_wrong():
    ex = {"truth": "Total: 161.86", "key_tokens": ["161.86"]}
    assert score_ocr("Total: 161.86", ex) == 1.0
    assert score_ocr("Total: 999.99", ex) < 0.6


def test_ocr_normalizes_dashes_and_quotes():
    ex = {"truth": "O'Brien-Ng", "key_tokens": ["O'Brien-Ng"]}
    # en-dash + curly quote should still match after normalization
    assert score_ocr("O’Brien–Ng", ex) == 1.0


# --- accept-list matching (vision / reasoning) ---

def test_accept_match_word_boundary():
    # "1" must not match inside "10"
    assert score_vision("the answer is 10", {"accept": ["1"]}) == 0.0
    assert score_vision("the answer is 1", {"accept": ["1"]}) == 1.0
    assert score_reasoning("I think it is Tokyo.", {"accept": ["tokyo"]}) == 1.0


def test_accept_match_no_substring_false_positive():
    assert score_reasoning("nowhere", {"accept": ["no"]}) == 0.0


# --- summary ---

def test_summary_keyfact_coverage():
    ex = {"must_include": ["alpha", "beta", "gamma", "delta"], "max_words": 50}
    assert score_summary("alpha beta gamma delta", ex) == 1.0
    assert score_summary("alpha beta", ex) == 0.5


def test_summary_length_penalty():
    ex = {"must_include": ["alpha"], "max_words": 3}
    long = "alpha " + "filler " * 50
    assert score_summary(long, ex) < 1.0


# --- code (runs the solution against asserts) ---

def test_code_pass_and_fail():
    ex = {"entrypoint": "f", "tests": ["assert f(2) == 4", "assert f(3) == 9"]}
    assert score_code("```python\ndef f(x):\n    return x*x\n```", ex) == 1.0
    assert score_code("def f(x):\n    return x", ex) == 0.0


def test_code_extracts_from_fenced_block():
    ex = {"entrypoint": "g", "tests": ["assert g() == 1"]}
    out = "Here you go:\n```python\ndef g():\n    return 1\n```\nDone."
    assert score_code(out, ex) == 1.0


def test_code_timeout_or_error_scores_zero():
    ex = {"entrypoint": "h", "tests": ["assert h() == 1"]}
    assert score_code("def h():\n    raise ValueError('boom')", ex) == 0.0


# --- cosine ---

def test_cosine_identity_and_orthogonal():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
    assert cosine([], []) == 0.0
