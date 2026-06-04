"""Registry: task set + RAM-tier model selection."""

import pytest

import registry
from registry import TASKS, Task, ModelTier, get_task

EXPECTED_TASKS = {"ocr", "vision", "reasoning", "code", "summary", "embed", "agent"}


def test_all_expected_tasks_present():
    assert EXPECTED_TASKS <= set(TASKS)


def test_task_kinds_valid():
    valid = {"text", "vision", "embed", "agent"}
    assert all(t.kind in valid for t in TASKS.values())


def test_get_task_unknown_raises():
    with pytest.raises(ValueError):
        get_task("nope")


def test_pick_model_picks_largest_that_fits():
    t = Task("x", "text", "d", tiers=[
        ModelTier("small", 6), ModelTier("mid", 16), ModelTier("big", 32),
    ])
    assert t.pick_model(8) == "small"
    assert t.pick_model(16) == "mid"
    assert t.pick_model(31) == "mid"
    assert t.pick_model(64) == "big"


def test_pick_model_below_smallest_falls_back_to_smallest():
    t = Task("x", "text", "d", tiers=[ModelTier("small", 6), ModelTier("big", 32)])
    assert t.pick_model(2) == "small"


@pytest.mark.parametrize("budget,expected", [
    (16, "qwen2.5:7b"), (24, "qwen2.5:14b"), (32, "qwen2.5:14b"), (48, "qwen2.5:14b"),
])
def test_reasoning_tiers_match_evidence(budget, expected):
    assert TASKS["reasoning"].pick_model(budget) == expected


@pytest.mark.parametrize("budget,expected", [
    (16, "llama3.1:8b"), (24, "llama3.1:8b"), (32, "qwen2.5:32b"), (48, "qwen2.5:32b"),
])
def test_summary_tiers_match_evidence(budget, expected):
    assert TASKS["summary"].pick_model(budget) == expected


def test_ocr_caps_at_7b():
    assert TASKS["ocr"].pick_model(48) == "qwen2.5vl:7b"


def test_embed_default_is_bge_m3_at_budget():
    assert TASKS["embed"].pick_model(16) == "bge-m3"
