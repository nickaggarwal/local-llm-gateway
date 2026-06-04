"""Eval candidate ladders + RAM-fit math."""

import candidates
from candidates import LADDERS, TIERS, fits, runtime_gb

import registry


def test_ladder_keys_are_real_tasks():
    # Every ladder must map to a registry task (minus the special 'agent' task).
    for name in LADDERS:
        assert name in registry.TASKS


def test_ladders_cover_text_vision_embed_tasks():
    assert {"ocr", "vision", "reasoning", "code", "summary", "embed"} <= set(LADDERS)


def test_runtime_gb_vision_has_more_overhead_than_text():
    assert runtime_gb(7.0, "vision") > runtime_gb(7.0, "text")


def test_fits_respects_os_headroom():
    # A model needing ~9.7 GB runtime fits 16 GB (16-5 headroom) but not 12.
    w, kind = 6.0, "vision"  # runtime = 6*1.2+2.5 = 9.7
    assert runtime_gb(w, kind) == 9.7
    assert fits(w, kind, 16) is True
    assert fits(w, kind, 12) is False


def test_tiers_are_expected():
    assert TIERS == [16, 24, 32, 48]


def test_cand_has_model_and_weights():
    kind, ladder = LADDERS["ocr"]
    assert kind == "vision"
    assert all(c.model and c.weights_gb > 0 for c in ladder)
