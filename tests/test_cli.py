"""CLI argument wiring (gateway faked)."""

import sys

import pytest

import cli
import gateway


def _run_cli(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cli.py", *argv])
    return cli.main()


def test_tasks_lists_all_tasks(monkeypatch, capsys):
    rc = _run_cli(["tasks"], monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    for name in ("ocr", "reasoning", "code", "summary", "embed"):
        assert name in out


def test_run_prints_text(monkeypatch, capsys):
    monkeypatch.setattr(gateway, "run",
                        lambda *a, **k: {"text": "the answer", "backend": "ollama", "model": "m"})
    rc = _run_cli(["run", "reasoning", "what is 2+2"], monkeypatch)
    assert rc == 0
    assert "the answer" in capsys.readouterr().out


def test_run_reports_error_nonzero(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("backend down")
    monkeypatch.setattr(gateway, "run", boom)
    rc = _run_cli(["run", "reasoning", "hi"], monkeypatch)
    assert rc == 1
    assert "backend down" in capsys.readouterr().err


def test_run_embed_prints_dim(monkeypatch, capsys):
    monkeypatch.setattr(gateway, "run",
                        lambda *a, **k: {"embedding": [0.1, 0.2, 0.3], "backend": "ollama", "model": "bge-m3"})
    _run_cli(["run", "embed", "hello"], monkeypatch)
    assert "3-dim" in capsys.readouterr().out


def test_invalid_task_rejected(monkeypatch):
    with pytest.raises(SystemExit):  # argparse choices rejects unknown task
        _run_cli(["run", "banana", "hi"], monkeypatch)
