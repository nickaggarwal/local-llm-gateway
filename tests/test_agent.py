"""Agent: tool-call parsing from model content (no Ollama/sandbox)."""

import agent
from agent import _parse_content_tool_calls, TOOL_NAMES


def test_tool_names():
    assert TOOL_NAMES == {"run_python", "install_package"}


def test_parse_single_tool_call_object():
    txt = '{"name": "run_python", "arguments": {"code": "print(1)"}}'
    calls = _parse_content_tool_calls(txt)
    assert calls and calls[0]["function"]["name"] == "run_python"
    assert calls[0]["function"]["arguments"] == {"code": "print(1)"}


def test_parse_tool_call_in_code_fence():
    txt = '```json\n{"name": "install_package", "arguments": {"package": "scipy"}}\n```'
    calls = _parse_content_tool_calls(txt)
    assert calls and calls[0]["function"]["name"] == "install_package"


def test_parse_list_of_tool_calls():
    txt = '[{"name":"run_python","arguments":{"code":"x"}},{"name":"run_python","arguments":{"code":"y"}}]'
    calls = _parse_content_tool_calls(txt)
    assert len(calls) == 2


def test_parse_plain_text_returns_none():
    assert _parse_content_tool_calls("Here is your answer: 42") is None


def test_parse_unknown_tool_returns_none():
    assert _parse_content_tool_calls('{"name": "rm_rf", "arguments": {}}') is None
