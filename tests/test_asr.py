"""
Tests for the ASR sermon-boundary detector's deterministic logic.

The LLM calls are mocked, so these run offline. They lock down transcript IO,
prompt formatting, structured-output parsing, and the detect() orchestration
(identify -> adversarial verify -> final boundaries), including the case where
the verifier overrides the proposal.
"""
import json
import types

import pytest

import asr_detect
from asr_detect import load_transcript, format_transcript, _extract_json, detect, AsrResult


# --------------------------------------------------
# transcript IO
# --------------------------------------------------
def test_load_transcript(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"start": 0.0, "end": 2.0, "text": "hello"}\n'
        '\n'  # blank line ignored
        '{"start": 2.0, "end": 4.0, "text": "world"}\n',
        encoding="utf-8",
    )
    segs = load_transcript(p)
    assert segs == [
        {"start": 0.0, "end": 2.0, "text": "hello"},
        {"start": 2.0, "end": 4.0, "text": "world"},
    ]


def test_format_transcript_uses_absolute_seconds_and_skips_empty():
    segs = [
        {"start": 3580.4, "end": 3582.0, "text": "Let us open our Bibles."},
        {"start": 3583.0, "end": 3584.0, "text": ""},      # skipped
        {"start": 3585.9, "end": 3587.0, "text": "Amen."},
    ]
    out = format_transcript(segs)
    assert out == "[3580] Let us open our Bibles.\n[3585] Amen."


# --------------------------------------------------
# structured-output parsing
# --------------------------------------------------
def _text_message(payload: dict):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def test_extract_json():
    msg = _text_message({"start_seconds": 10, "end_seconds": 20})
    assert _extract_json(msg) == {"start_seconds": 10, "end_seconds": 20}


def test_extract_json_raises_without_text():
    msg = types.SimpleNamespace(content=[types.SimpleNamespace(type="thinking", thinking="...")])
    with pytest.raises(ValueError):
        _extract_json(msg)


# --------------------------------------------------
# detect() orchestration with a mocked client
# --------------------------------------------------
class _Stream:
    def __init__(self, msg):
        self._msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._msg


class _FakeMessages:
    """Returns queued messages on successive .stream() calls."""
    def __init__(self, messages):
        self._queue = list(messages)
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _Stream(self._queue.pop(0))


class _FakeClient:
    def __init__(self, messages):
        self.messages = _FakeMessages(messages)


def _transcript_file(tmp_path):
    p = tmp_path / "svc.jsonl"
    p.write_text('{"start": 0, "end": 1, "text": "a"}\n', encoding="utf-8")
    return p


def test_detect_agreement(tmp_path):
    proposal = {"start_seconds": 3500, "end_seconds": 7000, "start_quote": "open your Bibles",
                "end_quote": "let us pray", "reasoning": "clear sermon block"}
    verdict = {"agree": True, "start_seconds": 3500, "end_seconds": 7000, "issue": "none"}
    client = _FakeClient([_text_message(proposal), _text_message(verdict)])

    r = detect(_transcript_file(tmp_path), client=client)
    assert isinstance(r, AsrResult)
    assert (r.start, r.end) == (3500, 7000)
    assert (r.proposed_start, r.proposed_end) == (3500, 7000)
    assert r.agreed is True
    # two LLM calls: identify + verify
    assert len(client.messages.calls) == 2


def test_detect_verifier_overrides(tmp_path):
    proposal = {"start_seconds": 4400, "end_seconds": 7000, "start_quote": "q1",
                "end_quote": "q2", "reasoning": "anchored on later plateau"}
    verdict = {"agree": False, "start_seconds": 3580, "end_seconds": 7100,
               "issue": "start was 14 min too late; sermon opens at 3580"}
    client = _FakeClient([_text_message(proposal), _text_message(verdict)])

    r = detect(_transcript_file(tmp_path), client=client)
    # final boundaries come from the verifier, proposal is retained for audit
    assert (r.start, r.end) == (3580, 7100)
    assert (r.proposed_start, r.proposed_end) == (4400, 7000)
    assert r.agreed is False
    assert "too late" in r.issue


def test_write_result_adds_end_padding(tmp_path, monkeypatch):
    import asr_results
    monkeypatch.setattr(asr_results, "ASR_TEST", tmp_path / "asr")
    monkeypatch.setattr(asr_results, "DOWNLOADED", tmp_path / "downloaded")  # no mp4 -> no trim

    rec = asr_results.write_result("2026-09-13", 3000, 6000, meta={"agree": True}, trim=False)

    assert rec["start_s"] == 3000
    assert rec["detected_end_s"] == 6000
    assert rec["end_padding_s"] == 5
    assert rec["end_s"] == 6005           # 5s padding added to the end
    assert rec["kept_s"] == 3005
    assert rec["agree"] is True
    # records written
    assert (tmp_path / "asr" / "2026-09-13" / "boundary.json").exists()
    assert (tmp_path / "asr" / "2026-09-13" / "summary.txt").exists()


def test_write_result_custom_padding(tmp_path, monkeypatch):
    import asr_results
    monkeypatch.setattr(asr_results, "ASR_TEST", tmp_path / "asr")
    monkeypatch.setattr(asr_results, "DOWNLOADED", tmp_path / "downloaded")
    rec = asr_results.write_result("d", 10, 20, trim=False, end_padding=0)
    assert rec["end_s"] == 20 and rec["end_padding_s"] == 0


def test_detect_passes_structured_output_config(tmp_path):
    proposal = {"start_seconds": 1, "end_seconds": 2, "start_quote": "x",
                "end_quote": "y", "reasoning": "r"}
    verdict = {"agree": True, "start_seconds": 1, "end_seconds": 2, "issue": "none"}
    client = _FakeClient([_text_message(proposal), _text_message(verdict)])

    detect(_transcript_file(tmp_path), client=client)
    # both calls must request structured JSON output and adaptive thinking
    for kw in client.messages.calls:
        assert kw["thinking"] == {"type": "adaptive"}
        assert kw["output_config"]["format"]["type"] == "json_schema"
        assert kw["model"] == asr_detect.DEFAULT_MODEL
