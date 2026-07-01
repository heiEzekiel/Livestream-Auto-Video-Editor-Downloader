"""
Tests for asr_transcribe.transcribe().

faster-whisper is faked (injected into sys.modules) so this runs without the
library or a real model, and lets us assert the decode parameters and the
cache-skip behavior.
"""
import json
import sys
import types

import asr_transcribe


class _Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def _patch_fake_whisper(monkeypatch, segments):
    """Install a fake faster_whisper.WhisperModel; return the captured-kwargs dict."""
    captured = {}

    class _FakeModel:
        def __init__(self, *a, **k):
            captured["init"] = {"args": a, "kwargs": k}

        def transcribe(self, audio, **kwargs):
            captured["transcribe"] = {"audio": audio, "kwargs": kwargs}
            info = types.SimpleNamespace(duration=len(segments) * 5)
            return iter(segments), info

    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = _FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)
    return captured


def test_transcribe_uses_greedy_decode_and_writes_jsonl(tmp_path, monkeypatch):
    segs = [_Seg(0.0, 2.0, " hello "), _Seg(2.0, 4.0, " world ")]
    captured = _patch_fake_whisper(monkeypatch, segs)

    out = tmp_path / "t.jsonl"
    result = asr_transcribe.transcribe(tmp_path / "in.mp3", out, model_size="tiny")

    # rank 3: greedy decode requested
    assert captured["transcribe"]["kwargs"]["beam_size"] == 1
    assert captured["transcribe"]["kwargs"]["language"] == "en"
    assert captured["transcribe"]["kwargs"]["vad_filter"] is True

    # transcript written, one JSON object per segment, text stripped
    assert result == out
    lines = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {"start": 0.0, "end": 2.0, "text": "hello"},
        {"start": 2.0, "end": 4.0, "text": "world"},
    ]


def test_transcribe_skips_when_cached(tmp_path, monkeypatch):
    out = tmp_path / "t.jsonl"
    out.write_text('{"start": 0, "end": 1, "text": "cached"}\n', encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("must not load the model when a transcript is cached")
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=_boom))

    assert asr_transcribe.transcribe(tmp_path / "in.mp3", out) == out
