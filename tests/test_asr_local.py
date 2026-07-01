"""
Tests for the self-hosted hybrid detector (asr_local.py) and the local-LLM
refinement (asr_llm.py). The LLM is mocked, so these run offline and do not
require llama-cpp-python or a model file.
"""
import json
import types

import pytest

import asr_local
import asr_llm
from asr_local import (is_handoff, sermon_score, longest_sermon_region,
                       heuristic_detect, detect)


# --------------------------------------------------
# lexical primitives
# --------------------------------------------------
def test_is_handoff_positive():
    assert is_handoff("If you are, help me to welcome pastor Lawrence.")
    assert is_handoff("Are you ready for the preaching of the word?")
    assert is_handoff("Let's welcome Pastor Gabriel for the preaching of the word.")
    assert is_handoff("Please join me as we welcome our beloved pastor, Mark.")


def test_is_handoff_negative():
    assert not is_handoff("Welcome to New Creation Church, so glad you're here.")
    assert not is_handoff("Let us sing this song together.")


def test_sermon_score_sign():
    assert sermon_score("Turn with me to Romans chapter 8, the grace of Jesus.") > 0
    assert sermon_score("Sign up this week for the youth event, registration online.") < 0


def test_longest_sermon_region():
    scores = [-1] * 10 + [2] * 40 + [-1] * 10
    assert longest_sermon_region(scores, thr=1.0, gap=2, min_len=20) == (10, 49)


# --------------------------------------------------
# heuristic_detect on a synthetic service
# --------------------------------------------------
def _seg(start, text):
    return {"start": float(start), "end": float(start) + 5, "text": text}


def _synthetic_service():
    segs = []
    # 0..1800s announcements (negative score)
    for s in range(0, 1800, 10):
        segs.append(_seg(s, "Sign up this week for the event, registration and dinner."))
    # 1800s handoff to the preacher
    segs.append(_seg(1800, "Are you ready for the preaching of the word? Help me welcome Pastor Lawrence."))
    # 1810..5400s sermon (positive score)
    for s in range(1810, 5400, 10):
        segs.append(_seg(s, "In Romans chapter 8, the grace of God in Jesus Christ, by faith."))
    # 5400..5520s preacher-led closing / altar call
    for s in range(5400, 5520, 10):
        segs.append(_seg(s, "Every head bowed and every eye closed, say this prayer, receive Jesus."))
    # 5520 dismissal
    segs.append(_seg(5520, "Thank you for joining us, see you next Sunday, God bless you."))
    return segs


def test_heuristic_detect_handoff_and_closing():
    h = heuristic_detect(_synthetic_service())
    assert h is not None
    assert h.start_s == 1800          # the handoff segment
    assert h.start_conf >= 0.8
    assert 5400 <= h.end_s <= 5520    # within the closing block
    assert h.end_conf >= 0.8


def test_heuristic_detect_no_handoff_low_confidence():
    # Drop the handoff line -> sermon region still found, but low start confidence.
    segs = [s for s in _synthetic_service() if "Help me welcome" not in s["text"]]
    h = heuristic_detect(segs)
    assert h is not None
    assert h.start_conf < 0.7         # triggers the local LLM in detect()


# --------------------------------------------------
# detect() orchestration (LLM mocked)
# --------------------------------------------------
def test_detect_high_confidence_skips_llm(tmp_path, monkeypatch):
    p = tmp_path / "svc.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in _synthetic_service()), encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("LLM must not be called for high-confidence boundaries")
    monkeypatch.setattr(asr_llm, "refine_boundaries", _boom)

    r = detect(p)
    assert r.method == "heuristic"
    assert r.start == 1800


def test_detect_low_confidence_uses_llm(tmp_path, monkeypatch):
    segs = [s for s in _synthetic_service() if "Help me welcome" not in s["text"]]
    p = tmp_path / "svc.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in segs), encoding="utf-8")

    monkeypatch.setattr(asr_llm, "refine_boundaries", lambda segs, h: (1815, 5499))
    r = detect(p)
    assert r.method == "heuristic+llm"
    assert (r.start, r.end) == (1815, 5499)


def _low_conf_file(tmp_path):
    segs = [s for s in _synthetic_service() if "Help me welcome" not in s["text"]]
    p = tmp_path / "svc.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in segs), encoding="utf-8")
    return p


def test_detect_rejects_invalid_llm_span(tmp_path, monkeypatch):
    # A nonsensical model span (start >= end) must be ignored, keeping the heuristic.
    p = _low_conf_file(tmp_path)
    monkeypatch.setattr(asr_llm, "refine_boundaries", lambda segs, h: (5000, 200))
    r = detect(p)
    assert r.method == "heuristic"
    assert 0 <= r.start < r.end


def test_detect_skip_llm_env(tmp_path, monkeypatch):
    # SKIP_LLM=1 must keep it heuristic-only even on a low-confidence boundary.
    p = _low_conf_file(tmp_path)
    monkeypatch.setenv("SKIP_LLM", "1")
    def _boom(*a, **k):
        raise AssertionError("LLM must not run when SKIP_LLM=1")
    monkeypatch.setattr(asr_llm, "refine_boundaries", _boom)
    r = detect(p)
    assert r.method == "heuristic"


def test_refine_rejects_invalid_span(monkeypatch):
    # Even with a low-confidence end, a start>=end model span falls back to heuristic.
    monkeypatch.setattr(asr_llm, "_get_llm",
                        lambda: _FakeLLM({"start_seconds": 9000, "end_seconds": 100}))
    h = types.SimpleNamespace(start_s=3600, end_s=6800, start_conf=0.4, end_conf=0.4)
    assert asr_llm.refine_boundaries([{"start": 0, "end": 1, "text": "x"}], h) == (3600, 6800)


# --------------------------------------------------
# asr_llm: downsample / parse / anchor-keeping (model mocked)
# --------------------------------------------------
def test_downsample_bins_segments():
    segs = [{"start": 0, "end": 1, "text": "a"}, {"start": 12, "end": 13, "text": "b"},
            {"start": 25, "end": 26, "text": "c"}]
    out = asr_llm.downsample(segs, bin_s=20)
    assert out == "[0] a b\n[20] c"


def test_parse_json():
    assert asr_llm._parse('noise {"start_seconds": 3600, "end_seconds": 7200} tail') == (3600, 7200)
    assert asr_llm._parse("no json here") is None


class _FakeLLM:
    def __init__(self, payload):
        self._payload = payload
    def create_chat_completion(self, messages, **kw):
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}


def test_refine_keeps_confident_start(monkeypatch):
    monkeypatch.setattr(asr_llm, "_get_llm",
                        lambda: _FakeLLM({"start_seconds": 9999, "end_seconds": 7000}))
    h = types.SimpleNamespace(start_s=3600, end_s=6800, start_conf=0.85, end_conf=0.4)
    start, end = asr_llm.refine_boundaries([{"start": 0, "end": 1, "text": "x"}], h)
    assert start == 3600     # high-confidence start kept, not the model's 9999
    assert end == 7000       # low-confidence end taken from the model


def test_refine_fallback_on_unparseable(monkeypatch):
    class _Bad:
        def create_chat_completion(self, messages, **kw):
            return {"choices": [{"message": {"content": "sorry, no json"}}]}
    monkeypatch.setattr(asr_llm, "_get_llm", lambda: _Bad())
    h = types.SimpleNamespace(start_s=3600, end_s=6800, start_conf=0.4, end_conf=0.4)
    assert asr_llm.refine_boundaries([{"start": 0, "end": 1, "text": "x"}], h) == (3600, 6800)


# --------------------------------------------------
# asr_llm: windowed refinement (rank 1) — only the uncertain boundary's
# neighbourhood is sent to the model, with a generous margin.
# --------------------------------------------------
def test_window_segs_keeps_only_neighbourhood_and_tolerates_offset():
    # segments every 20s across a long service
    segs = [{"start": s, "end": s + 1, "text": f"seg{s}"} for s in range(0, 6001, 20)]
    win = asr_llm.window_segs(segs, center_s=1800, window_s=420)  # [1380, 2220]
    starts = [int(s["start"]) for s in win]
    assert min(starts) == 1380 and max(starts) == 2220
    assert 1300 not in starts and 2300 not in starts          # outside the window
    # a true boundary up to ~5 min off the anchor is still inside the window
    assert 1700 in starts and 2100 in starts


class _RecordingLLM:
    """Captures the user prompt so we can assert which segments were sent."""
    def __init__(self, payload):
        self._payload = payload
        self.last_user = None
    def create_chat_completion(self, messages, **kw):
        self.last_user = next(m["content"] for m in messages if m["role"] == "user")
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}


def _long_segs():
    return [{"start": s, "end": s + 1, "text": f"seg{s}"} for s in range(0, 6001, 20)]


def test_refine_windows_uncertain_start_only(monkeypatch):
    # confident END, uncertain START -> only the window around start_s is sent
    rec = _RecordingLLM({"start_seconds": 1850, "end_seconds": 5400})
    monkeypatch.setattr(asr_llm, "_get_llm", lambda: rec)
    h = types.SimpleNamespace(start_s=1800, end_s=5400, start_conf=0.4, end_conf=0.85)
    start, end = asr_llm.refine_boundaries(_long_segs(), h)
    assert "seg1800" in rec.last_user                 # window around the uncertain start
    assert "seg0" not in rec.last_user                # far-earlier segments excluded
    assert "seg5400" not in rec.last_user             # the confident end side excluded
    assert start == 1850 and end == 5400              # start from model, end kept (anchor)


def test_refine_windows_uncertain_end_only(monkeypatch):
    rec = _RecordingLLM({"start_seconds": 1800, "end_seconds": 5380})
    monkeypatch.setattr(asr_llm, "_get_llm", lambda: rec)
    h = types.SimpleNamespace(start_s=1800, end_s=5400, start_conf=0.85, end_conf=0.4)
    start, end = asr_llm.refine_boundaries(_long_segs(), h)
    assert "seg5400" in rec.last_user
    assert "seg0" not in rec.last_user
    assert "seg1800" not in rec.last_user             # the confident start side excluded
    assert start == 1800 and end == 5380             # start kept (anchor), end from model


def test_refine_both_uncertain_sends_full_transcript(monkeypatch):
    rec = _RecordingLLM({"start_seconds": 1850, "end_seconds": 5380})
    monkeypatch.setattr(asr_llm, "_get_llm", lambda: rec)
    h = types.SimpleNamespace(start_s=1800, end_s=5400, start_conf=0.4, end_conf=0.4)
    asr_llm.refine_boundaries(_long_segs(), h)
    assert "seg0" in rec.last_user and "seg6000" in rec.last_user  # whole service sent


# --------------------------------------------------
# asr_llm._get_llm: env-overridable n_ctx (rank 2) / n_threads (rank 4).
# A fake llama_cpp module captures the Llama(...) kwargs so this runs without
# llama-cpp-python or a real model file.
# --------------------------------------------------
def _patch_fake_llama(monkeypatch, tmp_path):
    """Install a fake llama_cpp.Llama that records kwargs; returns the record dict."""
    import sys

    captured = {}

    class _FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_mod = types.ModuleType("llama_cpp")
    fake_mod.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)

    model = tmp_path / "model.gguf"
    model.write_bytes(b"\0")
    monkeypatch.setattr(asr_llm, "MODEL_PATH", model)
    monkeypatch.setattr(asr_llm, "_llm", None)  # bypass the singleton cache
    return captured


def test_get_llm_defaults(monkeypatch, tmp_path):
    captured = _patch_fake_llama(monkeypatch, tmp_path)
    monkeypatch.delenv("ASR_LLM_NCTX", raising=False)
    monkeypatch.delenv("ASR_LLM_THREADS", raising=False)

    asr_llm._get_llm()
    assert captured["n_ctx"] == 20480                      # right-sized default
    assert captured["n_threads"] == (asr_llm.os.cpu_count() or 4)  # all cores by default


def test_get_llm_env_overrides(monkeypatch, tmp_path):
    captured = _patch_fake_llama(monkeypatch, tmp_path)
    monkeypatch.setenv("ASR_LLM_NCTX", "8192")
    monkeypatch.setenv("ASR_LLM_THREADS", "3")

    asr_llm._get_llm()
    assert captured["n_ctx"] == 8192
    assert captured["n_threads"] == 3
