"""
Tests for the diarization / trim-detection logic in audio_utils.py.

The sermon is the long high-confidence plateau in the self-enrolled Preacher
similarity. ``get_trim_range`` anchors on that plateau and extends the two ends
*asymmetrically* (tight at the start to avoid bleeding into announcements,
forgiving at the end to keep the closing/altar-call). These tests lock down:

  * parse_speaker_scores      -- reading the saved text format
  * _smooth                   -- pre-detection moving average
  * _longest_run_above        -- the plateau "core" finder
  * _extend                   -- directional boundary extension
  * get_trim_range / get_*_trim -- thresholds + asymmetry + padding + fallbacks
  * find_longest_speech_run   -- loudness-band seed for self-enrollment
  * self_enroll_preacher      -- centroid + outlier-refinement
  * embeds_per_second / similarity_per_second -- per-second bucketing

The module reads a module-level `cfg`; we monkeypatch it with the temp config
so thresholds are deterministic regardless of the live config.yaml.
"""
import numpy as np
import pytest

import audio_utils
from audio_utils import AudioUtils


@pytest.fixture
def audio(monkeypatch, cfg):
    monkeypatch.setattr(audio_utils, "cfg", cfg)
    return AudioUtils()


def _line(sec, announcement, preacher):
    """Build one line in the similarity-per-second text format."""
    return f"{sec:04d}s | Announcement: {announcement:.2f} | Preacher: {preacher:.2f}"


# --------------------------------------------------
# parse_speaker_scores
# --------------------------------------------------
def test_parse_speaker_scores_extracts_one_speaker():
    lines = [_line(0, 0.1, 0.9), _line(1, 0.2, 0.8)]
    secs, scores = AudioUtils.parse_speaker_scores(lines, "Preacher")
    assert secs == [0, 1]
    assert scores == pytest.approx([0.9, 0.8])


def test_parse_speaker_scores_skips_missing_speaker():
    lines = ["0000s | Announcement: 0.50", _line(1, 0.2, 0.8)]
    secs, scores = AudioUtils.parse_speaker_scores(lines, "Preacher")
    assert secs == [1]
    assert scores == pytest.approx([0.8])


# --------------------------------------------------
# _smooth
# --------------------------------------------------
def test_smooth_window_one_is_identity():
    out = AudioUtils._smooth([0.1, 0.9, 0.2], 1)
    assert out == pytest.approx([0.1, 0.9, 0.2])


def test_smooth_averages_and_preserves_length():
    out = AudioUtils._smooth([0.0, 3.0, 0.0], 3)
    assert len(out) == 3
    assert out == pytest.approx([1.0, 1.0, 1.0])  # centred 3-pt moving average


# --------------------------------------------------
# _longest_run_above (plateau core)
# --------------------------------------------------
def test_longest_run_above_basic():
    scores = [0.1] * 10 + [0.9] * 20 + [0.1] * 10
    assert AudioUtils._longest_run_above(scores, thr=0.5, gap=2, min_len=5) == (10, 29)


def test_longest_run_above_bridges_short_gap():
    scores = [0.9] * 10 + [0.1] * 2 + [0.9] * 10
    assert AudioUtils._longest_run_above(scores, thr=0.5, gap=2, min_len=5) == (0, 21)


def test_longest_run_above_does_not_bridge_large_gap():
    scores = [0.9] * 10 + [0.1] * 2 + [0.9] * 10
    # gap=1 cannot bridge the 2s dip -> two runs; the earlier equal-length wins.
    assert AudioUtils._longest_run_above(scores, thr=0.5, gap=1, min_len=5) == (0, 9)


def test_longest_run_above_respects_min_len():
    scores = [0.1] * 5 + [0.9] * 3 + [0.1] * 5
    assert AudioUtils._longest_run_above(scores, thr=0.5, gap=1, min_len=5) is None


def test_longest_run_above_picks_longest():
    scores = [0.9] * 6 + [0.1] * 5 + [0.9] * 30
    assert AudioUtils._longest_run_above(scores, thr=0.5, gap=1, min_len=5) == (11, 40)


def test_longest_run_above_empty():
    assert AudioUtils._longest_run_above([0.1, 0.2], thr=0.5, gap=1, min_len=5) is None


# --------------------------------------------------
# _extend (directional boundary extension)
# --------------------------------------------------
def test_extend_back_stops_at_dip():
    scores = [0.2, 0.2, 0.7, 0.7, 0.9]
    # From the core (idx 4), walk back over the 0.7 shoulder, stop at the 0.2 dip.
    assert AudioUtils._extend(scores, anchor=4, thr=0.6, gap=0, step=-1) == 2


def test_extend_back_bridges_small_dip():
    scores = [0.7, 0.2, 0.7, 0.9]
    assert AudioUtils._extend(scores, anchor=3, thr=0.6, gap=1, step=-1) == 0
    # gap=0 cannot bridge the single 0.2 dip.
    assert AudioUtils._extend(scores, anchor=3, thr=0.6, gap=0, step=-1) == 2


def test_extend_forward_bridges_long_dip():
    scores = [0.9, 0.2, 0.2, 0.2, 0.7]
    # gap=3 bridges the 3s dip to keep the trailing 0.7 (the "closing").
    assert AudioUtils._extend(scores, anchor=0, thr=0.6, gap=3, step=+1) == 4
    # gap=2 cannot -> stays at the anchor.
    assert AudioUtils._extend(scores, anchor=0, thr=0.6, gap=2, step=+1) == 0


# --------------------------------------------------
# get_trim_range -- the asymmetric integration
# --------------------------------------------------
def _asym_config(cfg):
    """Thresholds chosen so the synthetic service below is fully deterministic."""
    cfg.update("trim_logic.smooth_window_seconds", 1, autosave=False)
    cfg.update("trim_logic.low_percentile", 60, autosave=False)
    cfg.update("trim_logic.high_percentile", 80, autosave=False)
    cfg.update("trim_logic.shoulder_ratio", 0.5, autosave=False)
    cfg.update("trim_logic.core_gap_seconds", 60, autosave=False)
    cfg.update("trim_logic.start_gap_seconds", 60, autosave=False)
    cfg.update("trim_logic.end_gap_seconds", 300, autosave=False)
    cfg.update("trim_logic.min_continuous_seconds", 10, autosave=False)
    cfg.update("trim_logic.start_padding", 10, autosave=False)
    cfg.update("trim_logic.padding", 5, autosave=False)


def _synthetic_service():
    """
    A 1000s service: quiet pre-roll, a pre-sermon bump (announcements), a long
    dip, the sermon core, another dip, a closing bump, then post-roll.
    The start must exclude the pre-bump; the end must keep the closing bump.
    """
    bands = [
        (0, 200, 0.20),    # pre-roll
        (200, 260, 0.65),  # pre-sermon bump  -> must be EXCLUDED from start
        (260, 380, 0.25),  # 120s dip (> start_gap) separates bump from sermon
        (380, 680, 0.90),  # sermon core
        (680, 800, 0.25),  # 120s dip (< end_gap) before the closing
        (800, 860, 0.65),  # closing / altar-call -> must be INCLUDED in end
        (860, 1000, 0.20), # post-roll
    ]
    lines = []
    for start, stop, val in bands:
        for sec in range(start, stop):
            lines.append(_line(sec, 0.10, val))
    return lines


def test_get_trim_range_asymmetric(audio, cfg):
    _asym_config(cfg)
    lines = _synthetic_service()

    start, end = audio.get_trim_range(lines)

    # Start: core begins at 380; back-extension stops at the dip (no bleed into
    # the 200-259 pre-bump); minus 10s start padding -> 370.
    assert start == 370
    # End: core ends at 679; forward-extension bridges the 680-799 dip and keeps
    # the 800-859 closing bump; +5s padding -> 864.
    assert end == 864


def test_get_trim_range_wrappers(audio, cfg):
    _asym_config(cfg)
    lines = _synthetic_service()
    assert audio.get_start_trim(lines) == 370
    assert audio.get_end_trim(lines) == 864


def test_get_trim_range_falls_back_without_preacher(audio):
    lines = [f"{sec:04d}s | Announcement: 0.50" for sec in range(50)]
    assert audio.get_trim_range(lines) == (0, 49)
    assert audio.get_start_trim(lines) == 0
    assert audio.get_end_trim(lines) == 49


# --------------------------------------------------
# speaker-dominance gate
# --------------------------------------------------
def test_advance_past_non_dominant_skips_leading_region():
    dominant = np.array([False] * 30 + [True] * 100)
    # First sustained-dominant second is index 30.
    assert AudioUtils._advance_past_non_dominant(None, dominant, 0, 80, 10) == 30


def test_advance_past_non_dominant_ignores_brief_blip():
    # A 3s dominant blip at 10, then sustained dominance from 40.
    dominant = np.array([False] * 130)
    dominant[10:13] = True
    dominant[40:] = True
    assert AudioUtils._advance_past_non_dominant(None, dominant, 0, 90, 10) == 40


def test_advance_past_non_dominant_no_op_when_all_dominant():
    dominant = np.ones(100, dtype=bool)
    assert AudioUtils._advance_past_non_dominant(None, dominant, 25, 80, 10) == 25


def test_advance_past_non_dominant_falls_back_to_core():
    dominant = np.zeros(100, dtype=bool)  # never dominant before the core
    assert AudioUtils._advance_past_non_dominant(None, dominant, 0, 50, 10) == 50


def test_aligned_column_zeros_when_missing():
    lines = [f"{s:04d}s | Preacher: 0.80" for s in range(20)]
    out = AudioUtils._aligned_column(lines, "Announcement", 20, 1)
    assert out.shape == (20,)
    assert np.all(out == 0.0)


def _line3(sec, p, a, w):
    return f"{sec:04d}s | Preacher: {p:.2f} | Announcement: {a:.2f} | Worship: {w:.2f}"


def test_get_trim_range_dominance_gate_skips_announcer(audio, cfg):
    # An announcer block (200-259) scores high on Preacher (0.88, above the
    # shoulder) but Announcement out-scores it (0.95). Without the dominance gate
    # back-extension would start at 200; the gate advances to the sermon at 260.
    for k, v in {"smooth_window_seconds": 1, "low_percentile": 40, "high_percentile": 65,
                 "shoulder_ratio": 0.9, "core_gap_seconds": 120, "start_gap_seconds": 60,
                 "end_gap_seconds": 300, "min_continuous_seconds": 10, "start_padding": 0,
                 "padding": 0, "dominance_margin": 0.0, "dominance_sustain_seconds": 10}.items():
        cfg.update(f"trim_logic.{k}", v, autosave=False)

    lines = []
    for s in range(200):
        lines.append(_line3(s, 0.20, 0.20, 0.10))   # pre-roll
    for s in range(200, 260):
        lines.append(_line3(s, 0.88, 0.95, 0.10))   # announcer: high P, but A dominates
    for s in range(260, 560):
        lines.append(_line3(s, 0.90, 0.50, 0.10))   # sermon: preacher dominant
    for s in range(560, 800):
        lines.append(_line3(s, 0.20, 0.20, 0.10))   # post-roll

    assert audio.get_start_trim(lines) == 260


# --------------------------------------------------
# find_longest_speech_run (loudness-band seed)
# --------------------------------------------------
def test_find_longest_speech_run_picks_longest(audio):
    levels = {}
    for sec in range(10):
        levels[sec] = -60.0          # silence
    for sec in range(10, 30):
        levels[sec] = -30.0          # speech (20s)
    for sec in range(30, 80):
        levels[sec] = -10.0          # loud worship (50s gap > bridge)
    for sec in range(80, 160):
        levels[sec] = -35.0          # speech (80s) -> the sermon
    for sec in range(160, 170):
        levels[sec] = -60.0          # silence

    assert audio.find_longest_speech_run(levels) == (80, 159)


def test_find_longest_speech_run_bridges_brief_dip(audio):
    levels = {sec: -30.0 for sec in range(40)}
    for sec in range(20, 25):
        levels[sec] = -10.0  # 5s loud blip within the speech run
    assert audio.find_longest_speech_run(levels) == (0, 39)


def test_find_longest_speech_run_empty(audio):
    assert audio.find_longest_speech_run({}) is None
    assert audio.find_longest_speech_run({0: -5.0, 1: -3.0}) is None  # all loud


# --------------------------------------------------
# self_enroll_preacher (centroid + refinement)
# --------------------------------------------------
def test_self_enroll_preacher_rejects_outliers(audio):
    per_sec = {sec: np.array([1.0, 0.0]) for sec in range(8)}
    per_sec[8] = np.array([0.0, 1.0])
    per_sec[9] = np.array([0.0, 1.0])

    embed = audio.self_enroll_preacher(per_sec, run=(0, 9))
    assert embed == pytest.approx(np.array([1.0, 0.0]), abs=1e-6)
    assert np.linalg.norm(embed) == pytest.approx(1.0)


def test_self_enroll_preacher_none_run(audio):
    assert audio.self_enroll_preacher({0: np.array([1.0, 0.0])}, run=None) is None


def test_self_enroll_preacher_empty_run(audio):
    assert audio.self_enroll_preacher({}, run=(0, 5)) is None


# --------------------------------------------------
# embeds_per_second
# --------------------------------------------------
def test_embeds_per_second_buckets_and_normalises(audio):
    from resemblyzer.hparams import sampling_rate as sr

    slices = [
        slice(0, int(0.5 * sr)),
        slice(int(0.5 * sr), int(1.0 * sr)),
        slice(int(1.0 * sr), int(2.0 * sr)),
    ]
    cont_embeds = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    out = audio.embeds_per_second(cont_embeds, slices)

    assert out[0] == pytest.approx(np.array([1.0, 0.0]))
    assert out[1] == pytest.approx(np.array([0.0, 1.0]))
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)


# --------------------------------------------------
# similarity_per_second
# --------------------------------------------------
def test_similarity_per_second_buckets_and_averages(audio):
    from resemblyzer.hparams import sampling_rate as sr

    slices = [
        slice(0, int(0.5 * sr)),
        slice(int(0.5 * sr), int(1.0 * sr)),
        slice(int(1.0 * sr), int(2.0 * sr)),
    ]
    sim = {"Preacher": np.array([0.1, 0.3, 0.9])}

    out = audio.similarity_per_second(sim, slices)

    assert out[0]["Preacher"] == pytest.approx(0.2)
    assert out[1]["Preacher"] == pytest.approx(0.9)