"""Tests for Configs (configs.py)."""
import logging
from datetime import date
from pathlib import Path

import pytest
import yaml

from configs import Configs


def _write_yaml(path: Path, data: dict):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# --------------------------------------------------
# Loading / validation
# --------------------------------------------------
def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Configs(tmp_path / "nope.yaml")


def test_non_dict_yaml_raises(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("just a plain string", encoding="utf-8")
    with pytest.raises(ValueError):
        Configs(bad)


# --------------------------------------------------
# Generic readers
# --------------------------------------------------
def test_get_dot_notation(cfg):
    assert cfg.get("audio.sample_rate") == 48000
    assert cfg.get("diarization.confidence_thresholds.announcement") == 0.8


def test_get_missing_returns_default(cfg):
    assert cfg.get("does.not.exist") is None
    assert cfg.get("does.not.exist", 42) == 42


def test_get_path(cfg):
    assert cfg.get_path("path.downloaded") == Path("downloaded")
    assert cfg.get_path("missing.key") is None


# --------------------------------------------------
# Updates persist to disk
# --------------------------------------------------
def test_update_persists(cfg, config_file):
    cfg.update("metadata.title", "Hello World")
    # Re-read from disk via a fresh instance to prove autosave worked
    reloaded = Configs(config_file)
    assert reloaded.get("metadata.title") == "Hello World"


def test_update_creates_nested_path(cfg):
    cfg.update("brand.new.value", 7)
    assert cfg.get("brand.new.value") == 7


# --------------------------------------------------
# Service date
# --------------------------------------------------
def test_service_date_get_set(cfg, config_file):
    assert cfg.service_date == "2025-01-12"
    cfg.service_date = "2026-06-07"
    assert Configs(config_file).service_date == "2026-06-07"


def test_service_date_obj(cfg):
    assert cfg.service_date_obj == date(2025, 1, 12)


# --------------------------------------------------
# Typed property accessors (previously broken keys)
# --------------------------------------------------
def test_threshold_properties(cfg):
    assert cfg.announcement_threshold == 0.8
    assert cfg.preacher_threshold == 0.6


def test_trim_logic_properties(cfg):
    assert cfg.min_continuous_seconds == 10
    assert cfg.end_ratio_threshold == 0.9


def test_audio_and_diarization_properties(cfg):
    assert cfg.sample_rate == 48000
    assert cfg.diarization_rate == 1.4


def test_speaker_start_seconds(cfg):
    assert cfg.speaker_start_seconds == {
        "Pre svc": 5,
        "Worship": 30,
        "Announcement": 50,
        "Preacher": 70,
    }


def test_set_speaker_start(cfg):
    cfg.set_speaker_start("Worship", 35)
    assert cfg.speaker_start_seconds["Worship"] == 35


# --------------------------------------------------
# Naming helpers
# --------------------------------------------------
def test_naming_helpers(cfg):
    assert cfg.similarity_filename() == "similarity_2025-01-12.txt"
    assert cfg.trimmed_video_filename() == "2025-01-12_sermon.mp4"
    assert cfg.trimmed_audio_filename() == "2025-01-12_sermon.mp3"


# --------------------------------------------------
# Logging
# --------------------------------------------------
def test_log_level(cfg):
    assert cfg.log_level == logging.DEBUG


def test_log_file_path(cfg):
    assert cfg.log_file == Path("logs") / "2025-01-12_pipeline.log"


# --------------------------------------------------
# Local git-ignored overlay (config.local.yaml)
# --------------------------------------------------
def test_local_overlay_overrides_main(tmp_path):
    main = tmp_path / "config.yaml"
    _write_yaml(main, {"metadata": {"service_date": "2025-01-01"}, "audio": {"sample_rate": 1}})
    _write_yaml(tmp_path / "config.local.yaml",
                {"urls": {"youtube": "https://overlay/yt"}, "audio": {"sample_rate": 999}})

    cfg = Configs(main)
    assert cfg.get("urls.youtube") == "https://overlay/yt"   # present only in overlay
    assert cfg.get("audio.sample_rate") == 999               # overlay wins over main


def test_local_overlay_falls_back_to_main_for_absent_keys(tmp_path):
    # Overlay has a `urls` dict but not `jp`: the lookup must still fall through
    # to main for `urls.jp` rather than stopping at the overlay's `urls` node.
    main = tmp_path / "config.yaml"
    _write_yaml(main, {"metadata": {"service_date": "x"}, "urls": {"jp": "main-jp"}})
    _write_yaml(tmp_path / "config.local.yaml", {"urls": {"youtube": "overlay-yt"}})

    cfg = Configs(main)
    assert cfg.get("urls.youtube") == "overlay-yt"   # from overlay
    assert cfg.get("urls.jp") == "main-jp"           # overlay lacks jp -> main
    assert cfg.get("urls.missing", "d") == "d"       # neither -> default


def test_save_never_writes_overlay_keys_back(tmp_path):
    main = tmp_path / "config.yaml"
    _write_yaml(main, {"metadata": {"service_date": "x", "title": ""}})
    _write_yaml(tmp_path / "config.local.yaml", {"urls": {"youtube": "secret"}})

    cfg = Configs(main)
    cfg.update("metadata.title", "hi")               # triggers save()

    raw = yaml.safe_load(main.read_text(encoding="utf-8"))
    assert "urls" not in raw                          # overlay never leaks into config.yaml
    assert raw["metadata"]["title"] == "hi"
    assert cfg.get("urls.youtube") == "secret"        # still readable via the overlay


def test_no_overlay_file_is_backward_compatible(tmp_path):
    main = tmp_path / "config.yaml"
    _write_yaml(main, {"metadata": {"service_date": "x"}, "urls": {"youtube": "main-yt"}})
    cfg = Configs(main)                               # no config.local.yaml present
    assert cfg.get("urls.youtube") == "main-yt"


def test_reload_refreshes_overlay(tmp_path):
    main = tmp_path / "config.yaml"
    _write_yaml(main, {"metadata": {"service_date": "x"}})
    overlay = tmp_path / "config.local.yaml"
    _write_yaml(overlay, {"urls": {"youtube": "v1"}})

    cfg = Configs(main)
    assert cfg.get("urls.youtube") == "v1"
    _write_yaml(overlay, {"urls": {"youtube": "v2"}})
    cfg.reload()
    assert cfg.get("urls.youtube") == "v2"
