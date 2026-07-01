"""
Shared pytest fixtures.

The application modules (audio_utils, video_utils) instantiate
`Configs(Path("./config.yaml"))` at import time, so they must be imported with
the repo root as the working directory. We chdir here once and add the root to
sys.path so tests run regardless of how pytest is invoked.
"""
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


# A self-contained config mirroring the real config.yaml schema. Tests build a
# temp file from this so they never touch (or depend on) the live config.
SAMPLE_CONFIG = {
    "metadata": {
        "service_date": "2025-01-12",
        "church": "Test Church",
        "title": "",
        "livestream": "",
    },
    "urls": {"youtube": "https://www.youtube.com/@Test/featured"},
    "logging": {"directory": "logs", "filename": "{date}_pipeline.log", "level": "DEBUG"},
    "path": {"archived": "archived", "downloaded": "downloaded", "trimmed": "trimmed"},
    "naming": {
        "similarity_output": "similarity_{date}.txt",
        "trimmed_video": "{date}_sermon.mp4",
        "trimmed_audio": "{date}_sermon.mp3",
    },
    "audio": {"sample_rate": 48000, "quiet_level": -50, "loud_level": -20},
    "diarization": {
        "rate": 1.4,
        "confidence_thresholds": {"announcement": 0.8, "preacher": 0.6},
        "self_enroll": {
            "gap_bridge_seconds": 20,
            "min_run_seconds": 300,
            "refine_keep_ratio": 0.5,
        },
    },
    "segment_duration": 256,
    "speakers": {
        "Pre_svc": {"start": 5},
        "Worship": {"start": 30},
        "Announcement": {"start": 50},
        "Preacher": {"start": 70},
    },
    "trim_logic": {
        "smooth_window_seconds": 15,
        "high_percentile": 65,
        "low_percentile": 40,
        "shoulder_ratio": 0.9,
        "dominance_margin": 0.0,
        "dominance_sustain_seconds": 45,
        "min_continuous_seconds": 10,
        "core_gap_seconds": 120,
        "start_gap_seconds": 60,
        "end_gap_seconds": 300,
        "start_padding": 15,
        "padding": 8,
        "end_must_be_after_ratio": 0.9,
    },
}


@pytest.fixture
def config_file(tmp_path):
    """Write SAMPLE_CONFIG to a temp YAML file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(SAMPLE_CONFIG, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def cfg(config_file):
    """A Configs instance backed by the temp config file."""
    from configs import Configs

    return Configs(config_file)
