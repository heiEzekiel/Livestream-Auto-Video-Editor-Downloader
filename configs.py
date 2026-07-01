from pathlib import Path
import yaml
from datetime import date
import logging
from typing import Any

_MISSING = object()  # sentinel: key absent (distinct from a stored None)


class Configs:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        # Optional git-ignored overlay next to the main config, e.g.
        # config.yaml -> config.local.yaml. Holds deployment-specific / private
        # values (channel urls, api keys) that must not be committed. Values here
        # take precedence on reads and are NEVER written back to config.yaml.
        self.local_path = config_path.with_name(
            f"{config_path.stem}.local{config_path.suffix}")
        self._data = self._load()
        self._local = self._load_local()

    # -------------------------
    # Internal load / save
    # -------------------------
    def _load(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Invalid config format")

        return data

    def _load_local(self) -> dict:
        """Load the optional git-ignored overlay; empty dict if absent/invalid."""
        if not self.local_path.exists():
            return {}
        with open(self.local_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def save(self):
        """Write current config back to YAML file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self._data,
                f,
                sort_keys=False,
                default_flow_style=False
            )

    def reload(self):
        """Reload config (and the local overlay) from disk."""
        self._data = self._load()
        self._local = self._load_local()

    # -------------------------
    # Generic readers
    # -------------------------
    @staticmethod
    def _lookup(data: dict, keys: list[str]) -> Any:
        """Walk ``keys`` through ``data``; return the value or ``_MISSING``."""
        node = data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return _MISSING
            node = node[key]
        return node

    def get(self, path: str, default: Any = None) -> Any:
        """
        Read config value using dot notation. The git-ignored overlay
        (config.local.yaml) takes precedence over the main config.

        Example:
            cfg.get("metadata.service_date")
            cfg.get("audio.sample_rate", 48000)
        """
        keys = path.split(".")
        value = self._lookup(self._local, keys)   # overlay wins
        if value is _MISSING:
            value = self._lookup(self._data, keys)
        return default if value is _MISSING else value

    def get_path(self, path: str, default: Any = None) -> Path | None:
        """
        Read config value and return it as a Path.
        """
        value = self.get(path, default)
        if value is None:
            return None
        return Path(value)


    # -------------------------
    # Generic updater
    # -------------------------
    def update(self, path: str, value: Any, autosave=True):
        """
        Update config using dot notation.
        Example:
            cfg.update("metadata.service_date", "2025-01-12")
        """
        keys = path.split(".")
        node = self._data

        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]

        node[keys[-1]] = value

        if autosave:
            self.save()

    # -------------------------
    # Metadata
    # -------------------------
    @property
    def service_date(self) -> str:
        return self._data["metadata"]["service_date"]

    @service_date.setter
    def service_date(self, value: str):
        self._data["metadata"]["service_date"] = value
        self.save()

    @property
    def service_date_obj(self) -> date:
        return date.fromisoformat(self.service_date)

    # -------------------------
    # Naming helpers
    # -------------------------
    def similarity_filename(self) -> str:
        return self._format(self._data["naming"]["similarity_output"])

    def trimmed_video_filename(self) -> str:
        return self._format(self._data["naming"]["trimmed_video"])

    def trimmed_audio_filename(self) -> str:
        return self._format(self._data["naming"]["trimmed_audio"])

    def _format(self, template: str) -> str:
        return template.format(date=self.service_date)

    # -------------------------
    # Paths
    # -------------------------
    @property
    def trimmed_dir(self) -> Path:
        return Path(self._data["path"]["trimmed"])

    @property
    def archive_dir(self) -> Path:
        return Path(self._data["path"]["archive"])

    @property
    def downloaded_dir(self) -> Path:
        return Path(self._data["path"]["downloaded"])

    # -------------------------
    # Audio
    # -------------------------
    @property
    def sample_rate(self) -> int:
        return self._data["audio"]["sample_rate"]

    # -------------------------
    # Diarization
    # -------------------------
    @property
    def diarization_rate(self) -> int:
        return self._data["diarization"]["rate"]

    @property
    def announcement_threshold(self) -> float:
        return self._data["diarization"]["confidence_thresholds"]["announcement"]

    @property
    def preacher_threshold(self) -> float:
        return self._data["diarization"]["confidence_thresholds"]["preacher"]

    # -------------------------
    # Speakers (UPDATED to match YAML)
    # -------------------------
    @property
    def speaker_start_seconds(self) -> dict:
        """
        Returns:
        {
          "Worship": 30,
          "Announcement": 50,
          "Preacher": 70
        }
        """
        return {
            name.replace("_", " "): cfg["start"]
            for name, cfg in self._data["speakers"].items()
        }

    def set_speaker_start(self, speaker: str, start_sec: int):
        key = speaker.replace(" ", "_")
        if key not in self._data["speakers"]:
            self._data["speakers"][key] = {}
        self._data["speakers"][key]["start"] = start_sec
        self.save()

    # -------------------------
    # Trim logic
    # -------------------------
    @property
    def min_continuous_seconds(self) -> int:
        return self._data["trim_logic"]["min_continuous_seconds"]

    @property
    def end_ratio_threshold(self) -> float:
        return self._data["trim_logic"]["end_must_be_after_ratio"]

    # -------------------------
    # Logging
    # -------------------------
    @property
    def log_level(self) -> int:
        level = self._data["logging"]["level"].upper()
        return getattr(logging, level, logging.INFO)

    @property
    def log_file(self) -> Path:
        log_dir = Path(self._data["logging"]["directory"])
        name = self._format(self._data["logging"]["filename"])
        return log_dir / name
