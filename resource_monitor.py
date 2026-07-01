"""
Compute-resource monitoring for the pipeline.

Designed for the deployment target -- a Raspberry Pi 5 (16 GB) running a full
OS plus other processes -- but works cross-platform so it can be exercised
during development on Windows. It samples, in a background thread:

  * the pipeline's own memory (this process + children such as ffmpeg/yt-dlp),
  * *system-wide* memory (so you see pressure from the OS and everything else,
    not just our footprint),
  * system-wide CPU load, and
  * on the Pi: SoC temperature and CPU frequency (to catch thermal throttling).

Usage:

    from resource_monitor import ResourceMonitor

    with ResourceMonitor("diarization") as m:
        ... heavy work ...
    # logs a one-line summary; m.summary() returns the numbers as a dict

Everything degrades gracefully: a metric the platform does not expose is simply
omitted rather than raising. psutil is the only dependency (already required).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

logger = logging.getLogger("sermon_pipeline")

_NVIDIA_SMI = shutil.which("nvidia-smi")  # None on the Pi / non-NVIDIA hosts


def _read_gpu() -> dict | None:
    """Best-effort NVIDIA GPU stats (util %, mem MB). None if no GPU/tool.

    The pipeline runs the models on CPU (Resemblyzer and faster-whisper are
    configured device='cpu'), so on a machine with a GPU this should read ~0%
    utilization — it confirms the GPU is idle rather than driving the work. On
    the Raspberry Pi there is no nvidia-smi, so this returns None and GPU
    metrics are simply omitted.
    """
    if not _NVIDIA_SMI:
        return None
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError):
        return None
    # aggregate across GPUs (max util, summed memory)
    util = 0.0
    mem_used = mem_total = 0.0
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                util = max(util, float(parts[0]))
                mem_used += float(parts[1])
                mem_total += float(parts[2])
            except ValueError:
                pass
    return {"util_pct": util, "mem_used_mb": mem_used, "mem_total_mb": mem_total}

# Pi-oriented warning thresholds (overridable per instance).
LOW_AVAILABLE_MB = 1024      # warn if system available RAM drops below this
HOT_TEMP_C = 80.0            # Pi 5 starts throttling around here


def _read_pi_temperature_c() -> float | None:
    """SoC temperature in Celsius, or None if unavailable (e.g. on Windows)."""
    # Preferred: psutil sensors (Linux).
    try:
        temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError):
        temps = {}
    for key in ("cpu_thermal", "coretemp", "soc_thermal"):
        if temps.get(key):
            return float(temps[key][0].current)
    if temps:
        first = next(iter(temps.values()))
        if first:
            return float(first[0].current)
    # Fallback: sysfs thermal zone (Raspberry Pi OS).
    zone = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return int(zone.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _read_cpu_freq_mhz() -> float | None:
    try:
        f = psutil.cpu_freq()
        return float(f.current) if f else None
    except (AttributeError, NotImplementedError, OSError):
        return None


def _load_avg_1m() -> float | None:
    """1-minute load average (Linux/macOS); None on platforms without it."""
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


@dataclass
class ResourceMonitor:
    """Context manager that samples resource usage on a background thread."""

    label: str = "stage"
    interval: float = 1.0
    low_available_mb: int = LOW_AVAILABLE_MB
    hot_temp_c: float = HOT_TEMP_C

    # collected results
    peak_proc_mb: float = 0.0
    peak_sys_used_pct: float = 0.0
    min_available_mb: float = field(default=float("inf"))
    peak_cpu_pct: float = 0.0
    peak_temp_c: float | None = None
    peak_gpu_util_pct: float | None = None
    peak_gpu_mem_mb: float | None = None
    samples: int = 0
    duration_s: float = 0.0

    def __post_init__(self):
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._warned_mem = False
        self._warned_temp = False
        vm = psutil.virtual_memory()
        self._total_mb = vm.total / 1024 / 1024

    # -- process (self + children) RSS ------------------------------------
    def _proc_tree_rss_mb(self) -> float:
        rss = self._proc.memory_info().rss
        try:
            for child in self._proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return rss / 1024 / 1024

    def _sample(self):
        vm = psutil.virtual_memory()
        proc_mb = self._proc_tree_rss_mb()
        avail_mb = vm.available / 1024 / 1024
        cpu = psutil.cpu_percent(interval=None)  # since last call, non-blocking
        temp = _read_pi_temperature_c()
        gpu = _read_gpu()

        self.peak_proc_mb = max(self.peak_proc_mb, proc_mb)
        self.peak_sys_used_pct = max(self.peak_sys_used_pct, vm.percent)
        self.min_available_mb = min(self.min_available_mb, avail_mb)
        self.peak_cpu_pct = max(self.peak_cpu_pct, cpu)
        if temp is not None:
            self.peak_temp_c = temp if self.peak_temp_c is None else max(self.peak_temp_c, temp)
        if gpu is not None:
            self.peak_gpu_util_pct = max(self.peak_gpu_util_pct or 0.0, gpu["util_pct"])
            self.peak_gpu_mem_mb = max(self.peak_gpu_mem_mb or 0.0, gpu["mem_used_mb"])
        self.samples += 1

        if avail_mb < self.low_available_mb and not self._warned_mem:
            logger.warning(
                f"[resmon:{self.label}] LOW MEMORY: {avail_mb:.0f} MB available "
                f"(< {self.low_available_mb} MB); pipeline using {proc_mb:.0f} MB."
            )
            self._warned_mem = True
        if temp is not None and temp >= self.hot_temp_c and not self._warned_temp:
            logger.warning(
                f"[resmon:{self.label}] HOT: SoC {temp:.0f}C (>= {self.hot_temp_c}C); "
                "thermal throttling likely."
            )
            self._warned_temp = True

    def _run(self):
        psutil.cpu_percent(interval=None)  # prime the CPU counter
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:  # never let monitoring crash the pipeline
                logger.debug("[resmon] sample failed", exc_info=True)

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "ResourceMonitor":
        self._t0 = time.time()
        self._sample()  # immediate baseline so very short stages still report
        self._thread = threading.Thread(target=self._run, name=f"resmon-{self.label}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
        self.duration_s = time.time() - self._t0
        logger.info(self.format_summary())
        return False

    def summary(self) -> dict:
        return {
            "label": self.label,
            "duration_s": round(self.duration_s, 1),
            "peak_proc_mb": round(self.peak_proc_mb, 1),
            "peak_sys_used_pct": round(self.peak_sys_used_pct, 1),
            "min_available_mb": round(self.min_available_mb, 1),
            "total_mb": round(self._total_mb, 1),
            "peak_cpu_pct": round(self.peak_cpu_pct, 1),
            "peak_temp_c": None if self.peak_temp_c is None else round(self.peak_temp_c, 1),
            "peak_gpu_util_pct": None if self.peak_gpu_util_pct is None else round(self.peak_gpu_util_pct, 1),
            "peak_gpu_mem_mb": None if self.peak_gpu_mem_mb is None else round(self.peak_gpu_mem_mb, 1),
            "samples": self.samples,
        }

    def format_summary(self) -> str:
        s = self.summary()
        parts = [
            f"[resmon:{self.label}] {s['duration_s']}s",
            f"pipeline_peak={s['peak_proc_mb']:.0f}MB",
            f"sys_peak={s['peak_sys_used_pct']:.0f}% of {s['total_mb']/1024:.1f}GB",
            f"min_free={s['min_available_mb']:.0f}MB",
            f"cpu_peak={s['peak_cpu_pct']:.0f}%",
        ]
        if s["peak_temp_c"] is not None:
            parts.append(f"temp_peak={s['peak_temp_c']:.0f}C")
        if s["peak_gpu_util_pct"] is not None:
            parts.append(f"gpu_peak={s['peak_gpu_util_pct']:.0f}% ({s['peak_gpu_mem_mb']:.0f}MB)")
        return " | ".join(parts)


def snapshot() -> dict:
    """One-off point-in-time reading of the key metrics (for startup banners)."""
    vm = psutil.virtual_memory()
    return {
        "total_mb": round(vm.total / 1024 / 1024, 1),
        "available_mb": round(vm.available / 1024 / 1024, 1),
        "used_pct": vm.percent,
        "cpu_count": psutil.cpu_count(logical=True),
        "load_avg_1m": _load_avg_1m(),
        "cpu_freq_mhz": _read_cpu_freq_mhz(),
        "temp_c": _read_pi_temperature_c(),
        "gpu": _read_gpu(),
    }


def log_environment():
    """Log a one-line banner describing the host (handy at pipeline start)."""
    s = snapshot()
    msg = (
        f"[resmon] host: {s['cpu_count']} CPUs, "
        f"{s['total_mb']/1024:.1f}GB RAM ({s['available_mb']:.0f}MB free), "
        f"used={s['used_pct']:.0f}%"
    )
    if s["load_avg_1m"] is not None:
        msg += f", load1m={s['load_avg_1m']:.2f}"
    if s["temp_c"] is not None:
        msg += f", temp={s['temp_c']:.0f}C"
    logger.info(msg)
    return s
