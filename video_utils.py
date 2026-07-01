import json
import logging
import re
import requests
import subprocess
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

from configs import Configs

cfg = Configs(Path("./config.yaml"))
logger = logging.getLogger("sermon_pipeline")

# Service time embedded in the title's middle field, e.g. "2.30pm", "11.30am", "9 am".
_SERVICE_TIME = re.compile(r"(\d{1,2})(?:[.:](\d{2}))?\s*([ap]m)", re.IGNORECASE)

class VideoUtils:
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path

    # --------------------------------------------------
    # Trim video by timestamps
    # --------------------------------------------------
    def trim_video(
        self,
        input_video: Path,
        output_video: Path,
        start_sec: int,
        end_sec: int,
        reencode: bool = False
    ) -> Path:
        """
        Trim a video using FFmpeg.

        - start_sec / end_sec are in seconds
        - reencode=False uses stream copy (fast, keyframe-aligned)
        - reencode=True is frame-accurate but slower
        """

        cmd = [
            self.ffmpeg,
            "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", str(input_video),
        ]

        if reencode:
            cmd += [
                "-c:v", "libx264",
                "-c:a", "aac"
            ]
        else:
            cmd += ["-c", "copy"]

        cmd.append(str(output_video))

        subprocess.run(cmd, check=True)
        return output_video

    def get_upcoming_streams(self, channel_url: str, date_str: str) -> list:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        resp = requests.get(channel_url, headers=headers)
        resp.raise_for_status()
        html = resp.text

        match = re.search(r"var ytInitialData = ({.*?});</script>", html)
        if not match:
            print("Could not find ytInitialData in the page.")
            return []

        initial_data = json.loads(match.group(1))
        streams = []

        def crawl(node):
            if isinstance(node, dict):
                lvm = node.get("lockupViewModel")
                if lvm:
                    video_id = lvm.get("contentId")
                    title = (
                        lvm.get("metadata", {})
                        .get("lockupMetadataViewModel", {})
                        .get("title", {})
                        .get("content", "")
                    )
                    if (
                        video_id
                        and date_str in title
                        and ("am" in title.lower() or "pm" in title.lower())
                        and title.count("|") == 2
                    ):
                        streams.append({
                            "title": title,
                            "url": f"https://www.youtube.com/watch?v={video_id}"
                        })

                for value in node.values():
                    crawl(value)

            elif isinstance(node, list):
                for item in node:
                    crawl(item)

        crawl(initial_data)

        logger.info(f"Found {len(streams)} services for {date_str}")

        return streams

    # --------------------------------------------------
    # Service selection (Sunday has several services / URLs)
    # --------------------------------------------------
    @staticmethod
    def parse_service_time(title: str) -> tuple[int, int] | None:
        """
        Extract the service time-of-day as (hour_24, minute) from a stream title
        like 'NCC English Service | 2.30pm | 8 Feb 2026'. Returns None if absent.
        """
        field = title.split("|")[1] if title.count("|") >= 2 else title
        m = _SERVICE_TIME.search(field) or _SERVICE_TIME.search(title)
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3).lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if not (0 <= hour < 24 and 0 <= minute < 60):
            return None
        return hour, minute

    def pick_latest_started(self, streams: list, now: datetime):
        """
        From the day's services, pick the LATEST one whose start time is at/before
        ``now`` (i.e. already live/started and therefore downloadable). If none have
        started yet, return the earliest upcoming one. ``now`` should be in the
        service's local timezone; its date is used to build each service's start.

        Returns a stream dict (with title/url) or None if the list is empty.
        """
        if not streams:
            return None

        timed = []
        for s in streams:
            t = self.parse_service_time(s.get("title", ""))
            if t is None:
                continue
            start = now.replace(hour=t[0], minute=t[1], second=0, microsecond=0)
            timed.append((start, s))

        if not timed:
            return streams[0]  # can't parse times -> fall back to first found

        timed.sort(key=lambda x: x[0])
        started = [s for start, s in timed if start <= now]
        chosen = started[-1] if started else timed[0][1]
        logger.info(
            f"Selected service '{chosen.get('title', '')}' "
            f"({len(started)}/{len(timed)} started by {now:%H:%M})"
        )
        return chosen