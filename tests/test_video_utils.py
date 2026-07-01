"""
Tests for VideoUtils (video_utils.py).

Network (requests) and the ffmpeg subprocess are mocked so these run offline
and without ffmpeg installed.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

import video_utils
from video_utils import VideoUtils


def _fake_response(text):
    class FakeResponse:
        def __init__(self, body):
            self.text = body

        def raise_for_status(self):
            return None

    return FakeResponse(text)


def _page_with_data(data):
    return "prefix var ytInitialData = " + json.dumps(data) + ";</script> suffix"


# --------------------------------------------------
# get_upcoming_streams
# --------------------------------------------------
def test_get_upcoming_streams_extracts_match(monkeypatch):
    data = {
        "contents": [
            {
                "lockupViewModel": {
                    "contentId": "abc123",
                    "metadata": {
                        "lockupMetadataViewModel": {
                            "title": {"content": "Sunday Service | 10 am | 9 Nov 2025"}
                        }
                    },
                }
            }
        ]
    }
    monkeypatch.setattr(
        video_utils.requests, "get", lambda *a, **k: _fake_response(_page_with_data(data))
    )

    streams = VideoUtils().get_upcoming_streams("http://example", "9 Nov 2025")

    assert streams == [
        {
            "title": "Sunday Service | 10 am | 9 Nov 2025",
            "url": "https://www.youtube.com/watch?v=abc123",
        }
    ]


def test_get_upcoming_streams_filters_wrong_date(monkeypatch):
    data = {
        "lockupViewModel": {
            "contentId": "xyz",
            "metadata": {
                "lockupMetadataViewModel": {
                    "title": {"content": "Sunday Service | 10 am | 2 Nov 2025"}
                }
            },
        }
    }
    monkeypatch.setattr(
        video_utils.requests, "get", lambda *a, **k: _fake_response(_page_with_data(data))
    )
    assert VideoUtils().get_upcoming_streams("http://example", "9 Nov 2025") == []


def test_get_upcoming_streams_no_initial_data(monkeypatch):
    monkeypatch.setattr(
        video_utils.requests, "get", lambda *a, **k: _fake_response("no data here")
    )
    assert VideoUtils().get_upcoming_streams("http://example", "9 Nov 2025") == []


# --------------------------------------------------
# parse_service_time
# --------------------------------------------------
@pytest.mark.parametrize("title,expected", [
    ("NCC English Service | 2.30pm | 8 Feb 2026", (14, 30)),
    ("NCC English Service | 11.30am | 8 Feb 2026", (11, 30)),
    ("NCC English Service | 9 am | 9 Nov 2025", (9, 0)),
    ("NCC English Service | 5pm | 9 Nov 2025", (17, 0)),
    ("NCC English Service | 12pm | 9 Nov 2025", (12, 0)),   # noon
    ("NCC English Service | 12am | 9 Nov 2025", (0, 0)),    # midnight
    ("NCC English Service | morning | 9 Nov 2025", None),   # no time
])
def test_parse_service_time(title, expected):
    assert VideoUtils.parse_service_time(title) == expected


# --------------------------------------------------
# pick_latest_started (which of several services to download)
# --------------------------------------------------
def _svc(t):
    return {"title": f"NCC English Service | {t} | 9 Nov 2025",
            "url": f"https://www.youtube.com/watch?v={t}"}


SERVICES = [_svc("9.30am"), _svc("11.30am"), _svc("2.30pm"), _svc("5pm")]


@pytest.mark.parametrize("hour,minute,expected_time", [
    (12, 0, "11.30am"),   # midday -> latest started is 11.30am
    (18, 0, "5pm"),       # evening -> 5pm
    (14, 45, "2.30pm"),   # just after 2.30pm
    (9, 35, "9.30am"),    # just after first service
])
def test_pick_latest_started(hour, minute, expected_time):
    now = datetime(2025, 11, 9, hour, minute)
    chosen = VideoUtils().pick_latest_started(SERVICES, now)
    assert expected_time in chosen["title"]


def test_pick_latest_started_before_any_service_returns_earliest():
    now = datetime(2025, 11, 9, 8, 0)  # before the 9.30am service
    chosen = VideoUtils().pick_latest_started(SERVICES, now)
    assert "9.30am" in chosen["title"]


def test_pick_latest_started_empty_and_unparseable():
    assert VideoUtils().pick_latest_started([], datetime(2025, 11, 9, 12, 0)) is None
    only = [{"title": "no time here", "url": "u"}]
    assert VideoUtils().pick_latest_started(only, datetime(2025, 11, 9, 12, 0)) == only[0]


# --------------------------------------------------
# trim_video
# --------------------------------------------------
def test_trim_video_stream_copy(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        video_utils.subprocess, "run", lambda cmd, check: captured.update(cmd=cmd, check=check)
    )

    out = VideoUtils().trim_video(Path("in.mp4"), Path("out.mp4"), 10, 20)

    assert out == Path("out.mp4")
    cmd = captured["cmd"]
    assert captured["check"] is True
    assert cmd[:1] == ["ffmpeg"]
    assert "-ss" in cmd and "10" in cmd
    assert "-to" in cmd and "20" in cmd
    assert "-c" in cmd and "copy" in cmd  # stream copy, no re-encode
    assert cmd[-1] == "out.mp4"


def test_trim_video_reencode(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        video_utils.subprocess, "run", lambda cmd, check: captured.update(cmd=cmd)
    )

    VideoUtils().trim_video(Path("in.mp4"), Path("out.mp4"), 0, 5, reencode=True)

    cmd = captured["cmd"]
    assert "libx264" in cmd and "aac" in cmd
    assert "copy" not in cmd
