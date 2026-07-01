"""Tests for CommonUtils (utils.py)."""
import os
import re

import pytest

from utils import CommonUtils


# --------------------------------------------------
# auto_convert_file_size
# --------------------------------------------------
@pytest.mark.parametrize(
    "size_bytes, expected",
    [
        (0, "0B"),
        (500, "500.0 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024 ** 2, "1.0 MB"),
        (400 * 1024 ** 2, "400.0 MB"),
        (1024 ** 3, "1.0 GB"),
    ],
)
def test_auto_convert_file_size(size_bytes, expected):
    assert CommonUtils.auto_convert_file_size(size_bytes) == expected


# --------------------------------------------------
# Date helpers
# --------------------------------------------------
@pytest.mark.parametrize(
    "date_str, expected",
    [
        ("2025-11-09", "9 Nov 2025"),
        ("2025-12-19", "19 Dec 2025"),
        ("2025-01-01", "1 Jan 2025"),
    ],
)
def test_format_url_date_str(date_str, expected):
    assert CommonUtils.format_url_date_str(date_str) == expected


def test_is_date_sunday():
    assert CommonUtils.is_date_sunday("2024-01-07") is True   # Sunday
    assert CommonUtils.is_date_sunday("2024-01-08") is False  # Monday


def test_get_today_date_str_format():
    today = CommonUtils.get_today_date_str()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", today)


# --------------------------------------------------
# mp4 discovery
# --------------------------------------------------
def test_get_count_of_mp4(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi")
    assert CommonUtils.get_count_of_mp4(tmp_path) == 2


def test_get_count_of_mp4_empty(tmp_path):
    assert CommonUtils.get_count_of_mp4(tmp_path) == 0


def test_get_latest_mp4_returns_newest(tmp_path):
    old = tmp_path / "old.mp4"
    new = tmp_path / "new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    # Force deterministic modification times
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    result = CommonUtils.get_latest_mp4(tmp_path)
    assert result is not None
    name, date_only, time_only = result
    assert name.endswith("new.mp4")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_only)
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", time_only)


def test_get_latest_mp4_none_when_empty(tmp_path):
    assert CommonUtils.get_latest_mp4(tmp_path) is None


# --------------------------------------------------
# Text file round-trip
# --------------------------------------------------
def test_save_and_load_text_file(tmp_path):
    path = tmp_path / "sub" / "similarity.txt"
    data = {
        0: {"Preacher": 0.5, "Worship": 0.1},
        65: {"Preacher": 0.95, "Worship": 0.2},
    }
    CommonUtils.save_text_file(path, data)

    assert path.exists()  # parent dir was created
    lines = CommonUtils.load_text_file(path)
    assert lines == [
        "0000s | Preacher: 0.50 | Worship: 0.10",
        "0065s | Preacher: 0.95 | Worship: 0.20",
    ]


# --------------------------------------------------
# Directory helpers
# --------------------------------------------------
def test_make_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    CommonUtils.make_directory(target)
    assert target.is_dir()


def test_delete_folder_removes_nested_contents(tmp_path):
    folder = tmp_path / "downloaded"
    nested = folder / "2025-01-12"
    nested.mkdir(parents=True)
    (nested / "video.mp4").write_bytes(b"x")
    (folder / "top.txt").write_text("hi")

    CommonUtils.delete_folder(folder)
    assert not folder.exists()


def test_delete_folder_noop_when_missing(tmp_path):
    # Should not raise on a path that does not exist
    CommonUtils.delete_folder(tmp_path / "does_not_exist")
