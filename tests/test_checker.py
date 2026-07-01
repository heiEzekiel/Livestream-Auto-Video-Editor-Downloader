"""
Tests for checker.classify() — the 3-state pipeline decision.

The decision must never delete a healthy download just because the trim is
missing (so an interrupted run resumes instead of re-downloading the whole
broadcast), and must only delete genuinely corrupt/incomplete artifacts.
"""
from pathlib import Path

import pytest

import checker


def _mp4(folder: Path, name: str, size: int):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"\0" * size)
    return p


MIN = 1000  # min "healthy" bytes for these tests


def _paths(tmp_path):
    return tmp_path / "downloaded" / "d", tmp_path / "trimmed" / "d"


# --------------------------------------------------
# largest_mp4
# --------------------------------------------------
def test_largest_mp4_missing_folder(tmp_path):
    assert checker.largest_mp4(tmp_path / "nope") is None


def test_largest_mp4_picks_biggest_and_excludes_sermon(tmp_path):
    _mp4(tmp_path, "small.mp4", 10)
    big = _mp4(tmp_path, "big.mp4", 500)
    _mp4(tmp_path, "huge_sermon.mp4", 9999)  # trimmed output, must be ignored
    assert checker.largest_mp4(tmp_path, exclude_sermon=True) == big


# --------------------------------------------------
# classify: DOWNLOAD
# --------------------------------------------------
def test_no_download_means_download(tmp_path):
    dl, tr = _paths(tmp_path)
    assert checker.classify(dl, tr, MIN) == ("DOWNLOAD", [])


def test_undersized_download_is_corrupt_download(tmp_path):
    dl, tr = _paths(tmp_path)
    _mp4(dl, "x.mp4", 10)            # below MIN -> incomplete
    status, to_delete = checker.classify(dl, tr, MIN)
    assert status == "DOWNLOAD"
    assert set(to_delete) == {dl, tr}


def test_multiple_downloads_is_ambiguous_download(tmp_path):
    dl, tr = _paths(tmp_path)
    _mp4(dl, "a.mp4", 2000)
    _mp4(dl, "b.mp4", 2000)         # two healthy candidates -> can't tell which
    status, to_delete = checker.classify(dl, tr, MIN)
    assert status == "DOWNLOAD"
    assert set(to_delete) == {dl, tr}


# --------------------------------------------------
# classify: PROCESS (keeps the download)
# --------------------------------------------------
def test_healthy_download_no_trim_is_process_and_keeps_download(tmp_path):
    dl, tr = _paths(tmp_path)
    _mp4(dl, "x.mp4", 5000)
    status, to_delete = checker.classify(dl, tr, MIN)
    assert status == "PROCESS"
    assert to_delete == []          # download must NOT be deleted


def test_undersized_trim_is_process_deleting_only_trim(tmp_path):
    dl, tr = _paths(tmp_path)
    _mp4(dl, "x.mp4", 5000)
    _mp4(tr, "x_sermon.mp4", 10)    # trim below MIN -> corrupt
    status, to_delete = checker.classify(dl, tr, MIN)
    assert status == "PROCESS"
    assert to_delete == [tr]


def test_trim_not_meaningfully_smaller_is_process(tmp_path):
    dl, tr = _paths(tmp_path)
    _mp4(dl, "x.mp4", 10000)
    _mp4(tr, "x_sermon.mp4", 9500)  # only 5% smaller -> trim almost certainly failed
    status, to_delete = checker.classify(dl, tr, MIN)
    assert status == "PROCESS"
    assert to_delete == [tr]


# --------------------------------------------------
# classify: TRUE
# --------------------------------------------------
def test_healthy_download_and_trim_is_true(tmp_path):
    dl, tr = _paths(tmp_path)
    _mp4(dl, "x.mp4", 10000)
    _mp4(tr, "x_sermon.mp4", 5000)  # well above MIN and >10% smaller
    assert checker.classify(dl, tr, MIN) == ("TRUE", [])
