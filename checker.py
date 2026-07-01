from pathlib import Path
from utils import CommonUtils
from configs import Configs
from logger import setup_logger

cfg = Configs(Path("./config.yaml"))
utils = CommonUtils()


def largest_mp4(folder: Path, exclude_sermon: bool = False):
    """Largest .mp4 in ``folder`` (by size), or None. Set ``exclude_sermon`` to
    skip trimmed outputs named '*_sermon.mp4' when looking at a download folder."""
    if not folder.is_dir():
        return None
    cands = list(folder.glob("*.mp4"))
    if exclude_sermon:
        cands = [p for p in cands if "sermon" not in p.name.lower()]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def classify(downloaded_path: Path, trimmed_path: Path, min_bytes: int):
    """
    Decide the pipeline state for a service date. Returns ``(status, to_delete)``:

      * ``"TRUE"``     -- healthy download AND healthy trim -> already done.
      * ``"PROCESS"``  -- healthy download, missing/corrupt trim -> (re)process.
                          The DOWNLOAD IS KEPT so a crashed/interrupted run can
                          resume without re-downloading the whole broadcast.
      * ``"DOWNLOAD"`` -- no usable download -> fetch from scratch.

    ``to_delete`` is the list of folders that should be removed before the next
    step (only genuinely corrupt/incomplete artifacts are ever deleted).
    """
    dl = largest_mp4(downloaded_path, exclude_sermon=True)
    dl_size = dl.stat().st_size if dl else 0

    if dl is None:
        return "DOWNLOAD", []                       # nothing downloaded yet
    if dl_size < min_bytes:
        return "DOWNLOAD", [downloaded_path, trimmed_path]   # incomplete/stale
    if len(list(downloaded_path.glob("*.mp4"))) > 1:
        return "DOWNLOAD", [downloaded_path, trimmed_path]   # ambiguous

    tr = largest_mp4(trimmed_path)
    tr_size = tr.stat().st_size if tr else 0

    if tr is None:
        return "PROCESS", []                        # downloaded, not trimmed yet
    if tr_size < min_bytes or (dl_size - tr_size) < (dl_size * 0.1):
        return "PROCESS", [trimmed_path]            # trim looks corrupt -> re-trim
    return "TRUE", []


if __name__ == "__main__":
    #1. get today's date
    cfg.service_date = utils.get_today_date_str()
    logger = setup_logger(cfg.log_file, cfg.log_level)

    min_bytes = cfg.get("download.min_healthy_mb", 400) * 1024 * 1024

    logger.info(f"Checker for livestream downloaded for {cfg.service_date}")
    downloaded_path = Path(str(cfg.get_path("path.downloaded")) + "/" + cfg.service_date)
    trimmed_path = Path(str(cfg.get_path("path.trimmed")) + "/" + cfg.service_date)

    status, to_delete = classify(downloaded_path, trimmed_path, min_bytes)

    for folder in to_delete:
        logger.warning(f"Removing corrupt/incomplete artifacts: {folder}")
        utils.delete_folder(folder)

    # NOTE: a healthy download is NEVER deleted just because the trim is missing,
    # so an interrupted run resumes (PROCESS) instead of re-downloading, and a
    # concurrent run can't wipe an in-progress download (run.sh also locks).
    msg = {
        "TRUE": "Already downloaded and trimmed; nothing to do.",
        "PROCESS": "Download present but not trimmed; will resume at processing.",
        "DOWNLOAD": "No usable download; will download from scratch.",
    }[status]
    logger.info(msg)

    # stdout = the status token (logs go to stderr); run.sh reads this.
    print(status)
    logger.info("Checker completed")
