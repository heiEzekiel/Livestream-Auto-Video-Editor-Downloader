from datetime import datetime
from pathlib import Path
from video_utils import VideoUtils
from utils import CommonUtils
from configs import Configs
from logger import setup_logger

cfg = Configs(Path("./config.yaml"))

utils = CommonUtils()
v_utils = VideoUtils()


def now_local() -> datetime:
    """Current time in the service's timezone (config: timezone, default Asia/Singapore).
    Falls back to system-local time if the tz database isn't available."""
    tz_name = cfg.get("timezone", "Asia/Singapore")
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now()

if __name__ == "__main__":
    #1. get today's date
    cfg.service_date = utils.get_today_date_str()
    #cfg.service_date = "2025-11-09"
    logger = setup_logger(cfg.log_file, cfg.log_level)
    
    if not utils.is_date_sunday(cfg.service_date):
        logger.error(f"Today is not a Sunday! {cfg.service_date}")
        exit()
        
    logger.info("Pipeline started")
    logger.info(f"Service date: {cfg.service_date}")
        
    logger.info(f"Starting Job for {cfg.service_date} livestream download")
    
    #2. make folders for today's job run
    utils.make_directory(Path(str(str(cfg.get_path("path.archived")) + "/" + cfg.service_date)))
    utils.make_directory(Path(str(str(cfg.get_path("path.downloaded")) + "/" + cfg.service_date)))
    utils.make_directory(Path(str(str(cfg.get_path("path.trimmed")) + "/" + cfg.service_date)))
    
    #3. resolve the livestream URL for today's sermon.
    #   The pipeline is run several times a day (cron) for redundancy, so each run
    #   RE-SCRAPES and refreshes config.yaml with the latest available URL — it no
    #   longer caches the first run's URL. A manual override short-circuits scraping;
    #   a transient empty scrape falls back to the last-known URL.
    override_url = cfg.get("metadata.override_livestream")
    if override_url:
        livestream_url = override_url
        livestream_title = cfg.get("metadata.title") or "(manual override)"
        cfg.update("metadata.livestream", livestream_url)
        logger.info(f"Using manual override livestream URL: {livestream_url}")
        print(livestream_url)
    else:
        url_date = utils.format_url_date_str(cfg.service_date)
        videos = v_utils.get_upcoming_streams(cfg.get("urls.youtube"), url_date)

        if len(videos) == 0:
            # Don't lose a URL we already resolved on an earlier run of the day.
            previous_url = cfg.get("metadata.livestream")
            if previous_url:
                logger.warning(
                    f"No livestream services found for {cfg.service_date}; "
                    f"reusing last-known URL: {previous_url}"
                )
                print(previous_url)
                exit()
            logger.error(f"No livestream services found for {cfg.service_date}!")
            exit()

        # There are several services (URLs) on a Sunday. Pick the LATEST one that
        # has already started (is live/available) at this cron run; if none have
        # started yet, the earliest upcoming one is returned.
        upcoming_livestream = v_utils.pick_latest_started(videos, now_local())
        livestream_title = upcoming_livestream['title'].encode('ascii', 'ignore').decode('ascii').strip()
        livestream_url = upcoming_livestream['url']

        # Always refresh config.yaml with the latest available URL/title.
        cfg.update("metadata.title", livestream_title)
        cfg.update("metadata.livestream", livestream_url)

        logger.info(f"Latest available livestream -> \n Title: {livestream_title} \n URL: {livestream_url}")
        print(livestream_url)
