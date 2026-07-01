from pathlib import Path
from audio_utils import AudioUtils
from video_utils import VideoUtils
from utils import CommonUtils
from configs import Configs
from logger import setup_logger
from resource_monitor import ResourceMonitor, log_environment
import asr_transcribe
import asr_local

cfg = Configs(Path("./config.yaml"))
utils = CommonUtils()
a_utils = AudioUtils()
v_utils = VideoUtils()
logger = setup_logger(cfg.log_file, cfg.log_level)

logger.info("Pipeline started")
logger.info(f"Service date: {cfg.service_date}")
log_environment()  # one-line host banner (RAM/CPU/temp) for the Pi

if __name__ == "__main__":
    #cfg.service_date =  "2025-12-14"
    main_path = Path("./")
    downloaded_path = Path(f"./downloaded/{cfg.service_date}")
    trimmed_path = Path(f"./trimmed/{cfg.service_date}")
    file_date = cfg.service_date
    min_bytes = cfg.get("download.min_healthy_mb", 400) * 1024 * 1024

    #5. locate the source video. Resume from an already-downloaded healthy file for
    #   today if present (e.g. a prior run downloaded but failed to trim) so we don't
    #   re-download the whole broadcast; otherwise move the fresh download out of root.
    existing = None
    if downloaded_path.is_dir():
        cands = [p for p in downloaded_path.glob("*.mp4") if "sermon" not in p.name.lower()]
        cands = [p for p in cands if p.stat().st_size >= min_bytes]
        if cands:
            existing = max(cands, key=lambda p: p.stat().st_size)

    if existing is not None:
        mp4_path = existing
        logger.info(f"Resuming from existing download: {mp4_path}")
    else:
        file_details = utils.get_latest_mp4(main_path)
        if not file_details:
            logger.error("Livestream failed to download! Please try next service")
            print("FALSE")
            exit()

        mp4_file, downloaded_date = file_details[0], file_details[1]
        if downloaded_date != cfg.service_date:
            logger.error(
                f"Livestream file is incorrect! Service Date: {cfg.service_date} | "
                f"File Date: {downloaded_date}"
            )
            print("FALSE")
            exit()

        downloaded_path.mkdir(parents=True, exist_ok=True)
        old_path = Path(str(main_path) + f"/{str(mp4_file)}")
        new_path = Path(str(downloaded_path) + f"/{file_date} " + mp4_file)
        old_path.rename(new_path)
        if not new_path.exists():
            logger.error(f"Failed to move downloaded file: {old_path}")
            print("FALSE")
            exit()
        logger.info(f"Livestream file renamed to: {new_path}")
        mp4_path = new_path

    #6. convert mp4 into mp3
    mp3_path = Path(str(downloaded_path) + f"/{file_date}" + ".mp3")
    with ResourceMonitor("convert"):
        mp3_file = a_utils.convert_audio_to_mp3(mp4_path, mp3_path)

    #7-9. detect sermon start/end. Primary: self-hosted ASR (transcribe + hybrid
    #     heuristic/local-LLM detector, no cloud API). Fallback: similarity
    #     diarization (Resemblyzer) if ASR is disabled, errors, or finds nothing.
    start_timestamp = end_timestamp = None

    if cfg.get("asr.enabled", True):
        try:
            #7. transcribe audio (faster-whisper, cached JSONL)
            transcript_path = Path(str(downloaded_path) + f"/{file_date}.jsonl")
            with ResourceMonitor("transcribe"):
                asr_transcribe.transcribe(
                    mp3_file, transcript_path, cfg.get("asr.whisper_model", "tiny")
                )

            #8. detect sermon boundaries (hybrid; local LLM only for low-confidence)
            with ResourceMonitor("detect"):
                asr_result = asr_local.detect(transcript_path)

            # accept only a sane span (a bad local-LLM boundary must not trim a broken video)
            if asr_result is not None and 0 <= asr_result.start < asr_result.end:
                start_pad = cfg.get("asr.start_padding", 5)
                end_pad = cfg.get("asr.end_padding", 5)
                start_timestamp = max(0, asr_result.start - start_pad)
                end_timestamp = asr_result.end + end_pad
                logger.info(
                    f"ASR sermon span: {start_timestamp}s -> {end_timestamp}s "
                    f"(detected {asr_result.start}-{asr_result.end}s, method={asr_result.method}, "
                    f"start_conf={asr_result.start_conf}, end_conf={asr_result.end_conf})"
                )
            elif asr_result is None:
                logger.warning("ASR found no sermon; falling back to diarization.")
            else:
                logger.warning(
                    f"ASR returned an invalid span ({asr_result.start}-{asr_result.end}); "
                    "falling back to diarization."
                )
        except Exception:
            logger.exception("ASR pipeline failed; falling back to diarization.")

    if start_timestamp is None:
        #7b-9b. fallback: similarity diarization (heaviest stage: full audio + embeddings)
        try:
            speaker_segments = a_utils.speaker_segments_generator(mp3_path)
            with ResourceMonitor("diarization"):
                per_second_similarity = a_utils.run_diarization(mp3_file, speaker_segments)

            sim_txt_file = Path(str(downloaded_path) + "/" +
                                cfg.get("naming.similarity_output").replace("{date}", file_date))
            CommonUtils.save_text_file(sim_txt_file, per_second_similarity)
            data_lines = CommonUtils.load_text_file(sim_txt_file)
            logger.debug(f"Length of Data Lines: {len(data_lines)}")
            start_timestamp, end_timestamp = a_utils.get_trim_range(data_lines)
            logger.info(f"Diarization sermon span: {start_timestamp}s -> {end_timestamp}s")
        except Exception:
            logger.exception("Both ASR and diarization failed; aborting without trimming.")
            print("FALSE")
            exit()

    #10. trim video to the timestamp
    trimmed_path.mkdir(parents=True, exist_ok=True)  # don't depend on pre-setup having created it
    trimmed_video_path = str(trimmed_path) + "/"  + cfg.get("naming.trimmed_video").replace("{date}",file_date)
    trimmed_video_path = Path(trimmed_video_path)

    with ResourceMonitor("trim"):
        trimmed_video = v_utils.trim_video(
            mp4_path,
            trimmed_video_path,
            start_timestamp,
            end_timestamp
        )
    logger.info(f"Trimmed video saved to: {trimmed_video}")
        
    #reset 
    cfg.update("metadata.title", "")
    cfg.update("metadata.livestream", "")

    print("TRUE")
    
    #11. upload full video to Youtube
    #12. get url of youtube video
    #13. upload trimmed video with url to telegram
    
    #14. check jp website for last weeks sermon title and update msg
    