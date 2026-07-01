import logging
import os
import shutil
import matplotlib.pyplot as plt
import math
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("sermon_pipeline")

class CommonUtils:
    # --------------------------------------------------
    # File IO
    # --------------------------------------------------
    @staticmethod
    def save_text_file(file_path: Path, dict_data: dict):
        """
        Saves similarity-per-second data to a text file.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for sec, scores in dict_data.items():
                line = f"{sec:04d}s | " + " | ".join(
                    f"{spk}: {score:.2f}" for spk, score in scores.items()
                )
                f.write(line + "\n")
        logger.debug(f"Saved similarity data to {file_path}")

    @staticmethod
    def load_text_file(file_path: Path) -> list[str]:
        """
        Loads a text file and returns stripped lines.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    # --------------------------------------------------
    # Directories
    # --------------------------------------------------
    @staticmethod
    def make_directory(dir_path: Path):
        """
        Create directory if it does not exist.
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory: {dir_path}")
        
    @staticmethod
    def delete_folder(folder_path: Path):
        """
        Deletes a folder and all its contents if it exists.
        """
        if folder_path.exists():
            shutil.rmtree(folder_path)
   
    # --------------------------------------------------
    # Dates & metadata
    # --------------------------------------------------
    @staticmethod
    def get_latest_mp4(directory: Path):
        """
        Returns the most recently modified .mp4 file in a directory.
        If none found, returns None.
        """
        mp4_files = list(directory.glob("*.mp4"))

        if not mp4_files:
            return None
        
        latest_file = max(mp4_files, key=lambda p: p.stat().st_mtime)
        mod_time = os.path.getmtime(latest_file)
        full = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        file_name_only = str(latest_file).replace("ytarchive-master\\","")
        file_name_only = str(file_name_only).replace("ytarchive-master/","")
        date_only = full.split(" ")[0]
        time_only = full.split(" ")[1]

        return [file_name_only, date_only, time_only]
    
    @staticmethod
    def get_count_of_mp4(directory: Path) -> int:
        """
        Returns the count of .mp4 files in a directory.
        """
        mp4_files = list(directory.glob("*.mp4"))
        return len(mp4_files)

    @staticmethod
    def get_today_date_str() -> str:
        """
        Returns today's date as YYYY-MM-DD.
        """
        return str(datetime.now().strftime("%Y-%m-%d"))
    
    @staticmethod
    def format_url_date_str(date_str: str) -> str:
        """
        Return string converted to dd-mth-YYY
        """
        return str(datetime.strptime(date_str, "%Y-%m-%d").strftime("%d %b %Y").lstrip("0"))

    @staticmethod
    def is_date_sunday(date_str: str):
        """
        Checks if a given date string (YYYY-MM-DD) falls on a Sunday.
        """
        # Convert the string to a datetime object
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        # .weekday() returns 0 for Monday, 6 for Sunday
        return date_obj.weekday() == 6
    
    @staticmethod
    def auto_convert_file_size(size_bytes: int) -> str:
        """
        Convert bytes to a human-readable format (KB, MB, GB).
        """
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------
    @staticmethod
    def plot_diarization(data_file: Path, show_time: bool = True):
        """
        Plot diarization similarity scores from text file.
        """
        data_lines = CommonUtils.load_text_file(data_file)

        time_secs = []
        speaker_scores = {}

        for line in data_lines:
            parts = line.split("|")
            time_sec = int(parts[0].strip().replace("s", ""))
            time_secs.append(time_sec)

            for part in parts[1:]:
                spk, score = part.strip().split(":")
                score = float(score)
                speaker_scores.setdefault(spk, []).append(score)

        plt.figure(figsize=(12, 5))
        for spk, scores in speaker_scores.items():
            plt.plot(time_secs, scores, label=spk)

        plt.xlabel("Time (s)")
        plt.ylabel("Similarity Score")
        plt.title("Speaker Similarity Over Time")
        plt.legend()
        plt.tight_layout()

        if show_time:
            plt.show()
