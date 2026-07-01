import librosa
import subprocess
import numpy as np
from pathlib import Path
from resemblyzer.hparams import sampling_rate
from resemblyzer import VoiceEncoder
from configs import Configs
from utils import CommonUtils
import logging
import os

cfg = Configs(Path("./config.yaml"))
logger = logging.getLogger("sermon_pipeline")
utils = CommonUtils()

class AudioUtils:
    def __init__(self, sr: int = sampling_rate):
        self.sr = sr

    # --------------------------------------------------
    # Load audio (no trimming)
    # --------------------------------------------------
    def load_audio_no_trim(self, path: Path) -> np.ndarray:
        """
        Load audio file without trimming silence.
        """
        wav, sr = librosa.load(path, sr=None, mono=True)
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
            logger.debug("Resampled audio file.")
        logger.info("Loaded audio file.")
        return wav

    # --------------------------------------------------
    # Loudness per second
    # --------------------------------------------------
    def loudness_per_second(self, audio_path: Path) -> dict[int, float]:
        """
        Returns loudness (dBFS) per second.
        """
        y, sr = librosa.load(audio_path, sr=cfg.get("audio.sample_rate"), mono=True)

        samples_per_sec = sr
        total_secs = int(np.ceil(len(y) / samples_per_sec))

        loudness = {}

        for sec in range(total_secs):
            start = sec * samples_per_sec
            end = min((sec + 1) * samples_per_sec, len(y))

            chunk = y[start:end]
            if len(chunk) == 0:
                continue

            rms = np.sqrt(np.mean(chunk ** 2))
            dbfs = 20 * np.log10(rms + 1e-10)

            loudness[sec] = dbfs

        if not loudness:
            logger.warning(f"File: {audio_path} | No audio samples found, returning empty loudness map.")
            return loudness

        logger.info(f"File: {audio_path} | Total Seconds: {len(loudness)}s | Max dBFS: {max(loudness.values()):.2f} | Min dBFS: {min(loudness.values()):.2f}")

        return loudness

    # --------------------------------------------------
    # Convert audio using ffmpeg
    # --------------------------------------------------
    def convert_audio_to_mp3(self, mp4_file: Path, mp3_file: Path) -> Path:
        """
        Convert MP4 audio to MP3 (CBR 192k, 48kHz).
        """
        if os.path.exists(mp3_file):
            return mp3_file
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(mp4_file),
            "-vn",
            "-c:a", "libmp3lame",
            "-ar", str(cfg.get("audio.sample_rate")),
            "-b:a", "192k",
            str(mp3_file),
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully converted video to audio {mp3_file}!")
        return mp3_file

    # --------------------------------------------------
    # Parse the similarity-per-second text format
    # --------------------------------------------------
    @staticmethod
    def parse_speaker_scores(data_lines: list[str], speaker: str = "Preacher"):
        """
        Pull out one speaker's per-second similarity scores from the saved text
        format (``0000s | Announcement: 0.50 | Preacher: 0.50``).

        Returns (secs, scores) as parallel lists ordered by second. Lines that
        do not carry the requested speaker are skipped.
        """
        secs: list[int] = []
        scores: list[float] = []

        for line in data_lines:
            parts = line.split("|")
            time_sec = int(parts[0].strip().replace("s", ""))

            for part in parts[1:]:
                spk, score = part.strip().split(":")
                if spk == speaker:
                    secs.append(time_sec)
                    scores.append(float(score))
                    break

        return secs, scores

    # --------------------------------------------------
    # Sermon span detection (plateau core + asymmetric extension)
    # --------------------------------------------------
    @staticmethod
    def _smooth(scores, window: int) -> np.ndarray:
        """Centred moving-average to damp per-second jitter before thresholding."""
        s = np.asarray(scores, dtype=float)
        if window <= 1 or len(s) == 0:
            return s
        return np.convolve(s, np.ones(window) / window, mode="same")

    @staticmethod
    def _longest_run_above(scores, thr: float, gap: int, min_len: int):
        """
        Longest contiguous run of seconds with score >= ``thr``, bridging dips of
        up to ``gap`` seconds, that is at least ``min_len`` long. This anchors us
        firmly inside the sermon's high-confidence plateau. Returns
        (start_idx, end_idx) or None.
        """
        s = np.asarray(scores, dtype=float)
        idx = np.where(s >= thr)[0]
        if len(idx) == 0:
            return None

        best = None
        start = prev = int(idx[0])
        for x in idx[1:]:
            x = int(x)
            if x - prev <= gap + 1:
                prev = x
            else:
                if (prev - start + 1) >= min_len and (best is None or prev - start > best[1] - best[0]):
                    best = (start, prev)
                start = prev = x
        if (prev - start + 1) >= min_len and (best is None or prev - start > best[1] - best[0]):
            best = (start, prev)

        return best

    @staticmethod
    def _extend(scores, anchor: int, thr: float, gap: int, step: int) -> int:
        """
        Walk outward from ``anchor`` in direction ``step`` (-1 back / +1 forward),
        tracking the furthest second that stays at/above ``thr`` while only dips of
        up to ``gap`` consecutive sub-threshold seconds are bridged. Returns the
        furthest reachable index.
        """
        s = np.asarray(scores, dtype=float)
        n = len(s)
        furthest = anchor
        run = 0
        k = anchor + step
        while 0 <= k < n:
            if s[k] >= thr:
                furthest = k
                run = 0
            else:
                run += 1
                if run > gap:
                    break
            k += step
        return furthest

    @staticmethod
    def _aligned_column(data_lines, speaker, n, window):
        """
        Smoothed per-second scores for ``speaker`` aligned to length ``n`` (the
        Preacher series). Returns zeros if the column is absent/misaligned, so a
        missing reference simply disables any check that uses it.
        """
        _, col = AudioUtils.parse_speaker_scores(data_lines, speaker)
        if len(col) != n:
            return np.zeros(n, dtype=float)
        return AudioUtils._smooth(col, window)

    @staticmethod
    def _advance_past_non_dominant(scores, dominant, start_idx, core_start, sustain):
        """
        Move the start forward off any leading region where the preacher is not
        the dominant speaker (announcements/worship/testimony that happen to
        score high against the preacher reference). Returns the first index in
        [start_idx, core_start] where dominance holds continuously for
        ``sustain`` seconds; falls back to ``core_start`` (always inside the
        sermon plateau) if none qualifies.
        """
        i = start_idx
        while i < core_start:
            if dominant[i] and bool(np.all(dominant[i:i + sustain])):
                return i
            i += 1
        return core_start

    def get_trim_range(self, data_lines: list[str]) -> tuple[int, int]:
        """
        Detect the (start, end) timestamps that bound the sermon.

        The sermon is the long high-confidence plateau in the self-enrolled
        Preacher similarity. We anchor on that plateau (the longest run above the
        ``high_percentile``) and then extend the two ends *asymmetrically*, which
        matches how a service is shaped:

          * **start** -- extend back only while scores stay above a *shoulder*
            threshold, bridging short dips. This stops at the gap that separates
            the sermon from the preceding announcements/worship, so the start
            does not bleed into pre-sermon speech. A *speaker-dominance* gate then
            trims any leading region where the Announcement/Worship reference
            out-scores the Preacher (pre-sermon content that scores high against
            the preacher reference).
          * **end** -- extend forward while scores stay above the *low* threshold,
            bridging *long* dips. This keeps the closing / altar-call that the
            same preacher delivers after a quiet stretch.

        All thresholds are per-service percentiles (relative, self-calibrating).
        Falls back to (0, video_length) when no clear plateau is found.
        """
        secs, raw = self.parse_speaker_scores(data_lines, "Preacher")
        vid_len = (secs[-1] if secs else len(data_lines) - 1)

        if not raw:
            logger.error("No Preacher scores found; returning full video range.")
            return 0, vid_len

        window = cfg.get("trim_logic.smooth_window_seconds")
        scores = self._smooth(raw, window)

        low_thr = float(np.percentile(scores, cfg.get("trim_logic.low_percentile")))
        high_thr = float(np.percentile(scores, cfg.get("trim_logic.high_percentile")))
        shoulder_thr = low_thr + cfg.get("trim_logic.shoulder_ratio") * (high_thr - low_thr)
        min_core = cfg.get("trim_logic.min_continuous_seconds")

        core = self._longest_run_above(
            scores, high_thr, cfg.get("trim_logic.core_gap_seconds"), min_core
        )
        if core is None:
            logger.warning(
                "No sermon plateau found (high=%.3f); using full video range.", high_thr,
            )
            return 0, vid_len

        c0, c1 = core
        start_idx = self._extend(scores, c0, shoulder_thr, cfg.get("trim_logic.start_gap_seconds"), -1)
        end_idx = self._extend(scores, c1, low_thr, cfg.get("trim_logic.end_gap_seconds"), +1)

        # Speaker-dominance gate: skip any leading non-preacher-dominant region.
        ann = self._aligned_column(data_lines, "Announcement", len(scores), window)
        wor = self._aligned_column(data_lines, "Worship", len(scores), window)
        dominant = scores >= (np.maximum(ann, wor) + cfg.get("trim_logic.dominance_margin"))
        start_idx = self._advance_past_non_dominant(
            scores, dominant, start_idx, c0, cfg.get("trim_logic.dominance_sustain_seconds")
        )

        start_sec = max(0, secs[start_idx] - cfg.get("trim_logic.start_padding"))
        end_sec = min(secs[end_idx] + cfg.get("trim_logic.padding"), vid_len)
        logger.info(
            f"Sermon span: {start_sec:04d}s -> {end_sec:04d}s "
            f"(core {secs[c0]}s-{secs[c1]}s | low={low_thr:.3f} "
            f"shoulder={shoulder_thr:.3f} high={high_thr:.3f})"
        )
        return start_sec, end_sec

    # --------------------------------------------------
    # Start / end trim (compat wrappers over get_trim_range)
    # --------------------------------------------------
    def get_start_trim(self, data_lines: list[str]) -> int:
        """Start timestamp of the sermon (see :meth:`get_trim_range`)."""
        start, _ = self.get_trim_range(data_lines)
        logger.info(f"Start timestamp: {start:04d}s")
        return start

    def get_end_trim(self, data_lines: list[str]) -> int:
        """End timestamp of the sermon (see :meth:`get_trim_range`)."""
        _, end = self.get_trim_range(data_lines)
        logger.info(f"End timestamp: {end:04d}s")
        return end


    def speaker_segments_generator(
        self,
        audio_file: Path
    ):
        """
        Generate reference segments for the *context* roles (Pre Svc, Worship,
        Announcement) used as informational columns in the similarity output.

        The Preacher reference is no longer guessed from the clock here -- it is
        self-enrolled from the audio in :meth:`run_diarization`
        (see :meth:`self_enroll_preacher`), which removes the biggest source of
        inconsistency: a clock-guessed reference clip landing on the wrong part
        of the service.

        Returns: dict mapping role name -> list of (start, end) second ranges.
        """

        segment_len = cfg.get("segment_duration")
        SECONDS = 60

        levels = self.loudness_per_second(audio_file)
        pre_svc_start = cfg.get("speakers.Pre_svc.start") * SECONDS  # 5 minutes (estimated)
        announcement_start = False

        loud_list = []
        for sec in range(0, len(levels), 1):
            if len(loud_list) >= 10:
                worship_start = loud_list[0]  # set worship start to 30 secs before loud part
                logger.info(f"Worship start adjusted to loud part at: {worship_start}s")
                announcement_start = loud_list[0] + (30 * SECONDS)  # 30 minutes after worship start
                logger.info(f"Announcement start set to: {announcement_start}s")
                break
            level = levels.get(sec)
            if level is not None and level >= cfg.get("audio.loud_level"):
                loud_list.append(sec)
            else:
                loud_list = []

        if len(loud_list) < 10:
            worship_start = cfg.get("speakers.Worship.start") * SECONDS  # 30 minutes (estimated)
            logger.info(f"Worship start set to default: {worship_start}s")
        if not announcement_start:
            logger.warning("Announcement start not found, setting to default 50 minutes")
            announcement_start = cfg.get("speakers.Announcement.start") * SECONDS  # 50 minutes (estimated)

        speaker_segments = {
            "Pre Svc": [(pre_svc_start, segment_len)],
            "Worship": [(worship_start, worship_start + segment_len)],
            "Announcement": [(announcement_start, announcement_start + segment_len)],
        }

        logger.debug(f"Speaker segments is: {speaker_segments}")
        return speaker_segments

    # --------------------------------------------------
    # Per-second embeddings + self-enrollment
    # --------------------------------------------------
    def embeds_per_second(self, cont_embeds: np.ndarray, wav_slices) -> dict[int, np.ndarray]:
        """
        Collapse the continuous (per-window) embeddings into one L2-normalised
        embedding per second, mirroring :meth:`similarity_per_second`'s bucketing.
        """
        times = np.array([
            (s.start + s.stop) / 2 / sampling_rate
            for s in wav_slices
        ])
        max_sec = int(np.ceil(times[-1]))
        per_second: dict[int, np.ndarray] = {}

        for sec in range(max_sec):
            idx = np.where((times >= sec) & (times < sec + 1))[0]
            if len(idx) == 0:
                continue
            mean = np.mean(cont_embeds[idx], axis=0)
            norm = np.linalg.norm(mean)
            per_second[sec] = mean / norm if norm > 0 else mean

        return per_second

    def find_longest_speech_run(self, levels: dict[int, float]) -> tuple[int, int] | None:
        """
        Locate the longest contiguous *speech* region from per-second loudness.

        Loudness bands (from config): below ``audio.quiet_level`` is silence,
        at/above ``audio.loud_level`` is worship/music, and the band in between
        is speech. The sermon is the longest such speech run. Brief excursions
        out of the band (pauses, a loud "Amen") of up to
        ``diarization.self_enroll.gap_bridge_seconds`` are bridged. This seeds
        self-enrollment; final boundaries come from embedding similarity, not
        from these loudness bands.
        """
        if not levels:
            return None

        quiet = cfg.get("audio.quiet_level")
        loud = cfg.get("audio.loud_level")
        gap = cfg.get("diarization.self_enroll.gap_bridge_seconds")

        speech_secs = sorted(
            sec for sec, lvl in levels.items() if quiet <= lvl < loud
        )
        if not speech_secs:
            return None

        best = None
        start = prev = speech_secs[0]
        for sec in speech_secs[1:]:
            if sec - prev <= gap + 1:
                prev = sec
            else:
                if best is None or (prev - start) > (best[1] - best[0]):
                    best = (start, prev)
                start = prev = sec
        if best is None or (prev - start) > (best[1] - best[0]):
            best = (start, prev)

        return best

    def self_enroll_preacher(
        self,
        per_sec_embeds: dict[int, np.ndarray],
        run: tuple[int, int] | None,
    ) -> np.ndarray | None:
        """
        Build the Preacher reference embedding from the audio itself.

        Takes the embeddings inside the longest speech ``run``, forms a centroid,
        then does one refinement pass keeping only the
        ``diarization.self_enroll.refine_keep_ratio`` most central embeddings
        before re-computing the centroid. This denoises the reference against any
        non-preacher seconds that slipped into the run. Returns a unit vector, or
        None if there is nothing to enroll from.
        """
        if run is None:
            return None

        start, end = run
        embeds = [per_sec_embeds[s] for s in range(start, end + 1) if s in per_sec_embeds]
        if not embeds:
            return None

        embeds = np.stack(embeds)

        def _centroid(vectors: np.ndarray) -> np.ndarray:
            mean = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mean)
            return mean / norm if norm > 0 else mean

        centroid = _centroid(embeds)

        # Refinement pass: keep the most central embeddings, drop outliers.
        keep_ratio = cfg.get("diarization.self_enroll.refine_keep_ratio")
        keep_n = max(1, int(len(embeds) * keep_ratio))
        sims = embeds @ centroid
        top_idx = np.argsort(sims)[-keep_n:]
        centroid = _centroid(embeds[top_idx])

        logger.info(
            f"Self-enrolled Preacher from {len(embeds)}s speech run "
            f"({start}s-{end}s), kept top {keep_n}s."
        )
        return centroid

    def similarity_per_second(self, similarity_dict, wav_slices):
        """
        similarity_dict: {speaker_name: np.array[N]}
        wav_slices: list of slice objects (same length N)
        """

        # Center time of each partial window
        times = np.array([
            (s.start + s.stop) / 2 / sampling_rate
            for s in wav_slices
        ])

        max_sec = int(np.ceil(times[-1]))
        per_second = {}

        for sec in range(max_sec):
            idx = np.where((times >= sec) & (times < sec + 1))[0]
            if len(idx) == 0:
                continue

            per_second[sec] = {
                speaker: float(np.mean(scores[idx]))
                for speaker, scores in similarity_dict.items()
            }

        return per_second
    
    def run_diarization(
        self,
        audio_path: Path,
        speaker_segments: dict,
    ):
        """
        audio_path: Takes in audio file
        speaker_segments: Takes in audio segments of speaker
        """
        

        # Load audio
        wav = self.load_audio_no_trim(audio_path)
        
        # Set rate
        rate = cfg.get("diarization.rate")

        # Initialize encoder
        encoder = VoiceEncoder("cpu")

        # Continuous embeddings
        _, cont_embeds, wav_slices = encoder.embed_utterance(
            wav,
            return_partials=True,
            rate=rate
        )

        # Enroll context speakers (informational columns). Guard against clips
        # that fall outside the audio (empty) so a bad clock guess can't crash.
        speaker_embeds = {}
        for name, segments in speaker_segments.items():
            clips = [
                wav[int(start * sampling_rate):int(end * sampling_rate)]
                for start, end in segments
            ]
            clips = [c for c in clips if len(c) > 0]
            if not clips:
                logger.warning(f"No audio for reference speaker '{name}'; skipping column.")
                continue
            speaker_embeds[name] = encoder.embed_speaker(clips)

        # Self-enroll the Preacher from the audio itself (replaces clock guess).
        per_sec_embeds = self.embeds_per_second(cont_embeds, wav_slices)
        levels = self.loudness_per_second(audio_path)
        speech_run = self.find_longest_speech_run(levels)
        preacher_embed = self.self_enroll_preacher(per_sec_embeds, speech_run)
        if preacher_embed is not None:
            speaker_embeds["Preacher"] = preacher_embed
        else:
            logger.error("Self-enrollment failed; no Preacher column produced.")

        # Similarity scores
        similarity_dict = {
            name: cont_embeds @ embed
            for name, embed in speaker_embeds.items()
        }

        # Per-second similarity
        return self.similarity_per_second(similarity_dict, wav_slices)