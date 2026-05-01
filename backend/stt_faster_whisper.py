import os

# Must be set BEFORE importing faster_whisper / ctranslate2
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class STTConfig:
    model_size: str = "base"  # "small" is also a good fast choice
    device: str = "cpu"  # set to "cuda" if you have NVIDIA GPU + CUDA
    compute_type: str = "int8"  # fastest on CPU; try "float16" on GPU
    sample_rate: int = 16_000


class FasterWhisperSTT:
    """
    Fast speech-to-text using faster-whisper.
    - Loads the model ONCE at startup (low latency per turn).
    - Records mic audio into memory (no WAV files).
    """

    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def record_audio(self, seconds: float) -> np.ndarray:
        print(f"Recording for {seconds:.1f}s... Speak now.")
        time.sleep(0.15)
        audio = sd.rec(
            int(seconds * self.cfg.sample_rate),
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio[:, 0].astype(np.float32)

    def record_until_silence(
        self,
        *,
        silence_seconds: float = 2.0,
        max_seconds: float = 20.0,
        min_seconds: float = 0.4,
        start_threshold: float = 0.02,
        stop_threshold: float = 0.012,
        chunk_ms: int = 30,
    ) -> np.ndarray:
        """
        Dynamic recording with simple silence detection (no extra deps).

        Behavior:
        - Wait for voice to start (level > start_threshold)
        - Keep recording while speaking
        - Stop after `silence_seconds` of silence (level < stop_threshold)
        - Hard stop at `max_seconds`
        """
        sr = self.cfg.sample_rate
        chunk = max(1, int(sr * (chunk_ms / 1000.0)))
        max_samples = int(max_seconds * sr)
        min_samples = int(min_seconds * sr)
        silence_samples_needed = int(silence_seconds * sr)

        print("Listening... Speak naturally. (Stops after ~2s silence)")
        time.sleep(0.05)

        frames: list[np.ndarray] = []
        started = False
        silence_run = 0
        total = 0

        def level(x: np.ndarray) -> float:
            # RMS (fast, simple)
            x = x.astype(np.float32, copy=False)
            return float(np.sqrt(np.mean(x * x) + 1e-12))

        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=chunk,
        ) as stream:
            while total < max_samples:
                data, _overflowed = stream.read(chunk)
                x = data[:, 0].astype(np.float32, copy=False)

                lv = level(x)

                if not started:
                    # Wait until we detect speech
                    if lv >= start_threshold:
                        started = True
                        frames.append(x.copy())
                        total += x.size
                    continue

                # After started, always buffer
                frames.append(x.copy())
                total += x.size

                # Silence tracking (only after minimum speech length)
                if total >= min_samples:
                    if lv < stop_threshold:
                        silence_run += x.size
                    else:
                        silence_run = 0

                    if silence_run >= silence_samples_needed:
                        break

        if not frames:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(frames, axis=0)
        # Trim trailing silence (optional small trim)
        return audio.astype(np.float32, copy=False)

    def transcribe_with_language(self, audio: np.ndarray) -> Tuple[str, str, float]:
        # Speed-first settings:
        # - beam_size=1 (greedy) is fastest
        # - vad_filter removes silence quickly
        segments, info = self.model.transcribe(
            audio,
            language=None,  # auto-detect language
            beam_size=1,
            vad_filter=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        lang = str(getattr(info, "language", "") or "unknown").lower()
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        return text, lang, prob

    def transcribe(self, audio: np.ndarray) -> str:
        text, _lang, _prob = self.transcribe_with_language(audio)
        return text

