import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd


@dataclass
class _AudioJob:
    samples: np.ndarray  # shape (n,) float32 in [-1, 1]
    sample_rate: int


class BackgroundSpeaker:
    """
    Non-blocking audio playback for a conversational assistant.

    - `speak()` returns immediately (plays in background).
    - New `speak()` interrupts the previous audio (more natural).
    - No files and no external media players.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[_AudioJob] = None
        self._job_event = threading.Event()
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._done_event.set()

        self._thread = threading.Thread(target=self._run, name="BackgroundSpeaker", daemon=True)
        self._thread.start()

    def speak(self, samples: np.ndarray, sample_rate: int) -> None:
        """
        Play audio asynchronously.
        `samples` must be 1-D float32 in [-1, 1].
        """
        if samples is None:
            return

        samples = np.asarray(samples, dtype=np.float32).reshape(-1)

        with self._lock:
            self._done_event.clear()
            self._job = _AudioJob(samples=samples, sample_rate=int(sample_rate))
            self._job_event.set()

    def speak_and_wait(self, samples: np.ndarray, sample_rate: int) -> None:
        """Play audio and block until it finishes (for speak-then-listen sync)."""
        self.speak(samples, sample_rate)
        self.wait_until_done()

    def wait_until_done(self, timeout_s: float | None = None) -> bool:
        """Return True if finished, False if timeout."""
        return self._done_event.wait(timeout=timeout_s)

    def stop(self) -> None:
        """Stop current playback (non-blocking)."""
        with self._lock:
            self._done_event.set()
            self._job = _AudioJob(samples=np.zeros(0, dtype=np.float32), sample_rate=16000)
            self._job_event.set()

    def close(self) -> None:
        """Stop the background thread."""
        self._stop_event.set()
        self._job_event.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        current_sr = 48000
        position = 0
        buffer = np.zeros(0, dtype=np.float32)

        def callback(outdata, frames, time, status):  # noqa: ARG001
            nonlocal current_sr, position, buffer

            if self._stop_event.is_set():
                outdata[:] = 0
                raise sd.CallbackStop()

            # Check for a new job (interrupt)
            if self._job_event.is_set():
                self._job_event.clear()
                with self._lock:
                    job = self._job
                    self._job = None
                if job is not None:
                    current_sr = job.sample_rate
                    buffer = job.samples
                    position = 0

            # If nothing to play, output silence
            if buffer.size == 0 or position >= buffer.size:
                outdata[:] = 0
                self._done_event.set()
                return

            end = min(position + frames, buffer.size)
            chunk = buffer[position:end]
            position = end

            # Mono -> (frames, 1)
            out = np.zeros((frames, 1), dtype=np.float32)
            out[: chunk.shape[0], 0] = chunk
            outdata[:] = out
            if position >= buffer.size:
                self._done_event.set()

        # We recreate the stream when sample rate changes.
        while not self._stop_event.is_set():
            # Wait until we have an initial job or stop
            self._job_event.wait(timeout=0.25)
            if self._stop_event.is_set():
                break

            # Grab pending job to learn sample rate before opening stream
            with self._lock:
                job = self._job
                self._job = None
                self._job_event.clear()

            if job is None:
                continue

            current_sr = job.sample_rate
            buffer = job.samples
            position = 0
            self._done_event.clear()

            try:
                with sd.OutputStream(
                    samplerate=current_sr,
                    channels=1,
                    dtype="float32",
                    callback=callback,
                    blocksize=0,  # let sounddevice choose low-latency blocksize
                ):
                    # Keep stream alive until buffer ends or new job arrives.
                    while not self._stop_event.is_set():
                        if self._job_event.is_set():
                            # new job; callback will pick it up
                            pass
                        if buffer.size == 0 or position >= buffer.size:
                            break
                        sd.sleep(20)
            except Exception:
                # If audio device errors, don't crash the whole assistant.
                # The caller can still keep using the program.
                buffer = np.zeros(0, dtype=np.float32)
                position = 0
                self._done_event.set()


if __name__ == "__main__":
    # Quick manual test: plays a 440Hz tone for ~0.7s (non-blocking)
    import time

    sr = 22050
    t = np.linspace(0, 0.7, int(0.7 * sr), endpoint=False)
    tone = 0.15 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    speaker = BackgroundSpeaker()
    speaker.speak(tone, sr)
    print("Playing tone in background...")
    time.sleep(1.0)
    speaker.close()

