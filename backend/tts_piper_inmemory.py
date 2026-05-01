import json
import pathlib
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PiperAudio:
    samples: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int


def _read_piper_sample_rate(config_path: pathlib.Path) -> int:
    """
    Piper voices ship with: <voice>.onnx and <voice>.onnx.json
    The json contains the correct sample_rate for raw audio playback.
    """
    obj = json.loads(config_path.read_text(encoding="utf-8"))

    # Common config shapes across voices/versions:
    # - {"audio": {"sample_rate": 22050, ...}, ...}
    # - {"sample_rate": 22050, ...}
    sr = None
    if isinstance(obj, dict):
        audio = obj.get("audio")
        if isinstance(audio, dict):
            sr = audio.get("sample_rate")
        if sr is None:
            sr = obj.get("sample_rate")

    if not isinstance(sr, int) or sr <= 0:
        raise ValueError(f"Could not find sample_rate in config: {config_path}")
    return sr


def synthesize_piper_inmemory(
    text: str,
    *,
    piper_exe: str = r"C:\piper\piper.exe",
    model_path: str,
    config_path: str | None = None,
) -> PiperAudio:
    """
    Generate speech audio in memory using Piper (no files).

    - Uses: piper.exe --output-raw (16-bit mono PCM)
    - Returns: float32 mono samples + sample rate
    """
    text = (text or "").strip()
    if not text:
        return PiperAudio(samples=np.zeros(0, dtype=np.float32), sample_rate=22050)

    model = pathlib.Path(model_path)
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")

    cfg = pathlib.Path(config_path) if config_path else pathlib.Path(str(model) + ".json")
    if not cfg.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg}\n"
            f"You need BOTH files:\n"
            f"- {model.name}\n"
            f"- {cfg.name}\n"
        )

    sample_rate = _read_piper_sample_rate(cfg)

    cmd = [
        str(pathlib.Path(piper_exe)),
        "--model",
        str(model),
        "--output-raw",
    ]

    # Piper reads text from stdin and writes raw PCM to stdout.
    proc = subprocess.run(
        cmd,
        input=(text + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        cwd=str(model.parent),
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Piper failed (code {proc.returncode}).\n{err}")

    raw = proc.stdout
    if not raw:
        return PiperAudio(samples=np.zeros(0, dtype=np.float32), sample_rate=sample_rate)

    # raw audio is signed 16-bit little-endian mono PCM
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    samples = (pcm16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return PiperAudio(samples=samples, sample_rate=sample_rate)


if __name__ == "__main__":
    # Manual test (no files). Requires:
    # - piper.exe
    # - a voice model (.onnx) and config (.onnx.json)
    from audio_speaker import BackgroundSpeaker

    speaker = BackgroundSpeaker()
    audio = synthesize_piper_inmemory(
        "Hello! This is Piper speaking in real time.",
        model_path=r"C:\piper\voices\en_GB-jenny_dioco-medium.onnx",
    )
    speaker.speak(audio.samples, audio.sample_rate)
    print("Speaking (non-blocking).")
    input("Press Enter to stop test...\n")
    speaker.close()

