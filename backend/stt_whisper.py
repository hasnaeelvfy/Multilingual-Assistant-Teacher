"""
argparse lets the program read arguments from the terminal.

Example:

python script.py --seconds 10

So the user can control:

recording duration
model used
output file

import time

Used to control time.

In this program it is used to wait a little before recording starts.

"""
import argparse
import pathlib
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


def record_wav(out_path: pathlib.Path, seconds: float, sample_rate: int = 16_000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording for {seconds:.1f}s... Speak now.")
    time.sleep(0.25)

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    pcm16 = np.clip(audio[:, 0], -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    wavfile.write(str(out_path), sample_rate, pcm16)
    print(f"Saved: {out_path}")


def transcribe_with_whisper(audio_path: pathlib.Path, model_name: str) -> str:
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), fp16=False, language="en")
    return (result.get("text") or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Record microphone audio and transcribe with Whisper.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording duration in seconds.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model: tiny, base, small, medium, large.")
    parser.add_argument("--out", type=str, default="recordings/input.wav", help="Output WAV path.")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    record_wav(out_path=out_path, seconds=args.seconds)
    text = transcribe_with_whisper(audio_path=out_path, model_name=args.model)

    print("\n--- TRANSCRIPT (English) ---")
    print(text if text else "(No speech detected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
