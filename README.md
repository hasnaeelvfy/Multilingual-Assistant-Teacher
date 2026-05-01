# English Voice Assistant (Real‑Time)

This project is a **real-time English speaking assistant**:

- You speak (microphone)
- It transcribes (fast speech-to-text)
- It answers + corrects your English (OpenRouter LLM)
- It speaks back (Piper TTS) **inside Python**
- It waits until speaking finishes, then listens again

## Requirements

### 1) Python

Use Python 3.10+ (you are using conda, that's fine).

### 2) Install Python packages

In your environment:

```powershell
cd "C:\Users\hasna elbahraui\Downloads\AI ASSISTANT"
pip install -r requirements.txt
```

### 3) Install Piper (TTS)

You already have:

- `C:\piper\piper.exe`

You also need **a voice model** (two files in the same folder):

- `VOICE.onnx`
- `VOICE.onnx.json`

Example:

- `C:\piper\voices\en_GB-jenny_dioco-medium.onnx`
- `C:\piper\voices\en_GB-jenny_dioco-medium.onnx.json`

## Configure API key (OpenRouter)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-or-v1-your-key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

You can copy from `.env.example`.

Important for GitHub:

- Keep `.env` local only
- `.env` is already ignored in `.gitignore`
- Commit `.env.example` only

Note: the code uses OpenRouter via:

- `base_url="https://openrouter.ai/api/v1"`
- model: from `OPENROUTER_MODEL` in `.env` (default `openai/gpt-4o-mini`)

Example stronger model for better multilingual understanding:

```env
OPENROUTER_MODEL=deepseek/deepseek-chat
```

## Run the voice assistant

```powershell
cd "C:\Users\hasna elbahraui\Downloads\AI ASSISTANT"
python backend\voice_assistant.py --piper-model "C:\piper\voices\en_GB-jenny_dioco-medium.onnx"
```

## Run the Robot UI (recommended)

This replaces the CMD/terminal experience with a browser window that:

- shows your `frontend/models/robot.glb` (Three.js)
- speaks assistant responses using the browser’s built-in SpeechSynthesis (free)
- procedurally animates the robot while speaking (no rig needed)

Start the UI server (it will also start the voice assistant automatically):

```powershell
cd "C:\Users\hasna elbahraui\Downloads\AI ASSISTANT"
python -m backend.run_ui
```

Then open:

- `http://127.0.0.1:8000/`

If port 8000 is already used:

```powershell
$env:PORT=8001
python -m backend.run_ui
```

If you want to run the voice assistant yourself (separately), you can disable auto-start:

```powershell
$env:START_ASSISTANT=0
python -m backend.run_ui
```

### How recording works (natural)

The assistant records until you stop speaking.
If there is ~2 seconds of silence, it stops recording automatically.
Optional tuning:

```powershell
python backend\voice_assistant.py --silence-seconds 2 --seconds 20
```

## Exit

Say: `exit` / `stop` / `goodbye` (even with punctuation like `stop!`)

Or press **Ctrl+C**.

The assistant will say:

> "Okay. I am here anytime. Goodbye."

and then the program stops.

## Language behavior

- The assistant accepts any spoken language.
- It always replies in English.
- If the input is not English:
  - It does not answer the question directly.
  - It asks you to speak in English.
  - It gives an English translation of your sentence.
  - It asks you to repeat in English.

## Notes

- The Python assistant still speaks with Piper (existing system is unchanged).
- The Robot UI speaks in the browser and animates while speaking.

