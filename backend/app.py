import asyncio
import json
import os
import pathlib
import subprocess
import sys
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv


ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "assistant_state.json"
FRONTEND_DIR = ROOT / "frontend"


app = FastAPI(title="Voice Assistant UI")


class _Hub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        msg = json.dumps(payload, ensure_ascii=False)
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                await self.remove(ws)


hub = _Hub()
_assistant_proc: subprocess.Popen[str] | None = None
_ask_lock = asyncio.Lock()
_stt = None
_stt_lock = asyncio.Lock()
_listen_session = None
_speaker_lock = asyncio.Lock()
_speaker = None


class _ListenSession:
    def __init__(self, *, sample_rate: int, chunk_ms: int, max_seconds: float) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk_ms = int(chunk_ms)
        self.max_seconds = float(max_seconds)
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.frames: list[Any] = []
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="ListenSession", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout_s: float | None = None) -> bool:
        self.thread.join(timeout=timeout_s)
        return not self.thread.is_alive()

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd

            sr = self.sample_rate
            chunk = max(1, int(sr * (self.chunk_ms / 1000.0)))
            max_samples = int(self.max_seconds * sr)
            total = 0

            with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=chunk) as stream:
                while not self.stop_event.is_set() and total < max_samples:
                    data, _overflowed = stream.read(chunk)
                    x = data[:, 0].astype(np.float32, copy=False)
                    self.frames.append(x.copy())
                    total += x.size
        except Exception as e:
            self.error = str(e)
        finally:
            self.done_event.set()


class AskRequest(BaseModel):
    text: str


class ListenRequest(BaseModel):
    max_seconds: float | None = 20.0
    silence_seconds: float | None = 2.0


class ListenStartRequest(BaseModel):
    max_seconds: float | None = 20.0
    chunk_ms: int | None = 30


class LanguageRequest(BaseModel):
    selected_language: str


def _read_state() -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


async def _watch_state_file() -> None:
    last_turn: int | None = None
    last_mtime_ns: int | None = None

    while True:
        await asyncio.sleep(0.25)
        try:
            st = STATE_PATH.stat()
        except FileNotFoundError:
            continue

        if last_mtime_ns is not None and st.st_mtime_ns == last_mtime_ns:
            continue
        last_mtime_ns = st.st_mtime_ns

        state = _read_state()
        if not state:
            continue

        turn = state.get("turn")
        if isinstance(turn, int) and turn == last_turn:
            continue
        if isinstance(turn, int):
            last_turn = turn

        await hub.broadcast({"type": "assistant_state", "state": state})


def _start_voice_assistant_subprocess() -> subprocess.Popen[str]:
    """
    Start the existing voice assistant loop as a subprocess.
    It keeps using the same mic/STT/TTS code; the UI only reads assistant_state.json.
    """
    voice_path = ROOT / "backend" / "voice_assistant.py"
    if not voice_path.exists():
        # Fallback (if user kept it in the root)
        voice_path = ROOT / "voice_assistant.py"

    return subprocess.Popen(
        [sys.executable, str(voice_path)],
        cwd=str(ROOT),
        stdout=None,
        stderr=None,
        text=True,
    )


def _process_user_text(user_text: str, *, detected_language: str | None = None) -> dict[str, Any]:
    """
    Reuse the existing assistant logic (LLM prompts + JSON parsing + response shaping)
    but run it from the UI server when the browser sends text.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        raise ValueError("Empty text")

    # Import from the existing code to avoid rewriting logic.
    from dotenv import load_dotenv
    from openai import OpenAI

    from backend.voice_assistant import (
        detect_text_language_simple,
        get_selected_language,
        process_user_text_strict_language,
    )

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Create a .env file with OPENAI_API_KEY=your_key")
    llm_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
    fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", "openai/gpt-4o-mini").strip()

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    selected_language = get_selected_language()
    # Manual language selection is authoritative. Never auto-switch.
    effective_language = selected_language
    # Detection is used only to choose behavior branch (match vs mismatch),
    # never to overwrite selected language.
    detected = (detected_language or detect_text_language_simple(user_text) or "").strip().lower()

    memory_context = ""
    facts = ""
    semantic = ""
    try:
        from backend.user_facts import facts_block_for_prompt

        facts = facts_block_for_prompt(selected_language=effective_language)
    except Exception as fact_exc:
        print(f"[user_facts] load skipped: {fact_exc}")
    try:
        from backend.conversation_memory import get_conversation_memory

        mem = get_conversation_memory()
        hits = mem.search(user_text, selected_language=effective_language, top_k=3)
        semantic = mem.format_for_prompt(hits) if hits else ""
    except Exception as mem_exc:
        print(f"[memory] retrieve skipped: {mem_exc}")
    try:
        from backend.user_facts import merge_memory_sections

        memory_context = merge_memory_sections(facts_block=facts, semantic_block=semantic)
    except Exception:
        memory_context = "\n\n".join(x for x in (facts, semantic) if x)

    def _is_provider_failure(err: Exception) -> bool:
        msg = str(err).lower()
        err_name = err.__class__.__name__.lower()
        return any(
            k in msg
            for k in [
                "503",
                "429",
                "rate limit",
                "provider returned error",
                "no healthy upstream",
                "upstream",
                "service unavailable",
                "timed out",
                "timeout",
                "connecttimeout",
                "connection error",
            ]
        ) or any(k in err_name for k in ["timeout", "connection", "apierror", "ratelimit"])

    def _simple_correct(text: str, selected: str) -> str:
        t = (text or "").strip()
        if not t:
            return t
        if selected in {"en", "fr", "es"}:
            t = t[0].upper() + t[1:] if t else t
            if t[-1] not in ".!?":
                t += "."
        return t

    def _local_teacher_fallback(*, selected: str, text: str) -> dict[str, str]:
        corrected_local = _simple_correct(text, selected)
        if selected == "fr":
            return {
                "corrected": corrected_local,
                "explanation": "Merci, continuons en français.",
                "answer": "J'ai bien compris votre phrase. Continuons en français: pouvez-vous me donner plus de details ?",
            }
        if selected == "ar":
            return {
                "corrected": corrected_local,
                "explanation": "رائع، لنكمل بالعربية.",
                "answer": "فهمت جملتك جيدا. لنواصل بالعربية: هل يمكنك إعطائي مزيدا من التفاصيل؟",
            }
        if selected == "es":
            return {
                "corrected": corrected_local,
                "explanation": "Gracias, continuemos en español.",
                "answer": "He entendido bien tu frase. Sigamos en español: ¿puedes darme un poco más de detalle?",
            }
        return {
            "corrected": corrected_local,
            "explanation": "Great, let's continue in English.",
            "answer": "I understood your sentence. Let's continue in English: could you share a little more detail?",
        }

    try:
        llm = process_user_text_strict_language(
            client,
            user_text=user_text,
            model_name=llm_model,
            selected_language=effective_language,
            detected_language=detected or None,
            memory_context=memory_context or None,
        )
    except Exception as e:
        # Retry once on provider-style failures with fallback model.
        if _is_provider_failure(e):
            try:
                llm = process_user_text_strict_language(
                    client,
                    user_text=user_text,
                    model_name=fallback_model or llm_model,
                    selected_language=effective_language,
                    detected_language=detected or None,
                    memory_context=memory_context or None,
                )
            except Exception:
                # Final graceful local fallback (never expose provider errors to user).
                llm = _local_teacher_fallback(selected=effective_language, text=user_text)
        else:
            raise

    answer = llm["answer"] or "Sorry, I could not generate an answer."
    corrected = llm["corrected"] or user_text
    explanation = llm["explanation"] or "Looks correct."

    state = {
        "turn": int((_read_state() or {}).get("turn") or 0) + 1,
        "user_text": user_text,
        "detected_language": detected or effective_language,
        "selected_language": effective_language,
        "answer": answer,
        "corrected": corrected,
        "explanation": explanation,
        "source": "browser",
    }
    try:
        from backend.conversation_memory import get_conversation_memory

        get_conversation_memory().add_turn(
            user_text=user_text,
            answer=answer,
            selected_language=effective_language,
            corrected=corrected,
            explanation=explanation,
        )
    except Exception as mem_exc:
        print(f"[memory] store skipped: {mem_exc}")
    try:
        from backend.user_facts import ingest_from_user_text

        ingest_from_user_text(user_text)
    except Exception as fact_exc:
        print(f"[user_facts] ingest skipped: {fact_exc}")
    return state


def _safe_error_state(*, user_text: str = "", detected_language: str | None = None) -> dict[str, Any]:
    from backend.voice_assistant import get_selected_language

    selected = get_selected_language()
    safe_msg = {
        "en": "Sorry, I had trouble processing your request. Please try again or switch the language.",
        "fr": "Desole, j'ai eu un probleme pour traiter votre demande. Veuillez reessayer ou changer la langue.",
        "ar": "عذرا، واجهت مشكلة أثناء معالجة طلبك. يرجى المحاولة مرة أخرى أو تغيير اللغة.",
        "es": "Lo siento, tuve un problema al procesar tu solicitud. Por favor, inténtalo de nuevo o cambia el idioma.",
    }.get(selected, "Sorry, I had trouble processing your request. Please try again or switch the language.")
    return {
        "turn": int((_read_state() or {}).get("turn") or 0) + 1,
        "user_text": user_text,
        "detected_language": detected_language or selected,
        "selected_language": selected,
        "answer": safe_msg,
        "corrected": "",
        "explanation": "",
        "source": "safe-fallback",
    }


def _ensure_speaker():
    global _speaker
    if _speaker is not None:
        return _speaker
    from backend.audio_speaker import BackgroundSpeaker

    _speaker = BackgroundSpeaker()
    return _speaker


def _resolve_voice_model_for_language(language_code: str) -> tuple[str, str]:
    """
    Strict language -> Piper ONNX model mapping.
    Falls back to English model only if a language model is missing.
    """
    lang = (language_code or "en").strip().lower().split("-")[0]
    voice_dir = pathlib.Path(os.getenv("PIPER_VOICES_DIR", r"C:\piper\voices"))

    explicit_map = {
        "ar": "ar_JO-kareem-medium.onnx",
        "en": "en_GB-jenny_dioco-medium.onnx",
        "fr": "fr_FR-siwis-medium.onnx",
        "es": "es_ES-mls_9972-low.onnx",
        "fa": "fa_IR-reza_ibrahim-medium.onnx",
    }

    def _first_existing(candidates: list[pathlib.Path]) -> pathlib.Path | None:
        for p in candidates:
            if p.exists():
                return p
        return None

    # Strict mapping for known installed models.
    mapped_name = explicit_map.get(lang)
    if mapped_name:
        p = voice_dir / mapped_name
        if p.exists():
            return str(p), lang

    # Generic pattern for language families.
    family = sorted(voice_dir.glob(f"{lang}_*medium.onnx")) + sorted(voice_dir.glob(f"{lang}_*.onnx"))
    if family:
        return str(family[0]), lang

    # Final fallback: English voice only when selected language model is unavailable.
    en_candidates = [
        voice_dir / explicit_map["en"],
        *sorted(voice_dir.glob("en_*medium.onnx")),
        *sorted(voice_dir.glob("en_*.onnx")),
    ]
    en_model = _first_existing(en_candidates)
    if en_model is not None:
        return str(en_model), "en"

    # Last-resort fallback to configured path (keeps system alive).
    return os.getenv("PIPER_MODEL", r"C:\piper\voices\en_GB-jenny_dioco-medium.onnx"), "en"


def _speak_text_blocking(text: str, *, selected_language: str) -> None:
    t = (text or "").strip()
    if not t:
        return
    from backend.tts_piper_inmemory import synthesize_piper_inmemory

    piper_exe = os.getenv("PIPER_EXE", r"C:\piper\piper.exe")
    piper_model, resolved_lang = _resolve_voice_model_for_language(selected_language)
    if resolved_lang != (selected_language or "en").split("-")[0]:
        print(f"[TTS] Voice fallback for '{selected_language}' -> '{resolved_lang}' model")
    audio = synthesize_piper_inmemory(t, piper_exe=piper_exe, model_path=piper_model)
    _ensure_speaker().speak_and_wait(audio.samples, audio.sample_rate)


async def _publish_and_speak_state(state: dict[str, Any]) -> None:
    # Always persist state first so UI can render even if audio fails.
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    await hub.broadcast({"type": "assistant_state", "state": state})

    answer = str(state.get("answer") or "").strip()
    if not answer:
        return
    selected_language = str(state.get("selected_language") or "en")

    await hub.broadcast({"type": "speech_state", "speaking": True})
    try:
        async with _speaker_lock:
            await asyncio.to_thread(_speak_text_blocking, answer, selected_language=selected_language)
    except Exception as e:
        # Never crash request flow on audio errors.
        print(f"[TTS ERROR] {e}")
    finally:
        await hub.broadcast({"type": "speech_state", "speaking": False})


def _ensure_stt():
    global _stt
    if _stt is not None:
        return _stt
    from backend.stt_faster_whisper import FasterWhisperSTT, STTConfig

    _stt = FasterWhisperSTT(
        STTConfig(
            model_size=os.getenv("STT_MODEL", "base"),
            device=os.getenv("STT_DEVICE", "cpu"),
            compute_type=os.getenv("STT_COMPUTE", "int8"),
        )
    )
    return _stt


def _listen_and_transcribe(*, max_seconds: float, silence_seconds: float) -> str:
    stt = _ensure_stt()
    audio = stt.record_until_silence(max_seconds=max_seconds, silence_seconds=silence_seconds)
    text, _lang, _prob = stt.transcribe_with_language(audio)
    return (text or "").strip()


def _transcribe_audio(audio) -> tuple[str, str]:
    stt = _ensure_stt()
    text, lang, _prob = stt.transcribe_with_language(audio)
    return (text or "").strip(), (lang or "").strip().lower()


@app.on_event("startup")
async def _startup() -> None:
    global _assistant_proc
    if not FRONTEND_DIR.exists():
        raise RuntimeError("frontend/ folder missing. Rebuild UI files.")

    load_dotenv()
    # Start background watcher (push updates to the browser for compatibility).
    asyncio.create_task(_watch_state_file())

    # IMPORTANT (UI mode): do NOT start the Python voice assistant loop.
    # The browser is responsible for TTS (single voice) to avoid duplicate speech.
    # If you ever need the old terminal voice assistant, run backend/voice_assistant.py manually.
    if str((os.getenv("START_ASSISTANT") or "0")).strip() not in {"0", "false", "False"}:
        _assistant_proc = _start_voice_assistant_subprocess()


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _assistant_proc, _speaker
    if _assistant_proc and _assistant_proc.poll() is None:
        try:
            _assistant_proc.terminate()
        except Exception:
            pass
    _assistant_proc = None
    if _speaker is not None:
        try:
            _speaker.close()
        except Exception:
            pass
        _speaker = None


@app.get("/api/latest")
def api_latest() -> dict[str, Any]:
    return _read_state() or {}


@app.get("/api/language")
def api_get_language() -> dict[str, Any]:
    from backend.voice_assistant import get_selected_language

    return {"selected_language": get_selected_language()}


@app.post("/api/language")
def api_set_language(req: LanguageRequest) -> dict[str, Any]:
    code = (req.selected_language or "").strip().lower().split("-")[0]
    allowed = {"en", "fr", "ar", "es"}
    if code not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {code}")
    os.environ["SELECTED_LANGUAGE"] = code
    return {"selected_language": code}


@app.post("/api/ask")
async def api_ask(req: AskRequest) -> dict[str, Any]:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")

    # Serialize requests so state/turn increments stay consistent.
    async with _ask_lock:
        try:
            state = await asyncio.to_thread(_process_user_text, text)
        except Exception:
            state = _safe_error_state(user_text=text)

        await _publish_and_speak_state(state)
        return state


@app.post("/api/listen")
async def api_listen(req: ListenRequest) -> dict[str, Any]:
    """
    Button-driven "Start Listening" that reuses the existing Python mic + STT.
    Browser SpeechRecognition is not reliable on all Windows setups (often reports 'network').
    """
    max_seconds = float(req.max_seconds or 20.0)
    silence_seconds = float(req.silence_seconds or 2.0)

    async with _ask_lock:
        try:
            # STT is blocking; run it off the event loop.
            user_text = await asyncio.to_thread(
                _listen_and_transcribe, max_seconds=max_seconds, silence_seconds=silence_seconds
            )
        except Exception:
            state = _safe_error_state()
            await _publish_and_speak_state(state)
            return state

        if not user_text:
            state = _safe_error_state()
            await _publish_and_speak_state(state)
            return state

        try:
            state = await asyncio.to_thread(_process_user_text, user_text)
        except Exception:
            state = _safe_error_state(user_text=user_text)

        await _publish_and_speak_state(state)
        return state


@app.post("/api/listen/start")
async def api_listen_start(req: ListenStartRequest) -> dict[str, Any]:
    """
    Start recording audio on the backend (toggle button UX).
    """
    global _listen_session
    max_seconds = float(req.max_seconds or 20.0)
    chunk_ms = int(req.chunk_ms or 30)

    async with _ask_lock:
        if _listen_session is not None:
            return {"ok": True}
        try:
            stt = _ensure_stt()
        except Exception:
            # Keep API stable; frontend should remain responsive.
            return {"ok": False}

        _listen_session = _ListenSession(
            sample_rate=int(getattr(stt.cfg, "sample_rate", 16000)),
            chunk_ms=chunk_ms,
            max_seconds=max_seconds,
        )
        _listen_session.start()
        return {"ok": True}


@app.post("/api/listen/stop")
async def api_listen_stop() -> dict[str, Any]:
    """
    Stop recording, transcribe, run assistant, write state (UI updates via watcher).
    """
    global _listen_session

    async with _ask_lock:
        sess = _listen_session
        if sess is None:
            state = _safe_error_state()
            await _publish_and_speak_state(state)
            return state

        sess.stop()
        await asyncio.to_thread(sess.join, 5.0)
        _listen_session = None

        if sess.error:
            state = _safe_error_state()
            await _publish_and_speak_state(state)
            return state

        try:
            import numpy as np

            if not sess.frames:
                state = _safe_error_state()
                await _publish_and_speak_state(state)
                return state
            audio = np.concatenate(sess.frames, axis=0).astype(np.float32, copy=False)
        except Exception:
            state = _safe_error_state()
            await _publish_and_speak_state(state)
            return state

        try:
            user_text, detected_language = await asyncio.to_thread(_transcribe_audio, audio)
        except Exception:
            state = _safe_error_state()
            await _publish_and_speak_state(state)
            return state

        if not user_text:
            state = _safe_error_state()
            await _publish_and_speak_state(state)
            return state

        try:
            state = await asyncio.to_thread(_process_user_text, user_text, detected_language=detected_language)
        except Exception:
            state = _safe_error_state(user_text=user_text, detected_language=detected_language)

        await _publish_and_speak_state(state)
        return state


@app.post("/api/welcome")
async def api_welcome() -> dict[str, Any]:
    from backend.voice_assistant import get_selected_language

    lang = get_selected_language()
    welcome_by_lang = {
        "en": "Hello! I'm your assistant. Click Start Listening and speak when you are ready.",
        "fr": "Bonjour ! Je suis votre assistant. Cliquez sur Start Listening puis parlez quand vous etes pret.",
        "ar": "مرحبا! أنا مساعدك. اضغط على Start Listening ثم تحدث عندما تكون جاهزا.",
        "es": "¡Hola! Soy tu asistente. Haz clic en Start Listening y habla cuando estés listo.",
    }
    answer = welcome_by_lang.get(lang, welcome_by_lang["en"])
    state = {
        "turn": int((_read_state() or {}).get("turn") or 0) + 1,
        "user_text": "",
        "detected_language": lang,
        "selected_language": lang,
        "answer": answer,
        "corrected": "",
        "explanation": "",
        "source": "system",
    }
    await _publish_and_speak_state(state)
    return state


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    await hub.add(ws)
    try:
        # Send current state immediately (if available).
        state = _read_state()
        if state:
            await ws.send_text(json.dumps({"type": "assistant_state", "state": state}, ensure_ascii=False))

        while True:
            # Keep socket open; we don't need inbound messages right now.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.remove(ws)


# Mount frontend last so it doesn't shadow /ws and /api routes.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

