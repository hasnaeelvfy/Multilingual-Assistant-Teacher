import argparse
import json
import os
import pathlib
import re
import signal
import textwrap
from typing import Any

from dotenv import load_dotenv

try:
    # Package-style imports (used by backend.app / UI server).
    from backend.audio_speaker import BackgroundSpeaker
    from backend.stt_faster_whisper import FasterWhisperSTT, STTConfig
    from backend.tts_piper_inmemory import synthesize_piper_inmemory
except ImportError:
    # Script-style fallback (when running this file directly from backend/).
    from audio_speaker import BackgroundSpeaker
    from stt_faster_whisper import FasterWhisperSTT, STTConfig
    from tts_piper_inmemory import synthesize_piper_inmemory


LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic",
    "es": "Spanish",
}


def _lang_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _normalize_lang_code(code: str | None) -> str:
    c = (code or "en").strip().lower()
    if not c:
        return "en"
    return c.split("-")[0]


def get_selected_language() -> str:
    return _normalize_lang_code(os.getenv("SELECTED_LANGUAGE", "en"))


def should_exit(user_text: str) -> bool:
    """
    Detect exit intent robustly (handles punctuation like "stop!" from STT).
    """
    t = (user_text or "").strip().lower()
    if not t:
        return False

    # Keep letters/numbers, treat everything else as a separator.
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", t) if tok]
    return any(tok in {"exit", "quit", "stop", "goodbye", "bye", "close"} for tok in tokens)


def speak_goodbye_and_exit(speaker: BackgroundSpeaker, *, piper_exe: str, piper_model: str) -> int:
    goodbye = "Okay. I am here anytime. Goodbye."
    audio = synthesize_piper_inmemory(goodbye, piper_exe=piper_exe, model_path=piper_model)
    speaker.speak_and_wait(audio.samples, audio.sample_rate)
    speaker.close()
    return 0


def _safe_json_extract(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Model did not return valid JSON.")


def _ask_json_streaming(client, *, system_prompt: str, user_prompt: str, model_name: str, max_tokens: int) -> dict[str, str]:
    """
    Generic JSON response helper (streaming).
    """
    stream = client.responses.stream(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_output_tokens=max_tokens,
    )

    chunks: list[str] = []
    with stream as s:
        for event in s:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)

    text = "".join(chunks).strip()
    obj = _safe_json_extract(text)
    return {
        "answer": str(obj.get("answer") or "").strip(),
        "corrected": str(obj.get("corrected") or "").strip(),
        "explanation": str(obj.get("explanation") or "").strip(),
    }


def detect_text_language_simple(text: str) -> str:
    """
    Lightweight fallback for text language detection.
    Keeps the system robust when STT language isn't available.
    """
    t = (text or "").strip().lower()
    if not t:
        return "en"

    # Arabic script range
    if re.search(r"[\u0600-\u06ff]", t):
        return "ar"

    french_markers = {"je", "tu", "vous", "nous", "bonjour", "merci", "est", "pas", "avec", "pour", "manger"}
    spanish_markers = {
        "hola",
        "gracias",
        "que",
        "como",
        "donde",
        "quiero",
        "pero",
        "buenos",
        "muy",
        "esta",
        "por",
        "favor",
        "cual",
        "bien",
    }
    english_markers = {"i", "you", "hello", "thanks", "what", "how", "the", "is", "are", "want"}
    tokens = [tok for tok in re.split(r"[^a-zA-Z\u0600-\u06ff]+", t) if tok]

    fr_hits = sum(tok in french_markers for tok in tokens)
    es_hits = sum(tok in spanish_markers for tok in tokens)
    en_hits = sum(tok in english_markers for tok in tokens)
    if es_hits > max(fr_hits, en_hits):
        return "es"
    if fr_hits > en_hits:
        return "fr"
    if en_hits > 0:
        return "en"
    return "en"


def enforce_language(response: dict[str, str], selected_language: str) -> dict[str, str]:
    """
    Last safety pass. We rely on prompt constraints as primary enforcement.
    """
    out = {
        "answer": str(response.get("answer") or "").strip(),
        "corrected": str(response.get("corrected") or "").strip(),
        "explanation": str(response.get("explanation") or "").strip(),
    }
    if not out["answer"]:
        fallback = {
            "en": "Please try again in English.",
            "fr": "Merci de reessayer en .",
            "ar": "يرجى المحاولة مرة أخرى بالعربية.",
            "es": "Por favor, inténtalo de nuevo en español.",
        }.get(selected_language, "Please repeat in the selected language.")
        out["answer"] = fallback
    return out


def process_user_text_strict_language(
    client,
    *,
    user_text: str,
    model_name: str,
    selected_language: str,
    detected_language: str | None = None,
    memory_context: str | None = None,
) -> dict[str, str]:
    """
    Strict language policy:
    - If input language matches selected language: correct + explain + answer.
    - If mismatch: do not answer directly, provide translated/corrected sentence + instruction only.
    """
    selected = _normalize_lang_code(selected_language)
    # Manual selection is strict: no automatic language detection/override here.
    detected = _normalize_lang_code(detected_language) if detected_language else selected
    selected_name = _lang_name(selected)

    ctx = (memory_context or "").strip()
    memory_block = ""
    if ctx:
        memory_block = (
            "Context below may include persistent facts (e.g. name) and past turns from any language. "
            "Use it when relevant; focus on the latest user message.\n"
            f"Your JSON fields must still be written entirely in {selected_name} only (no mixing languages).\n"
            f"{ctx}\n"
        )

    def _normalize_for_compare(s: str) -> str:
        return re.sub(r"[^a-z0-9\u0600-\u06ff]+", "", (s or "").strip().lower())

    if detected == selected:
        system_prompt = f"""You are a friendly but strict immersion tutor in {selected_name}.
Reply in VALID JSON only (no extra text):
{{
  "answer": string,
  "corrected": string,
  "explanation": string
}}

You were built by Hasna El Bahraoui, a talented and competent developer who holds a Bachelor's degree in Application Development and Artificial Intelligence. If anyone asks who created you, who your developer is, or anything about the person who built you, you must proudly introduce Hasna as your creator and describe her as intelligent, skilled, and passionate about AI development.

{memory_block}Rules:
- Write ALL fields in {selected_name} only.
- Keep a warm teacher tone (friendly, encouraging, natural).
- Keep medium length (2-4 short sentences).
- corrected = grammatically corrected version of user sentence in {selected_name}.
- explanation = very short, natural note in {selected_name}; no grammar lecture.
- answer = a real conversational response to the user's intent/question.
- answer MUST NOT simply copy or repeat corrected text.
- If corrected and user sentence are the same, answer should still continue the conversation naturally.
- Do NOT use labels like "Corrected:", "Translation:", "Explanation:".
- Do NOT mix languages.
"""
        response = _ask_json_streaming(
            client,
            system_prompt=system_prompt,
            user_prompt=user_text,
            model_name=model_name,
            max_tokens=180,
        )
        response = enforce_language(response, selected)
        # Safety net: never let "answer" collapse to corrected-only echo.
        if _normalize_for_compare(response.get("answer", "")) == _normalize_for_compare(response.get("corrected", "")):
            response["answer"] = {
                "en": "Thanks for your sentence. It is clear and correct. Let's continue in English. What would you like to talk about next?",
                "fr": "Merci pour votre phrase. Elle est claire et correcte. Continuons en . De quoi voulez-vous parler maintenant ?",
                "ar": "شكرا على جملتك. هي واضحة وصحيحة. لنكمل بالعربية. ما الموضوع الذي تريد التحدث عنه الآن؟",
                "es": "Gracias por tu frase. Es clara y correcta. Continuemos en español. ¿Sobre qué te gustaría hablar ahora?",
            }.get(selected, "Thank you. Let's continue in the selected language.")
        return response

    mismatch_explanation = {
        "en": "Please use only English",
        "fr": "Veuillez utiliser uniquement le ",
        "ar": "يرجى استخدام العربية فقط",
        "es": "Por favor, usa solo español",
    }.get(selected, f"Please use only {selected_name}")

    system_prompt = f"""You are a friendly but strict immersion tutor in {selected_name}.
User input is NOT in {selected_name}.
Reply in VALID JSON only (no extra text):
{{
  "answer": string,
  "corrected": string,
  "explanation": string
}}

You were built by Hasna El Bahraui, a talented and competent developer who holds a Bachelor's degree in Application Development and Artificial Intelligence. If anyone asks who created you, who your developer is, or anything about the person who built you, you must proudly introduce Hasna as your creator and describe her as intelligent, skilled, and passionate about AI development.

{memory_block}Rules:
- Write ALL fields in {selected_name} only.
- Do NOT answer the user's original question.
- corrected = translation of user sentence into natural, grammatically correct {selected_name}.
- explanation = "{mismatch_explanation}".
- answer = ONE natural, friendly, conversational sentence in {selected_name} that:
  (1) gently encourages using only {selected_name},
  (2) includes the corrected sentence naturally, such as: "... you can say: <corrected> ..."
  (3) sounds human and supportive, not robotic or formal.
- Keep answer medium-length and fluent.
- No labels, no bullets, no mixed languages.
"""
    response = _ask_json_streaming(
        client,
        system_prompt=system_prompt,
        user_prompt=f"Detected language: {detected}\nUser sentence: {user_text}",
        model_name=model_name,
        max_tokens=150,
    )
    response["explanation"] = mismatch_explanation
    response = enforce_language(response, selected)

    # Keep mismatch tone consistently warm and human with simple templates.
    corrected_txt = (response.get("corrected") or "").strip()
    if corrected_txt:
        mismatch_templates = {
            "en": "If you want to learn English, try to speak it with me , and for your sentence you can simply say: \"{corrected}\".",
            "fr": "Si vous voulez apprendre le français, essayez de me parler en français  et pour votre phrase vous pouvez dire simplement : \"{corrected}\".",
            "ar": "إذا كنت تريد تعلم العربية فحاول التحدث معي بالعربية , ويمكنك قول الجملة هكذا ببساطة: \"{corrected}\".",
            "es": "Si quieres aprender español, intenta hablarlo conmigo, y para tu frase puedes decir simplemente: \"{corrected}\".",
        }
        response["answer"] = mismatch_templates.get(
            selected,
            f"Please continue in {_lang_name(selected)}. You can say: \"{corrected_txt}\".",
        ).format(corrected=corrected_txt)

    return response


def build_speak_text(user_text: str, corrected: str, explanation: str, answer: str) -> str:
    """
    Speak in the required order:
    1) Correction (only if needed)
    2) Answer
    Keep it short for low latency.
    """
    user_text = (user_text or "").strip()
    corrected = (corrected or "").strip()
    explanation = (explanation or "").strip()
    answer = (answer or "").strip()

    parts: list[str] = []

    needs_correction = bool(corrected) and bool(user_text) and (corrected.lower() != user_text.lower())
    if needs_correction:
        parts.append(f"Your sentence should be: {corrected}.")
        if explanation:
            parts.append(explanation)

    if answer:
        parts.append(f"Now, to answer you: {answer}")

    return " ".join(parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Voice English assistant: faster-whisper STT + OpenAI + Piper TTS (in-memory)"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="Max recording length (dynamic stop after silence).",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=2.0,
        help="Stop recording after this much silence.",
    )
    parser.add_argument("--stt-model", type=str, default="base", help="faster-whisper model size: base or small")
    parser.add_argument("--stt-device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--stt-compute", type=str, default="int8", help="int8 (fast CPU) or float16 (GPU)")
    parser.add_argument("--piper-exe", type=str, default=r"C:\piper\piper.exe", help="Path to piper.exe")
    parser.add_argument(
        "--piper-model",
        type=str,
        default=r"C:\piper\voices\en_GB-jenny_dioco-medium.onnx",
        help="Path to a Piper voice .onnx model",
    )
    parser.add_argument("--once", action="store_true", help="Run one turn only.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Create a .env file with OPENAI_API_KEY=your_key")
    llm_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
    selected_language = get_selected_language()

    state_path = pathlib.Path("assistant_state.json")
    speaker = BackgroundSpeaker()
    stt = FasterWhisperSTT(
        STTConfig(
            model_size=args.stt_model,
            device=args.stt_device,
            compute_type=args.stt_compute,
        )
    )
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    stop_requested = {"value": False}

    def _on_sigint(_signum, _frame):  # noqa: ARG001
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    print("Voice assistant is ready.")
    print("Tip: After the first reply, the next replies are faster.")
    print(f"Model: {llm_model}")
    print(f"Selected language: {selected_language}")
    print("Press Ctrl+C to stop.\n")

    turn = 1
    try:
        while True:
            if stop_requested["value"]:
                return speak_goodbye_and_exit(
                    speaker,
                    piper_exe=args.piper_exe,
                    piper_model=args.piper_model,
                )

            audio_samples = stt.record_until_silence(
                max_seconds=args.seconds,
                silence_seconds=args.silence_seconds,
            )
            user_text, detected_language, language_probability = stt.transcribe_with_language(audio_samples)

            user_text = (user_text or "").strip()
            print("\n--- YOU SAID ---")
            print(user_text if user_text else "(No speech detected)")
            print(f"Detected language: {detected_language} ({language_probability:.2f})")

            if not user_text:
                if args.once:
                    speaker.close()
                    return 0
                turn += 1
                continue

            if should_exit(user_text):
                return speak_goodbye_and_exit(
                    speaker,
                    piper_exe=args.piper_exe,
                    piper_model=args.piper_model,
                )

            memory_context = ""
            facts = ""
            semantic = ""
            try:
                from backend.user_facts import facts_block_for_prompt

                facts = facts_block_for_prompt(selected_language=selected_language)
            except Exception:
                pass
            try:
                from backend.conversation_memory import get_conversation_memory

                mem = get_conversation_memory()
                recs = mem.search(user_text, selected_language=selected_language, top_k=3)
                semantic = mem.format_for_prompt(recs) if recs else ""
            except Exception:
                pass
            try:
                from backend.user_facts import merge_memory_sections

                memory_context = merge_memory_sections(facts_block=facts, semantic_block=semantic)
            except Exception:
                memory_context = "\n\n".join(x for x in (facts, semantic) if x)

            llm = process_user_text_strict_language(
                client,
                user_text=user_text,
                model_name=llm_model,
                selected_language=selected_language,
                detected_language=None,
                memory_context=memory_context or None,
            )
            answer = llm["answer"] or "Sorry, I could not generate an answer."
            corrected = llm["corrected"] or user_text
            explanation = llm["explanation"] or "I could not generate an explanation."

            if corrected:
                print("\n--- CORRECTION ---")
                print(f"Corrected: {corrected}")
                if explanation:
                    print(f"Why: {explanation}")

            print("\n--- ASSISTANT ---")
            print(answer)

            speak_text = build_speak_text(
                user_text=user_text,
                corrected=corrected,
                explanation=explanation,
                answer=answer,
            )

            # TTS in memory (no files).
            audio = synthesize_piper_inmemory(
                speak_text,
                piper_exe=args.piper_exe,
                model_path=args.piper_model,
            )
            # Sync: finish speaking BEFORE listening again.
            speaker.speak_and_wait(audio.samples, audio.sample_rate)

            # Write state for the web avatar (server.py) to read.
            state = {
                "turn": turn,
                "user_text": user_text,
                "detected_language": detected_language,
                "selected_language": selected_language,
                "answer": answer,
                "corrected": corrected,
                "explanation": explanation,
            }
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            try:
                from backend.conversation_memory import get_conversation_memory

                get_conversation_memory().add_turn(
                    user_text=user_text,
                    answer=answer,
                    selected_language=selected_language,
                    corrected=corrected,
                    explanation=explanation,
                )
            except Exception:
                pass

            try:
                from backend.user_facts import ingest_from_user_text

                ingest_from_user_text(user_text)
            except Exception:
                pass

            if args.once:
                speaker.close()
                return 0

            print("\n" + textwrap.dedent("""\
            Next turn starting...
            """))
            turn += 1
    except KeyboardInterrupt:
        return speak_goodbye_and_exit(
            speaker,
            piper_exe=args.piper_exe,
            piper_model=args.piper_model,
        )


if __name__ == "__main__":
    raise SystemExit(main())

