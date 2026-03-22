# ============================================================
#  DIY Voice Assistant — Backend Server
#  FastAPI for ESP32 Voice Assistant
#
#  Pipeline: Audio (HTTP POST) → Whisper STT → Groq LLM → gTTS → Audio back
#
#  Deploy on Render as a Web Service (Docker)
# ============================================================

import os
import io
import wave
import tempfile
import logging
import urllib.parse
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from groq import Groq
from gtts import gTTS

load_dotenv()

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-assistant")

# ── Config ───────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not set! Add it to .env or environment variables.")
    raise ValueError("GROQ_API_KEY is required")

# Audio format from ESP32 (must match config.h)
SAMPLE_RATE     = 16000
BITS_PER_SAMPLE = 16
CHANNELS        = 1

# A2DP playback format (44100Hz stereo for Bluetooth)
BT_SAMPLE_RATE  = 44100
BT_CHANNELS     = 2

# LLM system prompt — customize your assistant's personality!
SYSTEM_PROMPT = """You are Jarvis, a helpful and witty AI voice assistant. 
You are running on a DIY ESP32-based device. Keep your responses concise 
and conversational — ideally under 2-3 sentences since they'll be spoken aloud.
Be helpful, friendly, and occasionally humorous."""

# ── Initialize Clients ──────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(title="Voice Assistant Backend", version="1.0")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "service": "voice-assistant-backend"}


@app.get("/health")
async def health():
    """Health check for Render."""
    return {"status": "healthy"}


@app.post("/process")
async def process_audio(audio: UploadFile = File(...)):
    """
    Receive WAV audio from ESP32, process through STT → LLM → TTS pipeline,
    and return raw PCM audio (44100Hz, stereo, 16-bit) for A2DP playback.
    
    Custom response headers:
      X-Transcript: what the user said
      X-Response: what the AI responds
    """
    try:
        audio_data = await audio.read()
        logger.info(f"Received {len(audio_data)} bytes of audio")

        if len(audio_data) < 100:
            return Response(content=b"", status_code=400)

        # ── Step 1: Speech-to-Text (Whisper via Groq) ────────
        logger.info("[STT] Transcribing audio with Whisper...")

        transcript = transcribe_audio(audio_data)

        if not transcript or transcript.strip() == "":
            logger.warning("[STT] Empty transcript")
            return Response(content=b"", status_code=400)

        logger.info(f"[STT] Transcript: {transcript}")

        # ── Step 2: LLM Response (Groq) ──────────────────────
        logger.info("[LLM] Getting AI response...")

        response_text = get_llm_response(transcript)
        logger.info(f"[LLM] Response: {response_text}")

        # ── Step 3: Text-to-Speech → PCM for A2DP ────────────
        logger.info("[TTS] Converting response to speech...")

        pcm_audio = text_to_pcm(response_text)
        logger.info(f"[TTS] Generated {len(pcm_audio)} bytes of PCM audio")

        # ── Return audio with metadata headers ───────────────
        headers = {
            "X-Transcript": urllib.parse.quote(transcript[:200]),
            "X-Response": urllib.parse.quote(response_text[:200]),
            "Content-Type": "application/octet-stream"
        }

        return Response(
            content=pcm_audio,
            media_type="application/octet-stream",
            headers=headers
        )

    except Exception as e:
        logger.error(f"[ERROR] Pipeline failed: {e}", exc_info=True)
        return Response(content=b"", status_code=500)


def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio using Groq's Whisper API."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language="en",
                response_format="text"
            )
        return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
    finally:
        os.unlink(tmp_path)


def get_llm_response(user_message: str) -> str:
    """Get a response from Groq's LLM."""
    chat_completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=150,
        top_p=0.9
    )
    return chat_completion.choices[0].message.content.strip()


def text_to_pcm(text: str) -> bytes:
    """
    Convert text to raw PCM audio suitable for A2DP playback.
    gTTS → MP3 → pydub → raw PCM (44100Hz, stereo, 16-bit).
    """
    tts = gTTS(text=text, lang="en", slow=False)
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(mp3_buffer)
        audio = audio.set_frame_rate(BT_SAMPLE_RATE)
        audio = audio.set_channels(BT_CHANNELS)
        audio = audio.set_sample_width(2)  # 16-bit

        return audio.raw_data

    except ImportError:
        logger.warning("pydub not available, returning MP3 directly")
        mp3_buffer.seek(0)
        return mp3_buffer.read()


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
