# ============================================================
#  DIY Voice Assistant — Backend Server
#  FastAPI + WebSocket for ESP32 Voice Assistant
#
#  Pipeline: Audio (WebSocket) → Whisper STT → Groq LLM → gTTS → Audio back
#
#  Deploy on Render as a Web Service (Docker)
# ============================================================

import os
import io
import wave
import struct
import tempfile
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
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
SAMPLE_RATE    = 16000
BITS_PER_SAMPLE = 16
CHANNELS       = 1

# A2DP playback format (44100Hz stereo for Bluetooth)
BT_SAMPLE_RATE = 44100
BT_CHANNELS    = 2

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
async def process_http(request: Request):
    """
    HTTP endpoint for ESP32 voice assistant (fallback when WebSocket SSL fails).
    
    Accepts: Raw PCM audio bytes (16kHz, 16-bit, mono) in request body
    Returns: Raw PCM audio bytes (44100Hz, 16-bit, stereo) for A2DP playback
    
    Headers returned:
      X-Transcript: what the user said
      X-Response: what the AI responded
    """
    try:
        # Read raw PCM audio from request body
        audio_data = await request.body()
        logger.info(f"[HTTP] Received {len(audio_data)} bytes of audio")

        if len(audio_data) < 1000:
            return Response(content=b"", status_code=400)

        # Step 1: STT
        wav_buffer = create_wav(bytearray(audio_data))
        transcript = transcribe_audio(wav_buffer)
        if not transcript or transcript.strip() == "":
            logger.warning("[STT] Empty transcript")
            return Response(content=b"", status_code=400)
        logger.info(f"[STT] Transcript: {transcript}")

        # Step 2: LLM
        response_text = get_llm_response(transcript)
        logger.info(f"[LLM] Response: {response_text}")

        # Step 3: TTS → PCM
        pcm_audio = text_to_pcm(response_text)
        logger.info(f"[TTS] Generated {len(pcm_audio)} bytes of PCM")

        # Return PCM audio with transcript/response in headers
        return Response(
            content=pcm_audio,
            media_type="application/octet-stream",
            headers={
                "X-Transcript": transcript[:200],
                "X-Response": response_text[:200],
            }
        )
    except Exception as e:
        logger.error(f"[ERROR] HTTP pipeline failed: {e}")
        return Response(content=str(e).encode(), status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for the ESP32 voice assistant.
    
    Protocol:
      ESP32 → Server:
        - "START"     : Begin new utterance
        - binary data : Audio chunks (16kHz, 16-bit, mono PCM)
        - "END"       : Done speaking, process the audio
      
      Server → ESP32:
        - "TRANSCRIPT:..." : What the user said
        - "RESPONSE:..."   : What the AI responds
        - "AUDIO_START"    : About to send audio
        - binary data      : Raw PCM audio (44100Hz, 16-bit, stereo)
        - "AUDIO_END"      : Done sending audio
        - "ERROR"          : Something went wrong
    """
    await websocket.accept()
    logger.info("ESP32 client connected")

    audio_buffer = bytearray()
    is_recording = False

    try:
        while True:
            data = await websocket.receive()

            # Handle text messages (control commands)
            if "text" in data:
                message = data["text"]

                if message == "START":
                    logger.info("─── New utterance started ───")
                    audio_buffer = bytearray()
                    is_recording = True

                elif message == "END" and is_recording:
                    is_recording = False
                    logger.info(f"Utterance complete. Audio size: {len(audio_buffer)} bytes")

                    if len(audio_buffer) < 1000:
                        logger.warning("Audio too short, ignoring")
                        await websocket.send_text("ERROR")
                        continue

                    # Process the audio through the pipeline
                    await process_audio(websocket, audio_buffer)
                    audio_buffer = bytearray()

            # Handle binary messages (audio data)
            elif "bytes" in data:
                if is_recording:
                    audio_buffer.extend(data["bytes"])

    except WebSocketDisconnect:
        logger.info("ESP32 client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text("ERROR")
        except:
            pass


async def process_audio(websocket: WebSocket, audio_data: bytearray):
    """
    Full pipeline: Audio → STT → LLM → TTS → Audio response
    """
    try:
        # ── Step 1: Speech-to-Text (Whisper via Groq) ────────────
        logger.info("[STT] Transcribing audio with Whisper...")

        # Create a WAV file from raw PCM data
        wav_buffer = create_wav(audio_data)

        # Use Groq's Whisper API for fast transcription
        transcript = transcribe_audio(wav_buffer)

        if not transcript or transcript.strip() == "":
            logger.warning("[STT] Empty transcript")
            await websocket.send_text("ERROR")
            return

        logger.info(f"[STT] Transcript: {transcript}")
        await websocket.send_text(f"TRANSCRIPT:{transcript}")

        # ── Step 2: LLM Response (Groq) ──────────────────────────
        logger.info("[LLM] Getting AI response...")

        response_text = get_llm_response(transcript)
        logger.info(f"[LLM] Response: {response_text}")
        await websocket.send_text(f"RESPONSE:{response_text}")

        # ── Step 3: Text-to-Speech (gTTS → PCM for A2DP) ────────
        logger.info("[TTS] Converting response to speech...")

        pcm_audio = text_to_pcm(response_text)
        logger.info(f"[TTS] Generated {len(pcm_audio)} bytes of PCM audio")

        # ── Step 4: Send audio back to ESP32 ─────────────────────
        await websocket.send_text("AUDIO_START")

        # Send in chunks to avoid overwhelming the ESP32
        chunk_size = 4096
        for i in range(0, len(pcm_audio), chunk_size):
            chunk = pcm_audio[i:i + chunk_size]
            await websocket.send_bytes(chunk)

        await websocket.send_text("AUDIO_END")
        logger.info("[DONE] Response sent to ESP32")

    except Exception as e:
        logger.error(f"[ERROR] Pipeline failed: {e}")
        await websocket.send_text("ERROR")


def create_wav(pcm_data: bytearray) -> io.BytesIO:
    """Create a WAV file in memory from raw PCM data."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(BITS_PER_SAMPLE // 8)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(bytes(pcm_data))
    wav_buffer.seek(0)
    return wav_buffer


def transcribe_audio(wav_buffer: io.BytesIO) -> str:
    """Transcribe audio using Groq's Whisper API."""
    # Save to temp file (Groq SDK needs a file-like object with a name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_buffer.read())
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
        max_tokens=150,  # Keep responses short for voice
        top_p=0.9
    )
    return chat_completion.choices[0].message.content.strip()


def text_to_pcm(text: str) -> bytes:
    """
    Convert text to raw PCM audio suitable for A2DP playback.
    
    gTTS produces MP3 → we convert to raw PCM (44100Hz, 16-bit, stereo)
    using the audioop/wave approach to avoid needing ffmpeg.
    
    Since we can't easily decode MP3 on the ESP32 without a decoder library,
    we convert server-side and send raw PCM that the A2DP callback can 
    directly stream to the Bluetooth speaker.
    """
    # Generate MP3 with gTTS
    tts = gTTS(text=text, lang="en", slow=False)
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    # Decode MP3 to PCM using pydub (needs ffmpeg on the server)
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(mp3_buffer)
        # Convert to A2DP format: 44100Hz, stereo, 16-bit
        audio = audio.set_frame_rate(BT_SAMPLE_RATE)
        audio = audio.set_channels(BT_CHANNELS)
        audio = audio.set_sample_width(2)  # 16-bit

        return audio.raw_data

    except ImportError:
        logger.warning("pydub not available, sending MP3 directly")
        logger.warning("Install pydub and ffmpeg for proper audio conversion")
        mp3_buffer.seek(0)
        return mp3_buffer.read()


# ── Run with uvicorn ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
