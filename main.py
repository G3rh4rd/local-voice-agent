#!/usr/bin/env python3
# =============================================================================
# main.py — Real-time Interruptible Voice-to-Voice Agent
# =============================================================================
#
# Pipeline:
#   Microphone → Silero VAD → FasterWhisper (CPU) → vLLM (streaming)
#               → Punctuation chunking → Kokoro TTS → Speaker
#
# Threading model:
#   ┌─────────────────┐
#   │  sounddevice    │  Audio capture callback (runs in PortAudio thread)
#   │  InputStream    │──→ audio_ring deque
#   └─────────────────┘
#          ↓
#   ┌─────────────────┐
#   │   VAD thread    │  Silero VAD — detects speech start/end
#   │                 │  Barge-in: sets interrupt_event if agent is speaking
#   └─────────────────┘──→ speech_queue (complete utterance audio)
#          ↓
#   ┌─────────────────┐
#   │ Pipeline thread │  Whisper STT → vLLM (streaming) → Kokoro TTS chunks
#   │                 │  Respects interrupt_event at every yield boundary
#   └─────────────────┘──→ tts_audio_queue (float32 numpy arrays)
#          ↓
#   ┌─────────────────┐
#   │ Playback thread │  Plays TTS chunks via sounddevice
#   │                 │  Polls interrupt_event every 50ms during playback
#   └─────────────────┘
#
# Barge-in flow:
#   1. VAD thread detects speech while is_agent_speaking is set
#   2. interrupt_event.set()
#   3. Playback thread: stops sounddevice, drains tts_audio_queue
#   4. Pipeline thread: breaks out of LLM streaming loop
#   5. All threads reset interrupt_event and resume listening
# =============================================================================

import sys
import os
import glob
import threading
import queue
import time
import re
import collections
import numpy as np
import sounddevice as sd
import torch
import yaml

# ─── Load config.yaml ──────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[CONFIG] {path} not found — using built-in defaults.")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_cfg = _load_config(_CONFIG_PATH)

def _get(keys: str, default):
    """Dot-path accessor into the nested config dict, e.g. 'stt.language'."""
    node = _cfg
    for k in keys.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node

print(f"[CONFIG] Loaded {_CONFIG_PATH}")

# ─── Configuration ─────────────────────────────────────────────────────────────

# Audio (fixed — Silero VAD and Whisper require 16 kHz)
SAMPLE_RATE     = 16_000
CHUNK_SAMPLES   = 512           # Must be 256/512/768 for Silero VAD

AUDIO_DEVICE    = _get("audio.device", "pulse")
AUDIO_GAIN      = float(_get("audio.gain", 1.5))

# VAD
VAD_THRESHOLD   = float(_get("vad.threshold", 0.30))
_silence_secs   = float(_get("vad.silence_duration", 0.4))
SILENCE_CHUNKS  = max(1, int(_silence_secs * SAMPLE_RATE / CHUNK_SAMPLES))
MIN_SPEECH_CHUNKS = 3
MAX_SPEECH_SECS   = 30

# STT
STT_LANGUAGE    = _get("stt.language", None)    # None = auto-detect

# vLLM / LLM
VLLM_BASE_URL   = _get("llm.base_url", "http://localhost:8000/v1")
VLLM_API_KEY    = "not-needed"
LLM_MAX_TOKENS  = int(_get("llm.max_tokens", 512))
LLM_TEMPERATURE = float(_get("llm.temperature", 0.7))
SYSTEM_PROMPT   = _get("llm.system_prompt",
    "You are a helpful, concise voice assistant. "
    "Respond in short, natural sentences suitable for text-to-speech output. "
    "Avoid markdown, bullet points, numbered lists, and special characters.")

# Kokoro TTS
TTS_VOICE       = _get("tts.voice", "af_heart")
TTS_SPEED       = float(_get("tts.speed", 1.0))
TTS_LANG        = _get("tts.lang", "en-us")     # fixed from config (not auto-detected)
TTS_SAMPLE_RATE = 24_000        # Kokoro outputs at 24 kHz

# Punctuation pattern: flush TTS chunk when we detect a sentence/clause boundary.
# Matches end-of-sentence punctuation followed by whitespace OR end of token.
PUNCT_SPLIT_RE  = re.compile(r'(?<=[.!?])\s+|(?<=[,;:])\s+')

# Model paths (resolved dynamically from ./models/)
KOKORO_ONNX_PATH    = "./models/kokoro/kokoro-v1.0.int8.onnx"
KOKORO_VOICES_PATH  = "./models/kokoro/voices-v1.0.bin"


# ─── Shared State ──────────────────────────────────────────────────────────────

# sounddevice callback → VAD thread: ring buffer of raw audio chunks
_MAX_RING_SIZE  = int(MAX_SPEECH_SECS * SAMPLE_RATE / CHUNK_SAMPLES) + 50
audio_ring      = collections.deque(maxlen=_MAX_RING_SIZE)
audio_ring_lock = threading.Lock()
audio_ring_event= threading.Event()    # signals that new audio is available

# VAD thread → pipeline thread: complete speech utterances (numpy float32)
speech_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)

# Pipeline thread → playback thread: TTS audio chunks + None sentinel
tts_audio_queue: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=128)

# Control events
interrupt_event     = threading.Event()  # set to stop TTS + LLM stream
is_agent_speaking   = threading.Event()  # set while TTS audio is playing
shutdown_event      = threading.Event()  # set to terminate all threads

# Conversation history — protected by lock
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
conv_lock = threading.Lock()

# Module-level model handles (populated in load_models())
vad_model       = None
whisper_model   = None
kokoro_tts      = None
llm_client      = None
vllm_model_id   = None  # cached after first models.list() call


# ─── Utility: Model Path Resolution ────────────────────────────────────────────

def find_snapshot(repo_dir_name: str) -> str:
    """
    Locate the downloaded model directory.
    snapshot_download(local_dir=...) places files directly in:
        ./models/<repo_dir_name>/
    (no snapshots/<hash>/ subdirectory when local_dir is specified)
    """
    path = os.path.join("./models", repo_dir_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Model directory not found: {path}\n"
            "Run download_models.sh first."
        )
    return path


# ─── Model Loading ─────────────────────────────────────────────────────────────

def load_models() -> None:
    """Load all models into global variables. Called once at startup."""
    global vad_model, whisper_model, kokoro_tts, llm_client, vllm_model_id

    # ── 1. Silero VAD ──────────────────────────────────────────────────────────
    print("[INIT] Loading Silero VAD (torch.hub)...")
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
    vad_model.eval()
    print("[INIT] Silero VAD ready.")

    # ── 2. FasterWhisper (CPU, int8) ───────────────────────────────────────────
    print("[INIT] Loading FasterWhisper small (CPU / int8)...")
    from faster_whisper import WhisperModel
    whisper_path = find_snapshot("models--Systran--faster-whisper-small")
    whisper_model = WhisperModel(
        whisper_path,
        device="cpu",
        compute_type="int8",
        num_workers=2,          # use 2 CPU workers for parallel decoding
        cpu_threads=4,          # limit threads so we don't starve the LLM
    )
    print(f"[INIT] FasterWhisper ready  ({whisper_path})")

    # ── 3. Kokoro TTS (ONNX, CPU) ──────────────────────────────────────────────
    print("[INIT] Loading Kokoro TTS (ONNX)...")
    from kokoro_onnx import Kokoro
    if not os.path.exists(KOKORO_ONNX_PATH):
        raise FileNotFoundError(
            f"Kokoro model not found: {KOKORO_ONNX_PATH}\n"
            "Run download_models.sh first (downloads from GitHub releases)."
        )
    if not os.path.exists(KOKORO_VOICES_PATH):
        raise FileNotFoundError(
            f"Kokoro voices not found: {KOKORO_VOICES_PATH}\n"
            "Run download_models.sh first (downloads from GitHub releases)."
        )
    kokoro_tts = Kokoro(KOKORO_ONNX_PATH, KOKORO_VOICES_PATH)
    print(f"[INIT] Kokoro TTS ready  (voice={TTS_VOICE})")

    # ── 4. vLLM OpenAI client ──────────────────────────────────────────────────
    print("[INIT] Connecting to vLLM endpoint...")
    from openai import OpenAI
    llm_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    try:
        models = llm_client.models.list()
        vllm_model_id = models.data[0].id
        print(f"[INIT] vLLM connected  (model={vllm_model_id})")
    except Exception as exc:
        print(f"[WARN] vLLM health-check failed: {exc}")
        print("[WARN] Ensure start_vllm.sh container is running and healthy.")
        # Fall back to a known model name so we can still attempt inference
        vllm_model_id = "Qwen2.5-14B-Instruct-AWQ"

    print("[INIT] All models loaded.\n")


# ─── Audio Capture Callback ────────────────────────────────────────────────────

def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """
    Called by sounddevice in the PortAudio audio thread.
    Must be fast — only copies data and signals the VAD thread.
    """
    if status:
        print(f"[AUDIO] {status}", file=sys.stderr)
    # Extract mono channel, apply gain to compensate for quiet microphones
    chunk = indata[:, 0].copy().astype(np.float32) * AUDIO_GAIN
    np.clip(chunk, -1.0, 1.0, out=chunk)  # prevent clipping after gain
    with audio_ring_lock:
        audio_ring.append(chunk)
    audio_ring_event.set()


# ─── VAD Thread ────────────────────────────────────────────────────────────────

def vad_thread_fn() -> None:
    """
    Reads audio chunks from audio_ring, runs Silero VAD on each chunk.

    State machine:
        IDLE  →  speech detected  →  RECORDING
        RECORDING  →  silence ≥ SILENCE_CHUNKS  →  emit utterance  →  IDLE

    Barge-in: if the agent is currently speaking (is_agent_speaking is set)
    and new speech is detected, interrupt_event is set immediately.
    """
    print("[VAD] Thread started.")

    speech_buffer: list[np.ndarray] = []
    silence_count: int = 0
    in_speech: bool = False

    while not shutdown_event.is_set():
        # Block until new audio is available (with timeout to check shutdown)
        audio_ring_event.wait(timeout=0.1)
        audio_ring_event.clear()

        # Drain all available chunks from the ring buffer
        chunks: list[np.ndarray] = []
        with audio_ring_lock:
            while audio_ring:
                chunks.append(audio_ring.popleft())

        for chunk in chunks:
            # ── Run Silero VAD ──────────────────────────────────────────────
            tensor = torch.from_numpy(chunk).unsqueeze(0)   # shape: [1, T]
            with torch.no_grad():
                confidence: float = vad_model(tensor, SAMPLE_RATE).item()
            is_speech = confidence >= VAD_THRESHOLD

            # Debug: print confidence whenever it's non-trivial
            if confidence > 0.05 or in_speech:
                print(f"[VAD] conf={confidence:.3f} rms={float(np.sqrt(np.mean(chunk**2))):.4f}"
                      f"{'  SPEECH' if is_speech else ''}", end="\r")

            # ── Barge-in detection ─────────────────────────────────────────
            # If agent is speaking and user starts talking, interrupt immediately.
            if is_speech and is_agent_speaking.is_set():
                if not interrupt_event.is_set():
                    print("[VAD] *** Barge-in detected! Interrupting agent. ***")
                    interrupt_event.set()

            # ── State machine ───────────────────────────────────────────────
            if is_speech:
                if not in_speech:
                    print("[VAD] Speech start ↑")
                    in_speech = True
                    speech_buffer = []
                    silence_count = 0
                speech_buffer.append(chunk)
                silence_count = 0

            else:   # silence
                if in_speech:
                    silence_count += 1
                    speech_buffer.append(chunk)  # include trailing silence for natural pacing

                    # Check if we've accumulated enough silence to end the utterance
                    if silence_count >= SILENCE_CHUNKS:
                        utterance = np.concatenate(speech_buffer)
                        duration_s = len(utterance) / SAMPLE_RATE
                        print(f"[VAD] Speech end ↓  (duration={duration_s:.2f}s, "
                              f"chunks={len(speech_buffer)})")

                        if len(speech_buffer) >= MIN_SPEECH_CHUNKS:
                            try:
                                speech_queue.put(utterance, timeout=0.5)
                            except queue.Full:
                                print("[VAD] speech_queue full — utterance dropped.")
                        else:
                            print("[VAD] Utterance too short, discarding (noise).")

                        # Reset
                        in_speech = False
                        speech_buffer = []
                        silence_count = 0

    print("[VAD] Thread stopped.")


# ─── Pipeline Thread ────────────────────────────────────────────────────────────

WHISPER_TO_ESPEAK_LANG = {
    "en": "en-us",
    "de": "de",
    "fr": "fr-fr",
    "es": "es",
    "it": "it",
    "pt": "pt-br",
    "nl": "nl",
    "pl": "pl",
    "ja": "ja",
    "zh": "zh",
}

def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a numpy float32 audio array using FasterWhisper.
    Runs on CPU with int8 quantization.
    """
    # Reject audio that is too quiet to contain real speech
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.01:
        print(f"[STT]  Audio too quiet (RMS={rms:.4f}), skipping.")
        return ""

    segments, info = whisper_model.transcribe(
        audio,
        language=STT_LANGUAGE,  # None = auto-detect, or e.g. "de", "en"
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,            # Whisper's internal VAD removes silence & reduces hallucinations
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        no_speech_threshold=0.6,    # Reject segment if Whisper thinks it's not speech
        log_prob_threshold=-1.0,    # Reject low-confidence segments
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"[STT]  RMS={rms:.4f}  lang={info.language}({info.language_probability:.2f})  {text!r}")
    return text


def generate_tts_audio(text: str) -> "np.ndarray | None":
    """
    Generate TTS audio for a single text chunk using Kokoro ONNX.
    Returns float32 numpy array at TTS_SAMPLE_RATE, or None on error.
    """
    text = text.strip()
    if not text:
        return None
    try:
        samples, sr = kokoro_tts.create(
            text,
            voice=TTS_VOICE,
            speed=TTS_SPEED,
            lang=TTS_LANG,
        )
        return samples.astype(np.float32)
    except Exception as exc:
        print(f"[TTS]  Error generating audio for {text!r}: {exc}")
        return None


def pipeline_thread_fn() -> None:
    """
    Main pipeline loop:
      1. Wait for a speech utterance from speech_queue
      2. Transcribe with FasterWhisper
      3. Send to vLLM (streaming)
      4. Accumulate tokens; flush to TTS at punctuation boundaries
      5. Put TTS audio chunks into tts_audio_queue
      6. Emit a None sentinel to signal end-of-turn to playback thread
    """
    print("[PIPELINE] Thread started.")

    while not shutdown_event.is_set():
        # ── Wait for next utterance ─────────────────────────────────────────
        try:
            utterance: np.ndarray = speech_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Clear any leftover interrupt from the previous turn
        interrupt_event.clear()

        # ── 1. Transcription ───────────────────────────────────────────────
        text = transcribe(utterance)
        if not text:
            print("[PIPELINE] Empty transcription — skipping turn.")
            continue

        # ── 2. Build conversation context ──────────────────────────────────
        with conv_lock:
            conversation_history.append({"role": "user", "content": text})
            messages_snapshot = list(conversation_history)

        print(f"[LLM]  Sending query to vLLM...")

        # ── 3. Streaming LLM inference ─────────────────────────────────────
        full_response_tokens: list[str] = []
        text_buffer: str = ""           # accumulates tokens between punctuation
        tts_chunks_queued: int = 0

        try:
            stream = llm_client.chat.completions.create(
                model=vllm_model_id,
                messages=messages_snapshot,
                stream=True,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )

            for chunk in stream:
                # ── Barge-in check on every token ──────────────────────────
                if interrupt_event.is_set():
                    print("[PIPELINE] Interrupt — cancelling LLM stream.")
                    try:
                        stream.close()
                    except Exception:
                        pass
                    break

                delta = chunk.choices[0].delta.content
                if delta is None:
                    continue

                full_response_tokens.append(delta)
                text_buffer += delta

                # ── 4. Punctuation-based chunking ───────────────────────────
                # Split on sentence/clause boundaries; keep the last partial
                # piece in the buffer until the next punctuation arrives.
                parts = PUNCT_SPLIT_RE.split(text_buffer)

                if len(parts) > 1:
                    # All parts except the last are complete clauses → TTS
                    for part in parts[:-1]:
                        part = part.strip()
                        if not part:
                            continue
                        if interrupt_event.is_set():
                            break
                        print(f"[TTS]  Chunk: {part!r}")
                        audio = generate_tts_audio(part)
                        if audio is not None and not interrupt_event.is_set():
                            tts_audio_queue.put(audio)
                            tts_chunks_queued += 1
                    # Keep the incomplete last part for the next token
                    text_buffer = parts[-1]

        except Exception as exc:
            print(f"[LLM]  Error during streaming: {exc}")

        # ── Flush remaining buffer ──────────────────────────────────────────
        if text_buffer.strip() and not interrupt_event.is_set():
            print(f"[TTS]  Final chunk: {text_buffer.strip()!r}")
            audio = generate_tts_audio(text_buffer.strip())
            if audio is not None:
                tts_audio_queue.put(audio)
                tts_chunks_queued += 1

        # ── Always send sentinel to unblock playback thread ────────────────
        tts_audio_queue.put(None)

        # ── 5. Update conversation history ─────────────────────────────────
        if full_response_tokens and not interrupt_event.is_set():
            full_response = "".join(full_response_tokens)
            with conv_lock:
                conversation_history.append(
                    {"role": "assistant", "content": full_response}
                )
            print(f"[PIPELINE] Turn complete "
                  f"({tts_chunks_queued} TTS chunks, "
                  f"{len(full_response_tokens)} tokens).")
        elif interrupt_event.is_set():
            print("[PIPELINE] Turn interrupted — response not saved to history.")

    print("[PIPELINE] Thread stopped.")


# ─── Playback Thread ───────────────────────────────────────────────────────────

def _drain_tts_queue() -> int:
    """
    Discard all items in tts_audio_queue up to and including the next
    None sentinel. Returns the number of items drained.
    """
    drained = 0
    while True:
        try:
            item = tts_audio_queue.get_nowait()
            drained += 1
            if item is None:    # sentinel — end of this turn
                break
        except queue.Empty:
            break
    if drained:
        print(f"[PLAYBACK] Drained {drained} items from queue.")
    return drained


def playback_thread_fn() -> None:
    """
    Reads TTS audio chunks from tts_audio_queue and plays them sequentially.

    - Sets is_agent_speaking while audio is playing (enables barge-in).
    - Polls interrupt_event every 50 ms during playback.
    - On interrupt: stops sounddevice immediately, drains the queue.
    - None sentinel → clears is_agent_speaking, ready for next turn.
    """
    print("[PLAYBACK] Thread started.")
    POLL_INTERVAL = 0.05    # 50 ms polling granularity during playback

    while not shutdown_event.is_set():
        # ── Wait for next audio chunk ───────────────────────────────────────
        try:
            audio_chunk = tts_audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # ── Sentinel: end of turn ───────────────────────────────────────────
        if audio_chunk is None:
            is_agent_speaking.clear()
            continue

        # ── Pre-play interrupt check ────────────────────────────────────────
        if interrupt_event.is_set():
            # Already interrupted — skip this chunk and drain the rest
            _drain_tts_queue()
            is_agent_speaking.clear()
            continue

        # ── Play audio chunk ────────────────────────────────────────────────
        is_agent_speaking.set()
        try:
            sd.play(audio_chunk, samplerate=TTS_SAMPLE_RATE, blocking=False)

            chunk_duration_s = len(audio_chunk) / TTS_SAMPLE_RATE
            elapsed = 0.0

            # Poll for interrupt while the chunk is playing
            while elapsed < chunk_duration_s:
                if interrupt_event.is_set():
                    sd.stop()                   # immediate hardware stop
                    _drain_tts_queue()
                    is_agent_speaking.clear()
                    print("[PLAYBACK] Playback interrupted mid-chunk.")
                    break
                time.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL
            else:
                # Chunk finished normally — wait for sounddevice to flush
                sd.wait()

        except Exception as exc:
            print(f"[PLAYBACK] Error during playback: {exc}")
            sd.stop()
            is_agent_speaking.clear()

    print("[PLAYBACK] Thread stopped.")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  Local Voice-to-Voice Agent")
    print("  RTX 4080 (LLM via Docker/vLLM) + Intel i9 (STT/TTS on CPU)")
    print("=" * 62)

    # ── Load models ────────────────────────────────────────────────────────────
    load_models()

    # ── Start worker threads ───────────────────────────────────────────────────
    print("[MAIN] Starting worker threads...")
    worker_threads = [
        threading.Thread(target=vad_thread_fn,      name="VAD",      daemon=True),
        threading.Thread(target=pipeline_thread_fn, name="Pipeline", daemon=True),
        threading.Thread(target=playback_thread_fn, name="Playback", daemon=True),
    ]
    for t in worker_threads:
        t.start()

    # ── Open microphone and enter main loop ────────────────────────────────────
    print(f"[MAIN] Opening microphone (rate={SAMPLE_RATE} Hz, "
          f"chunk={CHUNK_SAMPLES} samples)...")

    try:
        with sd.InputStream(
            device=AUDIO_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback,
            latency="low",
        ):
            print("\n[MAIN] Voice agent is READY — speak into your microphone.")
            print("[MAIN] Press Ctrl+C to exit.\n")

            while not shutdown_event.is_set():
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[MAIN] Caught Ctrl+C — shutting down...")

    except Exception as exc:
        print(f"\n[MAIN] Fatal error: {exc}")

    finally:
        shutdown_event.set()
        interrupt_event.set()   # unblock any blocked thread

        print("[MAIN] Waiting for threads to finish...")
        for t in worker_threads:
            t.join(timeout=3.0)
            if t.is_alive():
                print(f"[MAIN] Thread {t.name} did not stop cleanly.")

        print("[MAIN] Goodbye.")


if __name__ == "__main__":
    main()
