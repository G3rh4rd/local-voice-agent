# Local Voice Agent

A real-time, interruptible voice-to-voice assistant running entirely on local hardware. No cloud APIs, no data leaving your machine.

**Hardware:** RTX 4080 (12 GB) + Intel i9
**Stack:** Silero VAD → FasterWhisper → vLLM (Qwen2.5-14B) → Kokoro TTS

---

## How it works

```
Microphone → Silero VAD → FasterWhisper (CPU) → vLLM / Qwen2.5-14B (GPU) → Kokoro TTS (CPU) → Speaker
```

| Component | Model | Device |
|---|---|---|
| Voice activity detection | Silero VAD | CPU |
| Speech-to-text | FasterWhisper small (int8) | CPU |
| Language model | Qwen2.5-14B-Instruct-AWQ | GPU via Docker/vLLM |
| Text-to-speech | Kokoro-82M ONNX (int8) | CPU |

**Barge-in:** If you speak while the agent is talking, it stops immediately and listens to you.

---

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download models

Downloads all models (~10 GB total) to `./models/`:

```bash
./download_models.sh
```

| Model | Size | Purpose |
|---|---|---|
| Qwen2.5-14B-Instruct-AWQ | ~8 GB | Language model |
| faster-whisper-small | ~250 MB | Speech recognition |
| Kokoro-82M ONNX int8 | ~88 MB | Text-to-speech |
| Silero VAD | ~2 MB | Voice activity detection |

### 3. Configure

Edit `config.yaml` before first run:

```yaml
stt:
  language: "de"        # "de" for German, "en" for English, null for auto-detect

tts:
  lang: "de"            # must match stt.language

llm:
  system_prompt: >
    Du bist ein hilfreicher Sprachassistent ...
```

See [Configuration](#configuration) for all options.

### 4. Start the LLM backend

In a **separate terminal**, run vLLM in Docker:

```bash
./start_vllm.sh
```

Wait until you see `Uvicorn running on ...` before starting the agent.

### 5. Run the agent

```bash
source venv/bin/activate
python main.py
```

---

## Configuration

All settings live in `config.yaml`. The agent picks up changes on the next restart.

```yaml
audio:
  device: "pulse"     # audio input device ("pulse", null = system default, or device index)
  gain: 1.5           # microphone pre-amplification (increase if mic is too quiet)

stt:
  language: "de"      # Whisper language code — null enables auto-detection

llm:
  base_url: "http://localhost:8000/v1"
  max_tokens: 512
  temperature: 0.7
  system_prompt: >
    Du bist ein hilfreicher, prägnanter Sprachassistent ...

tts:
  voice: "af_heart"   # Kokoro voice name (see available voices below)
  speed: 1.0          # speech rate (0.5 – 2.0)
  lang: "de"          # eSpeak-NG language code passed to Kokoro

vad:
  threshold: 0.30     # Silero confidence threshold (0.0 – 1.0)
  silence_duration: 0.4  # seconds of silence that end an utterance
```

### Language codes

| Language | `stt.language` | `tts.lang` |
|---|---|---|
| German | `de` | `de` |
| English (US) | `en` | `en-us` |
| English (UK) | `en` | `en-gb` |
| French | `fr` | `fr-fr` |
| Spanish | `es` | `es` |
| Italian | `it` | `it` |
| Japanese | `ja` | `ja` |

### Available TTS voices

The `voices-v1.0.bin` bundle includes:

| Prefix | Description |
|---|---|
| `af_*` | American Female (af_heart, af_bella, af_sarah, af_nova, …) |
| `am_*` | American Male (am_adam, am_michael, am_echo, …) |
| `bf_*` | British Female (bf_emma, bf_alice, bf_isabella, …) |
| `bm_*` | British Male (bm_george, bm_daniel, bm_lewis, …) |

> **Note:** Kokoro is primarily an English voice model. All voices will have an English accent when speaking other languages. For native-accent German TTS, a dedicated German TTS engine is recommended.

### Microphone troubleshooting

If the agent does not react to your voice, check the VAD confidence in the terminal output:

```
[VAD] conf=0.012 rms=0.0031
```

If `conf` stays near zero when you speak:

1. **Boost the microphone** in PulseAudio:
   ```bash
   # List sources
   pactl list sources short
   # Set your mic as default and boost volume
   pactl set-default-source <source-name>
   pactl set-source-volume <source-name> 150%
   ```
2. **Increase gain** in `config.yaml`:
   ```yaml
   audio:
     gain: 3.0
   ```
3. **Lower the VAD threshold** in `config.yaml`:
   ```yaml
   vad:
     threshold: 0.15
   ```

---

## Architecture

```
┌─────────────────┐
│  sounddevice    │  PortAudio callback — writes 512-sample chunks to ring buffer
│  InputStream    │
└────────┬────────┘
         │
┌────────▼────────┐
│   VAD thread    │  Silero VAD on every 32 ms chunk
│                 │  Speech start/end detection (0.4 s silence = end of utterance)
│                 │  Barge-in: sets interrupt_event if agent is speaking
└────────┬────────┘
         │ speech_queue (complete utterances)
┌────────▼────────┐
│ Pipeline thread │  1. FasterWhisper transcription
│                 │  2. vLLM streaming inference
│                 │  3. Punctuation-based chunking
│                 │  4. Kokoro TTS per chunk
└────────┬────────┘
         │ tts_audio_queue (float32 audio chunks)
┌────────▼────────┐
│ Playback thread │  sounddevice playback
│                 │  Polls interrupt_event every 50 ms → stops on barge-in
└─────────────────┘
```

### Barge-in flow

1. VAD detects speech while `is_agent_speaking` is set
2. `interrupt_event.set()`
3. Playback thread: calls `sd.stop()`, drains `tts_audio_queue`
4. Pipeline thread: breaks out of the vLLM streaming loop
5. New utterance is processed from scratch

---

## Project structure

```
local-voice-agent/
├── config.yaml          # all user-facing settings
├── main.py              # pipeline implementation
├── download_models.sh   # one-time model download script
├── start_vllm.sh        # launches vLLM in Docker
├── requirements.txt     # Python dependencies
└── models/              # downloaded model files (git-ignored)
    ├── models--Qwen--Qwen2.5-14B-Instruct-AWQ/
    ├── models--Systran--faster-whisper-small/
    └── kokoro/
        ├── kokoro-v1.0.int8.onnx
        └── voices-v1.0.bin
```

---

## Requirements

- Python 3.10+
- Docker with NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- NVIDIA GPU with ≥12 GB VRAM (tested on RTX 4080)
- PulseAudio or PipeWire for microphone input
- ~12 GB free disk space for models
