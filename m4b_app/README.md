# VibeVoice M4B Audiobook Converter

Converts EPUB and TXT books to M4B audiobooks using VibeVoice TTS. Runs as a
Tkinter GUI inside WSL (displayed via WSLg on Windows 11).

## Requirements

```bash
pip install ebooklib beautifulsoup4 lxml torch
# ffmpeg must be on PATH
sudo apt install ffmpeg
```

## Running

```bash
cd m4b_app
python app1.py
```

## UI Settings

| Setting | Description |
|---|---|
| Quality (steps) | DDPM diffusion steps: Fast=5, Balanced=10, Best=20. 10 is the sweet spot — 5 is noticeably rougher, 20 gives minimal improvement. |
| CFG Scale | Classifier-Free Guidance — how strongly the output adheres to the voice prompt. Default 1.5. Higher = more voice-like but can introduce artifacts. |
| Parallel chunks | Number of worker threads preparing chunks concurrently. GPU inference is serialized by a lock (shared scheduler state); workers overlap tokenization and voice-prompt cloning. Default 2. |
| Debug logging | Emits per-chunk timing: clone time, prep time, inference time, GPU memory. Use to profile bottlenecks. |

## File Layout

| File | Role |
|---|---|
| `app1.py` | Tkinter GUI. Runs converter directly in a background thread (no subprocess). |
| `converter.py` | Shared utilities: text parsing, chunking, ffmpeg M4B builder, `get_converter()` factory. |
| `converter_streaming.py` | `VibeVoiceStreamingConverter` — VibeVoice-Realtime-0.5B implementation. |
| `converter_full.py` | `VibeVoiceFullConverter` stub — 1.5B model not yet implemented. |
| `wslapp.py` | CLI entry point (headless use). Not used by the GUI. |

## Architecture Notes

**Voice prompt cloning** — The voice prompt (loaded `.pt` file) is cloned for
every chunk using a recursive `_clone_prompt()` that calls `tensor.clone()` to
keep tensors on GPU instead of doing a CPU `deepcopy`.

**Inference lock** — The DPM scheduler inside the model has mutable instance
state (`step_index`, `model_outputs`). A `threading.Lock` serializes
`model.generate()` calls; worker threads overlap on prep work only.

**Future: model pool** — Loading N model instances into a `queue.Queue` pool
would allow truly parallel inference (each instance has its own scheduler state).
At ~3 GB VRAM per instance a 3060 12 GB can fit 3 instances alongside the OS.

## Chapter Detection (TXT files)

Detects `Chapter X` / `Part X` headings. Books with number-word chapters
(`One.`, `Two.`) or other formats are parsed as a single chapter — audio
quality is unaffected, only the M4B chapter markers differ.
