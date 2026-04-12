# Audio Censoring Tool (ACT)

Live audio pipeline: capture from a microphone, run **faster-whisper** on overlapping chunks with word-level timestamps, replace dictionary hits with a short tone, and play out through a chosen device on a **fixed delay** (default 4 seconds). That delay gives ASR time to finish before the censored audio is heard—useful for streaming through a virtual sink into OBS or similar.

## Prerequisites

- **Python 3.10+** (tested in development with 3.12+).
- **ffmpeg** on your PATH (recommended for faster-whisper tooling and general audio work).
- A microphone and an output device (physical speakers or a virtual sink).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the Whisper weights for the model you select (default `tiny`).

## Linux: virtual sink (PipeWire / PulseAudio)

Create a null sink so OBS (or another app) can capture censored audio without tying up your real speakers:

```bash
pactl load-module module-null-sink sink_name=act_output
```

List devices and pick the sink that matches `act_output` (often exposed via PipeWire):

```bash
python -m act --list-devices
python -m act --output-device <N>   # add --input-device if needed
```

Unload the module when finished (id from `pactl list short modules`):

```bash
pactl unload-module <module_id>
```

## Windows

Install a virtual cable driver (for example **VB-Audio Virtual Cable**), then choose its playback device as `--output-device` and its corresponding input in OBS.

## Usage

```bash
python -m act --help
```

Useful options:

| Option | Meaning |
|--------|---------|
| `--list-devices` | Print input/output devices and exit |
| `--input-device N` / `--output-device N` | Device indices from the list |
| `--delay SEC` | Playback delay (default `4`) |
| `--asr-window SEC` | Length of each Whisper chunk (default `3`) |
| `--model SIZE` | e.g. `tiny`, `base` |
| `--word-list PATH` | One word/phrase per line (`#` comments allowed) |
| `--compute-type TYPE` | e.g. `int8` on CPU (default), `float16` on GPU |
| `--device cpu` / `cuda` | faster-whisper inference device |

Default word list: [data/default_wordlist.txt](data/default_wordlist.txt) (edit or replace).

Stop with **Ctrl+C**.

## Limitations (MVP)

- **English** is forced for transcription (`language="en"`); extend the engine if you need auto-detect or other languages.
- **Word boundaries** on `tiny` can be sloppy; larger models are more accurate but slower—keep `--delay` above worst-case ASR latency.
- **CPU vs GPU**: same stack; use `--device cuda` and an appropriate `--compute-type` when CUDA is available.
- Matches are **literal** after normalization (case and simple punctuation stripping), not stemmers or phonetic fuzzy match.
- Words split across two ASR windows may be missed; increase `--asr-window` or add overlap logic later if needed.

## Project layout

- `act/` — package (`buffer`, `engine`, `cli`, …)
- `data/default_wordlist.txt` — starter list
- `requirements.txt` — Python dependencies

## License

See [LICENSE](LICENSE).
