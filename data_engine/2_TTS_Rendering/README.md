#  TTS Rendering

Renders the dialogue scripts into multi-channel audio. Each speaker is placed on a dedicated channel; overlap,
interruption, and silence turns from the script are realised as inter-channel
timing. The audio is then post-processed (stereo down-mix, gap compression,
duration/alignment manifest) so it can feed stage 3 (word alignment) and
downstream training.


## Files

| File | Purpose |
|---|---|
| `tts.py` | Main TTS worker. Loads `IndexTTS2`, samples a same-gender speaker prompt per participant, runs inference per dialogue turn, mixes turns into a multichannel array (assistant on channel 0, others by appearance order), handles `overlap` / barge-in / silence turns, writes `.wav` + sidecar `.json`. Multiprocessing across GPUs; respects `RANK_V` / `WORLD_SIZE_V` for distributed sharding. |
| `convert_to_stereo.py` | Recursively converts every `.wav` / `.flac` under a directory to stereo FLAC (`*_stereo.flac`). Mono is duplicated; multichannel keeps channel 0 as left and averages channels 1+ as right. Skips files that already have a stereo or stereo-compressed sibling. |
| `compress_user_assistant_gaps.py` | Compresses long silence gaps between user and assistant turns inside the rendered audio, producing `*_compressed.flac` plus an updated metadata JSON. Atomic writes; resumable. |
| `check_stereo.py` | Sanity check — opens each audio file referenced by a manifest JSONL and verifies it has exactly the expected channel count. Exits non-zero if any file fails. |
| `collect_wav.py` | Scans dialogue-script JSONLs and emits a manifest of `{audio, metadata}` pairs that actually exist on disk under `DATASETS_DIR`, resolving the extension/compression variant (`.wav` → `.flac` → `_compressed.flac`). |
| `gen_duration_and_alignments.py` | Walks a wav-list manifest, validates the metadata, populates `alignments` from the per-word transcript if present, and writes a final manifest with `path / duration / voice / language / system`. |

## Install

The TTS step itself depends on **IndexTTS2** (an external project) for inference.
Install it separately and point `INDEX_TTS_DIR` at the install root that contains
`config.yaml` and the model weights.

```bash
# Audio I/O + utilities used across the stage
pip install soundfile librosa numpy tqdm orjson

# Optional: TTS-quality eval branch in tts.py (--enable-eval)
pip install openai-whisper torch
# SpeechMOS (UTMOS) checkpoint must be available locally; see SPEECHMOS_DIR below.
```

## Inputs

1. **Dialogue scripts** — JSONL produced by stage 1
   (`output*/raw_dialogues.jsonl` or `filtered_dialogues.jsonl`). Each record
   must contain `request_id`, `participants[*].name/role/gender`, `dialogue[*]`
   with `speaker / text / voice`, and `config.language`.

2. **Speaker-prompt audio** — directory of reference clips for IndexTTS2
   zero-shot voice cloning. Layout the renderer expects under
   `AUDIO_PROMPTS_DIR`:

   ```text
   audioprompts/
     ├── 男/                # zh male prompts (.wav)
     ├── 女/                # zh female prompts (.wav)
     └── audio_info_with_mos_text.json   # en prompts: list of {wavpath, ...}
                                         # paths classified by "/female/" or "/male/"
   ```

3. **IndexTTS2 checkpoints** — at `INDEX_TTS_DIR/config.yaml` plus the model
   weight files referenced from that config.

## Quick start

```bash
# 0) Tell each script where things live (or pass via CLI flags / argparse).
export INDEX_TTS_DIR=/path/to/IndexTTS2
export AUDIO_PROMPTS_DIR=./assets/audioprompts
export INPUT_PATH=./tts_output/dialogue_scripts.jsonl     # from stage 1
export OUTPUT_DIR=./tts_output/multichannel_wavs
export NUM_WORKERS=3                                      # GPU workers per rank

# 1) Render every dialogue to a multichannel wav + json sidecar.
#    Single host:
python tts.py
#    Multi-rank (e.g. via torchrun / SLURM): use the wrapper, which forwards
#    RANK / WORLD_SIZE into RANK_V / WORLD_SIZE_V.
bash tts_multiusers.sh

# 2) Down-mix every rendered .wav into stereo .flac (assistant L, users R-mix).
python convert_to_stereo.py "$OUTPUT_DIR"

# 3) (optional) compress long user↔assistant silences in-place.
DURATIONS_JSONL=./tts_output/durations_stereo.jsonl \
    python compress_user_assistant_gaps.py

# 4) Sanity check — every manifest entry is stereo.
python check_stereo.py --data-path ./tts_output/durations_stereo_compressed.jsonl

# 5) Collect rendered files into a {audio, metadata} manifest.
DATASETS_DIR="$OUTPUT_DIR" \
DIALOGUE_JSONLS="$INPUT_PATH" \
WAV_LIST_OUT=./tts_output/full_duplex_dialogues_wav_lst.jsonl \
    python collect_wav.py

# 6) Final manifest with durations, voices and word alignments.
INPUT_PATH=./tts_output/full_duplex_dialogues_wav_lst.jsonl \
OUTPUT_DIR="$OUTPUT_DIR" \
OUTPUT_JSONL=./tts_output/durations_compressed.jsonl \
    python gen_duration_and_alignments.py
```

## Environment variables

All scripts read these (each falls back to a sensible default when unset):

| Variable | Used by | Purpose |
|---|---|---|
| `INDEX_TTS_DIR` | `tts.py` | Root of the IndexTTS2 install (must contain `config.yaml`). |
| `AUDIO_PROMPTS_DIR` | `tts.py` | Speaker prompt directory (see *Inputs* §2). |
| `INPUT_PATH` | `tts.py`, `gen_duration_and_alignments.py` | Input JSONL. |
| `OUTPUT_DIR` | `tts.py`, `gen_duration_and_alignments.py` | Where rendered audio is written / read. |
| `NUM_WORKERS` | most scripts | Multiprocessing worker count. |
| `BATCH_SIZE` / `CHUNKSIZE` | `compress_user_assistant_gaps.py` | Tune per-batch executor dispatch. |
| `RANK_V` / `WORLD_SIZE_V` | `tts.py`, `gen_duration_and_alignments.py` | Distributed sharding (set by `tts_multiusers.sh`). |
| `ENABLE_EVAL` | `tts.py` | If `true`, also load Whisper + UTMOS for per-utterance ASR / MOS. |
| `WHISPER_CACHE` | `tts.py` (eval) | Whisper download cache. |
| `SPEECHMOS_DIR` | `tts.py` (eval) | Local SpeechMOS / UTMOS checkpoint dir (used as `torch.hub.load(..., source='local')`). |
| `DATASETS_DIR` | `collect_wav.py` | Root of the per-record audio tree (= `OUTPUT_DIR` in step 1). |
| `DIALOGUE_JSONLS` | `collect_wav.py` | Comma-separated input JSONL list. |
| `WAV_LIST_OUT` | `collect_wav.py` | Manifest output path. |
| `OUTPUT_JSONL` | `gen_duration_and_alignments.py` | Final duration manifest path. |
| `DURATIONS_JSONL` | `compress_user_assistant_gaps.py` | Manifest whose audio entries should be gap-compressed. |

## Outputs

For each input dialogue script (`request_id = R`):

- `OUTPUT_DIR/<R[:2]>/<R>.wav` — multichannel rendered audio
  (channel 0 = assistant, channels 1..N = users in order of appearance).
- `OUTPUT_DIR/<R[:2]>/<R>.json` — sidecar metadata:
  ```jsonc
  {
    "id": "...",
    "speaker_to_channel": {"<assistant_name>": 0, "<user_a>": 1, "...": 2},
    "duration": 87.3,
    "num_channels": 3,
    "voice": {"<name>": "<wav prompt path>", ...},
    "metadata:transcript": [
      {"start": 0.0, "end": 1.4, "transcript": "...", "words": [...]}
    ]
  }
  ```
- After stereo down-mix: `<R>_stereo.flac` (and after gap compression:
  `<R>_stereo_compressed.flac` plus matching `*_compressed.json`).
- After step 6: a manifest JSONL with one entry per accepted record:
  ```jsonc
  {
    "path":     "<R[:2]>/<R>_stereo_compressed.flac",
    "duration": 87.3,
    "voice":    {"<name>": "<wav prompt path>", ...},
    "system":   "You are a helpful, knowledgeable assistant. Respond in English."
  }
  ```

## Notes

- **Channel layout convention.** The assistant is *always* on channel 0; this is
  what `convert_to_stereo.py` relies on when collapsing to a 2-channel mix
  (assistant → left, mean of users → right).
- **Overlap and barge-in.** `tts.py:mix_to_multichannel_wav` reads
  `dialogue[i].overlap == True` to place adjacent turns on overlapping time
  windows, and reads trailing `——` on the previous turn to extend the
  barge-in distribution (50–200 ms on a normal switch, 500–1000 ms when the
  previous speaker trails off).
- **Silence turns.** Records with `speaker == "silence"` and text in `[Xs]`
  form are realised as global silence (no channel assignment). They consume
  the duration written in their text.
- **Resume.** Every script skips records whose output already exists on disk
  and does not rewrite finished outputs, so a job can be killed and restarted.
