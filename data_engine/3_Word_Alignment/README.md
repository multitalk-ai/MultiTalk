# Word Alignment

Upgrades the sentence-level transcript manifest produced by stage 2
(`2_TTS_Rendering`) to **word-level alignment** using the
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (MFA).
Each assistant turn gains per-word `(word, start, end)` timestamps that
downstream training uses for streaming-aware speech modelling.

## Pipeline

```
manifest.jsonl  (stage 2 output)
    └─► align_duplex.py          # for each record:
                                 #   read audio + transcript,
                                 #   group segments by language,
                                 #   batch through MFAAligner (one subprocess per batch),
                                 #   inject per-word timestamps back into the JSON
                                 # writes manifest_word_aligned.jsonl
```

## Files

| File | Purpose |
|---|---|
| `align_duplex.py` | Driver. Streams the input manifest in chunks of `line_batch` lines, parses each record into `(audio_path, text_info, segments)`, groups segments by detected language, and dispatches batches of `batch_size` segments to a pool of `jobs` worker processes. Each worker reuses one `MFAAligner` instance. Writes word-aligned records back to a sibling JSONL. Resumable via a per-rank `processed_ids.txt` cache. |
| `align_duplex.sh` | Launcher that wires up distributed environment (`RANK` / `WORLD_SIZE` / NCCL knobs) and shells out to `align_duplex.py`. |
| `transcript_mfa_aligner.py` | Self-contained MFA wrapper (`MFAAligner` class). Handles language detection, audio resampling, text → MFA `.lab` conversion, per-process isolated `MFA_ROOT_DIR`, TextGrid parsing, and exposes `align_audio_segments_batch()` for batched alignment. |

## Install

```bash
# Montreal Forced Aligner (3.3.9 is what this code is tested against)
conda install -c conda-forge montreal-forced-aligner=3.3.9
pip install textgrid soundfile numpy tqdm

# Pre-download the acoustic + dictionary models you need (per language).
mfa model download acoustic   mandarin_mfa
mfa model download dictionary mandarin_mfa
mfa model download acoustic   english_mfa
mfa model download dictionary english_mfa
# Optional extras (used when the transcript language is detected as ja / ko):
mfa model download acoustic   japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download acoustic   korean_mfa
mfa model download dictionary korean_mfa
```

After download, the pretrained models live under
`<MFA_ROOT>/pretrained_models/`. Point `--mfa-root` (or `MFA_ROOT_DIR`) at
`<MFA_ROOT>` (default: `~/Documents/MFA`).

## Inputs

`align_duplex.py` accepts two record shapes — it auto-detects per line:

1. **`metadata_file` format** (the stage 2 manifest):

   ```jsonc
   {
     "audio":    "<R[:2]>/<R>_stereo_compressed.flac",
     "metadata": "<R[:2]>/<R>_stereo_compressed.json",
     ...
   }
   ```

   The sidecar JSON must contain a `metadata:transcript` array of
   `{start, end, transcript}` segments (the `metadata:transcript` field
   written by stage 2's `tts.py`).

2. **`legacy` format**:

   ```jsonc
   {
     "audio": [
       {"path": [["MODEL", "<audio path>"], ...], ...},     // audio_info
       {"text": {"alignments": [["sentence", [s, e], "speaker"], ...]}}  // text_info
     ]
   }
   ```

   Each existing `[text, [start, end], speaker]` triple becomes the segment
   to re-align at the word level.

Records that fail to parse are skipped with a warning; everything else is
batched and processed.

## Quick start

```bash
# Single-host run.
export INPUT_PATH=./tts_output/durations_compressed.jsonl   # stage 2 manifest
export MFA_ROOT=/path/to/MFA                                # contains pretrained_models/

python align_duplex.py "$INPUT_PATH" \
    --mfa-root "$MFA_ROOT" \
    --jobs       16 \
    --batch-size 32 \
    --line-batch 5000

# Multi-rank — each rank processes a disjoint shard of input lines.
# The launcher forwards RANK / WORLD_SIZE from torchrun / SLURM and exports
# the NCCL / distributed knobs the cluster needs.
INPUT_PATH=./tts_output/durations_compressed.jsonl \
MFA_ROOT=/path/to/MFA \
JOBS=16 BATCH_SIZE=32 LINE_BATCH=5000 \
    bash align_duplex.sh
```

## CLI / environment

| Flag / env | Default | Purpose |
|---|---|---|
| `input` (positional) / `INPUT_PATH` | — | Manifest JSONL produced by stage 2. |
| `-o, --output` | `<input>_word_aligned[_rank<R>].jsonl` | Output manifest path. |
| `-j, --jobs` / `JOBS` | 16 | Worker processes; each holds one `MFAAligner`. |
| `-b, --batch-size` / `BATCH_SIZE` | 32 | Segments packed into one MFA subprocess call. Larger = better throughput, more RAM. |
| `-l, --line-batch` / `LINE_BATCH` | 500 (CLI) / 5000 (shell wrapper) | Lines streamed in per outer iteration; caps memory peak. |
| `--rank` / `RANK` | env or 0 | This rank's id; rank `r` processes lines where `idx % world_size == r`. |
| `--world-size` / `WORLD_SIZE` | env or 1 | Total ranks. |
| `--mfa-root` / `MFA_ROOT_DIR` | `~/Documents/MFA` | MFA install root containing `pretrained_models/`. |
| `--no-resume` | off | Disable the `processed_ids.txt` resume cache for this run. |
| `--max-tasks` | unlimited | Stop after this many records (useful for smoke tests). |

## Outputs

For each input record, the same JSON is written back with word-level timing
added:

- **`metadata_file` records** — each `metadata:transcript[i]` segment gains
  a `"words"` list:
  ```jsonc
  {
    "start": 0.0,
    "end":   1.42,
    "transcript": "...",
    "words": [
      {"word": "hello", "start": 0.05, "end": 0.32},
      {"word": "world", "start": 0.36, "end": 0.78},
      ...
    ]
  }
  ```

- **`legacy` records** — `text.alignments` is replaced by a flat list of
  `[word, [start, end], speaker]` triples, one per word.

Auxiliary files written next to the input:

- `<input>_word_aligned.jsonl` (or `..._rank<R>.jsonl` under multi-rank)
- `.cache_<input_stem>/processed_ids.txt` (resume cache, per rank)

## Notes

- **Language bucketing.** Segments inside the same MFA batch must share an
  acoustic model, so `align_duplex.py` runs `detect_language()` per segment
  and groups by language before dispatching. Mixing zh/en in one batch would
  silently misalign — this is enforced by construction.
- **Per-process MFA root.** Each batch call materialises a temp
  `MFA_ROOT_DIR` with a symlinked `pretrained_models/`, so concurrent worker
  processes never write into a shared MFA state directory.
- **Resume.** Sample IDs are appended to the rank-local
  `processed_ids.txt` after each successful batch flush; rerunning the same
  command picks up where it stopped. Pass `--no-resume` to start fresh.
- **Same-audio batching.** When multiple segments inside a batch reference
  the same audio file, the file is read and resampled once and reused — this
  alone removes most of the I/O cost when the input has many short turns
  per audio.
- **Failure handling.** A record where MFA returns no words for some
  segments retains those segments unchanged; alignment failures are logged
  and counted but never abort the batch.
