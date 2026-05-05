# MultiTalkBench

**MultiTalkBench is the first benchmark to jointly evaluate long, multi-party, and bilingual full-duplex dialogue.** It tests speech-to-speech systems on:

- **(a) Long interactions** — conversations longer than ten minutes, with explicit probes for long-range entity tracking and topic coherence.
- **(b) One-model-many-user multi-party interaction** — quantitative addressee-selection and turn-taking metrics.
- **(c) Chinese–English bilingual ability.**

To our knowledge, no prior benchmark addresses these axes jointly in a fully interactive, end-to-end speech setting.

The dataset (audio + alignments + persona prompts) is distributed via 🤗 Hugging Face: <https://huggingface.co/datasets/MultiTalk/MultiTalkBench>. This repository contains only the evaluation and inference code — see § *Quick start* below.


---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/multitalk-ai/MultiTalk.git
cd MultiTalk/MultiTalkBench
pip install -r requirements.txt
```

Model adapters have additional dependencies — see `requirements.txt` and the docstrings of `models/<adapter>.py` for what to install per model.

### 2. Download the dataset

```bash
huggingface-cli download MultiTalk/MultiTalkBench \
    --repo-type dataset --local-dir ./data
```

`./data/` will then contain `audio/`, `alignments/`, `personas/`, `meetings/`, `metadata.jsonl`, etc. (see the dataset card for the schema). Point the pipeline at it via either:

```bash
export MULTITALKBENCH_DATA=/abs/path/to/data
```

or pass `--data-root /abs/path/to/data` on every command.

### 3. Download model checkpoints

| Model | Source | Notes |
|---|---|---|
| Moshiko 7B | `kyutai/moshiko-pytorch-bf16` (HF) | `huggingface-cli download kyutai/moshiko-pytorch-bf16 --local-dir <ckpt>` |
| PersonaPlex 7B | `nvidia/personaplex-7b-v1` (HF) | needs PersonaPlex's bundled moshi clone for `moshi.offline` |
| MiniCPM-o 4.5 | `openbmb/MiniCPM-o-4_5` (HF) | duplex / streaming model |
| Qwen3-Omni 30B-A3B-Instruct | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (HF) | needs `qwen-omni-utils` |

Moshi adapters also need a clone of the [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) repo on disk, since they shell out to its `inference/offline.py`.

### 4. Bring up a judge LLM endpoint

The benchmark scores each sample with **6 calls** to a Gemma-4-it judge served behind any OpenAI-compatible endpoint (e.g. vLLM):

```bash
vllm serve google/gemma-4-31B-it --port 8000
# repeat with --port 8001, 8002, … for parallelism
```

Then either:

```bash
cp endpoints.example.json endpoints.json
$EDITOR endpoints.json    # paste your URLs (omit /chat/completions)
```

or:

```bash
export MULTITALKBENCH_JUDGE_ENDPOINTS="http://host:8000/v1,http://host:8001/v1"
```

`endpoints.json` is `.gitignore`d. The endpoint **must support write** — a read-only mirror won't accept POSTs.

### 5. Run inference + evaluation

```bash
export MULTITALKBENCH_DATA=/abs/path/to/data
export MULTITALKBENCH_RESULTS=./results        # optional, default ./results

CKPT_DIR=/abs/path/to/your/moshi/ckpt \
INFERENCE_SCRIPT=models/moshi_multitalk.py \
USE_DETAILED_SYS=1 \
bash eval.sh \
    --moshi-repo /abs/path/to/moshi-finetune \
    --tokenizer  /abs/path/to/tokenizer_spm_61k_bi.model
```

`eval.sh` forwards extra args (`--moshi-repo`, `--tokenizer`, `--ref-audio`, …) verbatim to the model adapter, so each adapter's model-specific flags reach it without `eval.py` having to know about them.

`eval.sh` orchestrates three stages — `1` (model inference), `2` (Gemma judge), `3` (cross-model report). `STAGES=1,2` is the default. Pass `STAGES=2` to score a pre-existing offline output dir without re-running inference.

Per-sample results land at:
```
results/<model_name>/run_<ts>/full/samples/<lang>/<meeting>/<target_speaker>.json
results/<model_name>/run_<ts>/full/{summary.json,samples.csv,meetings.csv}
```

### 6. Aggregate across models

```bash
python report_results.py        # writes results/model_report_<ts>/report.md
```

---

## Local layout

After running the steps above your tree should look like:

```
MultiTalk/
└── MultiTalkBench/                 # this repo
    ├── data/                       # cloned HF dataset (gitignored)
    │   ├── audio/{en,zh}/<sample_id>.flac
    │   ├── alignments/{en,zh}/<sample_id>.json
    │   ├── personas/{en,zh}/<sample_id>.txt
    │   ├── meetings/{en,zh}/<meeting_id>/{profile.json,transcript_context.txt}
    │   └── metadata.jsonl
    ├── results/                    # run outputs (gitignored)
    │   └── <model_name>/run_<ts>/full/...
    ├── endpoints.json              # judge URLs (gitignored)
    ├── config.py                   # paths/env resolution
    ├── data.py                     # metadata.jsonl reader
    ├── benchmark.py                # judge pipeline
    ├── eval.py / eval.sh           # orchestrator + wrapper
    ├── prompt_builder.py           # 28-dim rubric → judge prompts
    ├── report_results.py           # cross-model aggregation
    ├── rescore_all.py              # rescore in place under new weights
    ├── run_benchmark_test.py       # smoke + full integration runner
    ├── judge/{__init__.py,gemma4.py}
    └── models/                     # model inference adapters
        ├── adapter.py              # shared CLI / sample iterator
        ├── moshiko.py              # kyutai/moshiko-pytorch-bf16
        ├── moshi_multitalk.py      # MultiTalk-tuned Moshi 7B
        ├── personaplex.py          # nvidia/personaplex-7b-v1
        ├── minicpm_o45.py          # openbmb/MiniCPM-o-4_5
        └── qwen3_omni.py           # Qwen/Qwen3-Omni-30B-A3B-Instruct
```

`results/` and `data/` should be `.gitignore`d — they're not part of the repo.

---

## Adapter contract

Every script in `models/` accepts the same baseline CLI:

```bash
python models/<adapter>.py \
    --data-root  ./data \
    --ckpt-dir   /abs/path/to/checkpoint \
    --output-dir ./results/<model_name>/offline_output[_detailed_sys] \
    [--use-detailed-sys] \
    [--rank N --world-size W] \
    [--limit K] [--force]
```

Each adapter writes one JSON per sample to `<output-dir>/<lang>/<sample_id>.json` matching `benchmark.load_offline_utterances`'s schema. Sharding is fully external: every rank shards by `index % world_size == rank` over the canonical `metadata.jsonl` order.

To add a new model, copy `models/personaplex.py` (smallest reference) or `models/minicpm_o45.py` (full streaming example) and reuse `from . import adapter`.

---

## Configuration cheat sheet

| Env var | CLI flag | Purpose |
|---|---|---|
| `MULTITALKBENCH_DATA` | `--data-root` | HF dataset clone (required) |
| `MULTITALKBENCH_RESULTS` | `--results-root` | Output root (default `./results`) |
| `MULTITALKBENCH_JUDGE_ENDPOINTS` | `--judge-endpoints` | Comma-separated Gemma URLs (or use `endpoints.json`) |
| `MULTITALKBENCH_JUDGE_MODEL` | — | Model name in OpenAI requests (default `google/gemma-4-31B-it`) |
| `MULTITALKBENCH_JUDGE_API_KEY` | — | Bearer token (default `EMPTY`, accepted by vLLM) |
| `RANK` / `WORLD_SIZE` | `--rank` / `--world-size` | Multi-rank sharding (used by both `eval.py` and adapters) |

---

## License

Code in this repository (the evaluation harness, model adapters, judge
client, and data-engine scripts) is released under **Apache-2.0** —
see [`LICENSE`](../LICENSE).

The accompanying scoring scripts, persona prompts, segmentation indices,
and judge templates are likewise released under Apache-2.0. The
dataset itself (audio + alignments hosted at
[`MultiTalk/MultiTalkBench`](https://huggingface.co/datasets/MultiTalk/MultiTalkBench))
is distributed under **CC-BY-SA-4.0**.
