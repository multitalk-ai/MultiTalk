# Dialogue Script Synthesis

Two-stage LLM pipeline that synthesizes multi-party spoken-dialogue **scripts**. 
Each script is a JSON record carrying participants, scenario, and a
turn-level dialogue with TTS voice annotations (emotion / intensity / speed / volume /
dialect). The output of this stage is the input of stage 2 (TTS rendering) and stage
3 (word alignment).

## Pipeline

```
seed_sentence (emotion)
  └─► sample_scenario()        # dimension config (num_users, language, turn_count, ...)
        └─► Stage 1 prompt     # LLM → scenario + cast (named characters, role cards)
              └─► Stage 2 prompt   # LLM → spoken dialogue with per-turn voice tags
                    └─► quality_filter()   # 4 layers: format / length / roles / content
                          └─► train / val / test split (98 / 1 / 1)
```


## Files

| File | Purpose |
|---|---|
| `generate_dialogues.py` | Main entry. Orchestrates sample → generate → filter → split. |
| `scenario_config.py` | Dimension definitions (num_users, language, turn_count, AI involvement, industry × topic for work-meeting mode) and `ScenarioConfig` dataclass. |
| `prompt_builder.py` | Stage-1 / Stage-2 system + user prompt templates (default / role / work_meeting / ai_assistant variants). |
| `api_caller.py` | Two-stage LLM caller with concurrency, retry, resume-from-output, and three backends (Anthropic, vLLM, Claude Code CLI). |
| `quality_filter.py` | Heuristic filters: format / length / roles / content. |
| `gemma4_list.py` | Optional list of vLLM endpoints for load-balanced local inference. Replace placeholders with your own. |

## Install

```bash
pip install anthropic openai pandas tqdm pyarrow httpx
```

## Seed data

1. **Emotion seeds** — download `dair-ai/emotion` from Hugging Face:

   ```bash
   mkdir -p seed_data
   # Download the train and unsplit parquet files into seed_data/
   # https://huggingface.co/datasets/dair-ai/emotion
   # Expected: seed_data/emotion_train.parquet, seed_data/emotion_unsplit.parquet
   # (Or pass any parquet path(s) via --data_path.)
   ```

2. **(Optional) predefined assistant roles** — a JSONL file where each line is
   `{"id": "...", "role": {"name": "...", "summary": "...", ...}}`. Used with
   `--role_path`.

## Quick start

```bash
# Default mode — Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...
python generate_dialogues.py \
    --output_dir ./output \
    --model claude-sonnet-4-6 \
    --max_workers 8 \
    --num_per_seed 1

# Preview a single sample without writing batch outputs
python generate_dialogues.py --output_dir ./output --preview

# Filter only (skip generation; reuses output/raw_dialogues.jsonl)
python generate_dialogues.py --output_dir ./output --filter_only
```

### Modes

Each mode adds a directory suffix and rewires Stage-1 / Stage-2 system prompts.

| Flag | Suffix | Behavior |
|---|---|---|
| _(none)_ | `output/` | Default; assistant is a normal human in the group. |
| `--role_path PATH` | `_role` | Assistant uses a predefined role card from the JSONL. |
| `--seed-industry` | `_work_meeting` | Industry × topic round-robin; professional meeting register. |
| `--ai-assistant` | `_ai_assistant` | Assistant is a real AI voice assistant (Siri/Alexa-style). Mutually exclusive with `--role_path` and `--seed-industry`. |

### Backends

| Flag | Suffix | Notes |
|---|---|---|
| _(none)_ | — | Anthropic API (`messages.stream`). Reads `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`. |
| `--gemma4 --api_base URL` | `_gemma4` | Single vLLM OpenAI-compatible endpoint. Sampling fixed to `temperature=1.0, top_p=0.95, top_k=64`. |
| `--gemma4 --gemma4_list` | `_gemma4` | Load-balance across endpoints in `gemma4_list.py`. |
| `--claude-code` | `_cc` | Use the local `claude -p` CLI (subprocess). No `--temperature` / `max_tokens`. |

## Outputs

Inside `output_dir/`:

- `generation_config.json` — full hyperparameter snapshot (model, temperature, workers, mode flags).
- `raw_dialogues.jsonl` — every successfully generated record (one JSON per line).
- `filtered_dialogues.jsonl` — records that passed all quality filters.
- `rejected_dialogues.jsonl` — failed records, each with a `_reject_reasons` list.
- `train.jsonl` / `val.jsonl` / `test.jsonl` — 98 / 1 / 1 split of the filtered set.

A single record contains:

```jsonc
{
  "request_id": "...",
  "config": { "seed_sentence": ..., "num_users": ..., "language": ..., ... },
  "design": { "scenario_type": ..., "dialogue_style": ..., "relationship_type": ..., ... },
  "scenario": "rich 4-6 sentence paragraph",
  "participants": [
    { "name": ..., "role": "user"|"assistant", "gender": "male"|"female",
      "summary": ..., "personality_traits": ..., "roleplay_guidelines": ... }
  ],
  "dialogue": [
    { "speaker": "<name>", "text": "...",
      "voice": { "emotion": "calm", "emotion_intensity": 0.3,
                 "speed": "normal", "volume": "normal", "dialect": "standard_mandarin" } },
    { "speaker": "silence", "text": "[2s]" },
    ...
  ],
  "usage": { "input_tokens": ..., "output_tokens": ..., "cost_usd": ... }
}
```

## Resume / safety

- `call_two_stage_batch` reads `raw_dialogues.jsonl` on startup and skips any
  `seed_sentence` already present, so re-running the same command resumes naturally.
- `--max_cost USD` aborts the run when the cumulative `cost_usd` exceeds the budget.
- `--max_samples N` truncates the config list before generation (useful for ablations).

