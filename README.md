# MultiTalk

**MultiTalk is a research project on long, multi-party, bilingual full-duplex
spoken dialogue.** This monorepo contains both the evaluation benchmark used to
score speech-to-speech systems and the synthetic-data pipeline used to build
the training corpus.

| Directory | Purpose |
|---|---|
| [`MultiTalkBench/`](MultiTalkBench/) | Evaluation benchmark — Gemma-judged 6-call rubric, five model adapters (Moshiko / Moshi-MultiTalk / PersonaPlex / MiniCPM-o 4.5 / Qwen3-Omni), cross-model report. |
| [`data_engine/`](data_engine/) | Three-stage synthetic data pipeline: (1) LLM-driven dialogue script synthesis, (2) IndexTTS2 multi-channel rendering, (3) MFA word-level alignment. |

The released dataset (audio + alignments + persona prompts) is distributed
through HuggingFace:
<https://huggingface.co/datasets/MultiTalk/MultiTalkBench>.

## What's covered by the benchmark

MultiTalkBench is — to our knowledge — the first benchmark to jointly evaluate:

- **(a) Long interactions** (>10 min) with explicit probes for long-range entity tracking and topic coherence.
- **(b) One-model-many-user multi-party interaction** — quantitative addressee-selection and turn-taking metrics.
- **(c) Chinese–English bilingual ability.**

See [`MultiTalkBench/README.md`](MultiTalkBench/README.md) for the full
evaluation contract, adapter API, and configuration cheat sheet.

## Reproducing the data pipeline

The audio + alignments shipped on HF are produced end-to-end by the scripts
under [`data_engine/`](data_engine/). Each stage has its own README:

1. [`data_engine/1_Dialogue_Script_Synthesis/`](data_engine/1_Dialogue_Script_Synthesis/) — scenario-conditioned dialogue generation via Anthropic / vLLM-Gemma backends, with a 4-layer quality filter.
2. [`data_engine/2_TTS_Rendering/`](data_engine/2_TTS_Rendering/) — IndexTTS2 multi-channel rendering with overlap / barge-in / silence handling and per-speaker gap compression.
3. [`data_engine/3_Word_Alignment/`](data_engine/3_Word_Alignment/) — Montreal Forced Aligner word-level alignment (zh / en / ja / ko).

You don't need the data engine to run the benchmark — the benchmark consumes
the HF dataset directly.

## License

- **Code** (everything in this repository, including evaluation harness, model
  adapters, judge client, data-engine scripts, scoring scripts, persona
  prompts, segmentation indices, and judge templates) — Apache-2.0; see
  [`LICENSE`](LICENSE).
- **Dataset** (audio + alignments on HuggingFace) — CC-BY-SA-4.0.
