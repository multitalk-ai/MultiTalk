#!/usr/bin/env bash
# Thin wrapper around eval.py. All knobs are read from env vars; extra CLI args
# are forwarded. Activate your Python env (conda / venv / etc.) before calling.
#
# Required env (or pass equivalent CLI flags):
#   MULTITALKBENCH_DATA       local clone of MultiTalk/MultiTalkBench dataset
#   CKPT_DIR                  model checkpoint dir (passed as --ckpt-dir)
#   INFERENCE_SCRIPT          adapter under ./models/, e.g. models/moshi_multitalk.py
#
# Optional env:
#   USE_DETAILED_SYS=1|0      default 1 (persona-on)
#   STAGES                    comma-list, default "1,2"
#   MODEL_NAME                override results/<MODEL_NAME>/
#   POLL_SEC                  watcher poll interval (default 10s)
#   EVAL_WORKERS              concurrent samples in stage 2 (default 16)
#   RANK / WORLD_SIZE         multi-rank sharding (default 0/1)
#   MULTITALKBENCH_RESULTS         results root (default ./results)
#   MULTITALKBENCH_JUDGE_ENDPOINTS comma-separated judge URLs (or use endpoints.json)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CKPT_DIR:?CKPT_DIR required (model checkpoint to evaluate)}"
: "${INFERENCE_SCRIPT:?INFERENCE_SCRIPT required (e.g. models/moshi_multitalk.py)}"
: "${USE_DETAILED_SYS:=1}"
: "${STAGES:=1,2}"
: "${POLL_SEC:=10}"
: "${EVAL_WORKERS:=16}"

detailed_flag=()
[[ "$USE_DETAILED_SYS" == "1" ]] && detailed_flag+=(--use-detailed-sys) || detailed_flag+=(--no-detailed-sys)

force_flag=();   [[ "${FORCE:-0}"        == "1" ]] && force_flag+=(--force)
rescore_flag=(); [[ "${RESCORE_ONLY:-0}" == "1" ]] && rescore_flag+=(--rescore-only)
resume_flag=();  [[ "${RESUME:-0}"       == "1" ]] && resume_flag+=(--resume)

exec python -u "$HERE/eval.py" \
  --ckpt-dir         "$CKPT_DIR" \
  --inference-script "$INFERENCE_SCRIPT" \
  --stages           "$STAGES" \
  --poll-sec         "$POLL_SEC" \
  --eval-workers     "$EVAL_WORKERS" \
  ${MODEL_NAME:+--model-name "$MODEL_NAME"} \
  ${RUN_TS:+--run-ts "$RUN_TS"} \
  "${detailed_flag[@]}" \
  "${force_flag[@]}" \
  "${rescore_flag[@]}" \
  "${resume_flag[@]}" \
  "$@"
