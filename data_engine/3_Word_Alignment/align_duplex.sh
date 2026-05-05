#!/usr/bin/env bash
# Launch script for word-level alignment with Montreal Forced Aligner (MFA).
#
# Required environment:
#   INPUT_PATH       — manifest JSONL (output of stage 2 / `gen_duration_and_alignments.py`)
#   MFA_ROOT         — root of the MFA install that contains pretrained_models/
#                      (default ~/Documents/MFA when unset)
#   RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT — set by the launcher
#                      (e.g. torchrun) for distributed sharding
set -euo pipefail

# Activate the project's MFA conda env (edit the env name to match your install).
# eval "$(conda shell.bash hook)"
# conda activate mfa

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_TIMEOUT=60
export NCCL_IB_TIMEOUT=25
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_AR_THRESHOLD=0
export NCCL_IB_DISABLE=0
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-65535}

: "${INPUT_PATH:?set INPUT_PATH to the stage-2 manifest JSONL}"
: "${MFA_ROOT:?set MFA_ROOT to the MFA install root containing pretrained_models/}"

python "$(dirname "$0")/align_duplex.py" \
    "$INPUT_PATH" \
    --mfa-root "$MFA_ROOT" \
    --jobs       "${JOBS:-16}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --line-batch "${LINE_BATCH:-5000}"
