#!/usr/bin/env python3
"""Inference adapter for the public Kyutai Moshiko 7B model.

This is a thin shell wrapper around moshi-finetune's ``inference/offline.py``.
You must clone https://github.com/kyutai-labs/moshi-finetune (or your fork) and
point ``--moshi-repo`` at it. Model weights are typically downloaded with::

    huggingface-cli download kyutai/moshiko-pytorch-bf16 \\
        --local-dir <ckpt-dir>

Example::

    python models/moshiko.py \\
        --data-root  ./data \\
        --ckpt-dir   ~/models/moshiko-pytorch-bf16 \\
        --output-dir ./results/moshiko/offline_output \\
        --moshi-repo ~/code/moshi-finetune
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import adapter as _adapter


def parse_args():
    p = _adapter.common_parser(__doc__)
    p.add_argument("--moshi-repo", type=Path, required=True,
                   help="Path to a moshi-finetune clone (provides inference/offline.py).")
    p.add_argument("--moshi-weight", type=Path, default=None,
                   help="Override path to model.safetensors. Default: <ckpt-dir>/model.safetensors.")
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="Override path to the SentencePiece tokenizer file. "
                        "Default: <ckpt-dir>/tokenizer_spm_32k_3.model.")
    return p.parse_args()


def _build_command(sample, system_prompt: str, output_path: Path, args) -> list[str]:
    moshi_weight = args.moshi_weight or args.ckpt_dir / "model.safetensors"
    tokenizer = args.tokenizer or args.ckpt_dir / "tokenizer_spm_32k_3.model"
    output_wav = output_path.with_suffix(".wav")
    return [
        "python", "./inference/offline.py",
        f"--moshi-weight={moshi_weight}",
        f"--tokenizer={tokenizer}",
        f"--input-wav={sample.audio_path}",
        f"--output-wav={output_wav}",
        f"--output-text={output_path}",
        f"--text-prompt={system_prompt}",
        "--voice-prompt=",
        "--voice-prompt-dir=inference/voices",
        "--post-input-silence-seconds=30.0",
    ]


def run_one(sample, system_prompt, output_path, args):
    cmd = _build_command(sample, system_prompt, output_path, args)
    rc = subprocess.call(cmd, cwd=str(args.moshi_repo))
    if rc != 0:
        raise RuntimeError(f"moshi offline.py exited with code {rc} for {sample.sample_id}")


def main() -> int:
    args = parse_args()
    return _adapter.run_loop(args, run_one=run_one)


if __name__ == "__main__":
    raise SystemExit(main())
