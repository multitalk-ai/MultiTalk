#!/usr/bin/env python3
"""Inference adapter for NVIDIA PersonaPlex 7B.

Shells out to ``moshi.offline`` from PersonaPlex's bundled moshi fork. Clone
the model card (``nvidia/personaplex-7b-v1``) and pass that directory as
``--ckpt-dir``. ``--moshi-repo`` should point at PersonaPlex's moshi clone
that exposes ``moshi.offline``.

Example::

    python models/personaplex.py \\
        --data-root  ./data \\
        --ckpt-dir   ~/models/personaplex-7b-v1 \\
        --output-dir ./results/personaplex/offline_output \\
        --moshi-repo ~/code/personaplex-moshi
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import adapter as _adapter


def parse_args():
    p = _adapter.common_parser(__doc__)
    p.add_argument("--moshi-repo", type=Path, required=True,
                   help="PersonaPlex moshi clone (provides `python -m moshi.offline`).")
    p.add_argument("--voice-prompt", type=str, default="NATF0.pt",
                   help="Filename inside <ckpt-dir>/voices/ to use as voice prompt.")
    return p.parse_args()


def _build_command(sample, system_prompt, output_path, args) -> list[str]:
    output_wav = output_path.with_suffix(".wav")
    return [
        "python", "-m", "moshi.offline",
        f"--moshi-weight={args.ckpt_dir / 'model.safetensors'}",
        f"--input-wav={sample.audio_path}",
        f"--output-wav={output_wav}",
        f"--output-text={output_path}",
        f"--text-prompt={system_prompt}",
        f"--voice-prompt={args.voice_prompt}",
        f"--voice-prompt-dir={args.ckpt_dir / 'voices'}",
        f"--tokenizer={args.ckpt_dir / 'tokenizer_spm_32k_3.model'}",
        f"--mimi-weight={args.ckpt_dir / 'tokenizer-e351c8d8-checkpoint125.safetensors'}",
        "--post-input-silence-seconds=30.0",
    ]


def run_one(sample, system_prompt, output_path, args):
    cmd = _build_command(sample, system_prompt, output_path, args)
    env = {**os.environ, "PYTHONPATH": str(args.moshi_repo)}
    rc = subprocess.call(cmd, cwd=str(args.moshi_repo), env=env)
    if rc != 0:
        raise RuntimeError(f"moshi.offline exited with code {rc} for {sample.sample_id}")


def main() -> int:
    args = parse_args()
    return _adapter.run_loop(args, run_one=run_one)


if __name__ == "__main__":
    raise SystemExit(main())
