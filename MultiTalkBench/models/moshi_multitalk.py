#!/usr/bin/env python3
"""Inference adapter for the MultiTalkBench-tuned Moshi 7B variant ("Moshi-MTB").

Like ``moshiko.py`` this shells out to moshi-finetune's ``inference/offline.py``,
but additionally:

  * loads weights from a *consolidated* checkpoint dir
    (``<ckpt-dir>/consolidated/{consolidated_single.safetensors,config.json}``)
    instead of a single ``model.safetensors``;
  * supports an optional voice-prompt pool to bias the generated voice toward
    the target speaker's gender (parsed out of the persona prompt). Set
    ``--voice-prompt-dir`` to a directory laid out as either::

        <voice-prompt-dir>/male/*.wav
        <voice-prompt-dir>/female/*.wav

    or (legacy Chinese)::

        <voice-prompt-dir>/男/*.wav
        <voice-prompt-dir>/女/*.wav

    Pass ``--no-voice-prompt`` to disable. Without ``--voice-prompt-dir`` the
    adapter calls offline.py with an empty voice prompt.

Example::

    python models/moshi_multitalk.py \\
        --data-root      ./data \\
        --ckpt-dir       ~/saves/moshi_multitalk/checkpoint_000200 \\
        --output-dir     ./results/moshi_multitalk/offline_output_detailed_sys \\
        --moshi-repo     ~/code/moshi-finetune \\
        --tokenizer      ~/code/moshi-finetune/tokenizer/tokenizer_spm_61k_bi.model \\
        --use-detailed-sys
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from . import adapter as _adapter


def parse_args():
    p = _adapter.common_parser(__doc__)
    p.add_argument("--moshi-repo", type=Path, required=True,
                   help="Path to a moshi-finetune clone.")
    p.add_argument("--tokenizer", type=Path, required=True,
                   help="SentencePiece tokenizer file (e.g. tokenizer_spm_61k_bi.model).")
    p.add_argument("--moshi-weight", type=Path, default=None,
                   help="Override path. Default: <ckpt-dir>/consolidated/consolidated_single.safetensors.")
    p.add_argument("--lm-config", type=Path, default=None,
                   help="Override path. Default: <ckpt-dir>/consolidated/config.json.")
    p.add_argument("--voice-prompt-dir", type=Path, default=None,
                   help="Voice prompt pool directory. Optional; see module docstring for layout.")
    p.add_argument("--no-voice-prompt", action="store_true",
                   help="Disable voice prompts entirely.")
    p.add_argument("--voice-prompt-gender", default="auto",
                   choices=["male", "female", "both", "auto"],
                   help="When --voice-prompt-dir is set, restrict by gender. "
                        "'auto' parses Male/Female (or 男性/女性) out of the persona.")
    p.add_argument("--context-len", type=int, default=None,
                   help="Override LM attention context in 12.5 Hz frames.")
    return p.parse_args()


# ───────────────────────── voice prompt selection ───────────────────────────

def _list_wavs(d: Optional[Path]) -> List[Path]:
    return sorted(d.glob("*.wav")) if d and d.exists() else []


def _voice_pool(voice_root: Path, gender: str) -> List[Path]:
    """Return wavs matching the requested gender, supporting EN/CN dir names."""
    if gender == "male":
        return _list_wavs(voice_root / "male") + _list_wavs(voice_root / "男")
    if gender == "female":
        return _list_wavs(voice_root / "female") + _list_wavs(voice_root / "女")
    return (
        _list_wavs(voice_root / "male") + _list_wavs(voice_root / "female")
        + _list_wavs(voice_root / "男") + _list_wavs(voice_root / "女")
    )


def _detect_gender(system_prompt: str) -> Optional[str]:
    if "女性" in system_prompt:
        return "female"
    if "男性" in system_prompt:
        return "male"
    if re.search(r"\bFemale\b", system_prompt, re.IGNORECASE):
        return "female"
    if re.search(r"\bMale\b", system_prompt, re.IGNORECASE):
        return "male"
    return None


def _pick_voice_prompt(args, sample, system_prompt: str) -> Optional[Path]:
    if args.no_voice_prompt or args.voice_prompt_dir is None:
        return None
    gender = args.voice_prompt_gender
    if gender == "auto":
        gender = _detect_gender(system_prompt) or "both"
    pool = _voice_pool(args.voice_prompt_dir, gender)
    if not pool:
        return None
    digest = hashlib.md5(str(sample.audio_path).encode("utf-8")).hexdigest()
    return pool[int(digest, 16) % len(pool)]


# ───────────────────────── command construction ─────────────────────────────

def _build_command(sample, system_prompt, output_path, args) -> list[str]:
    moshi_weight = args.moshi_weight or args.ckpt_dir / "consolidated" / "consolidated_single.safetensors"
    lm_config = args.lm_config or args.ckpt_dir / "consolidated" / "config.json"
    output_wav = output_path.with_suffix(".wav")

    voice_prompt = _pick_voice_prompt(args, sample, system_prompt)
    voice_prompt_dir = (
        str(voice_prompt.parent) if voice_prompt else "inference/voices"
    )
    voice_prompt_base = voice_prompt.name if voice_prompt else ""

    cmd = [
        "python", "./inference/offline.py",
        f"--moshi-weight={moshi_weight}",
        f"--tokenizer={args.tokenizer}",
        f"--lm-config={lm_config}",
        f"--input-wav={sample.audio_path}",
        f"--output-wav={output_wav}",
        f"--output-text={output_path}",
        f"--text-prompt={system_prompt}",
        f"--voice-prompt={voice_prompt_base}",
        f"--voice-prompt-dir={voice_prompt_dir}",
        "--post-input-silence-seconds=30.0",
    ]
    if args.context_len is not None:
        cmd.append(f"--context-len={args.context_len}")
    return cmd


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
