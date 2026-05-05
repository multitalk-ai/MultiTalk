"""Shared CLI + sample iterator used by every model adapter.

Adapters import this and supply a ``run_one(sample, system_prompt, output_path, args)``
callback. The helper handles argparse, dataset loading, RANK/WORLD_SIZE
sharding, and skip-on-exists logic.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Make `import config`, `import data` work when running from anywhere.
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

import config  # noqa: E402
import data as dataset  # noqa: E402


def add_standard_args(parser: argparse.ArgumentParser) -> None:
    """Add the CLI flags every adapter must accept."""
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Local clone of MultiTalk/MultiTalkBench dataset "
                             "(falls back to $MULTITALKBENCH_DATA).")
    parser.add_argument("--ckpt-dir", type=Path, required=True,
                        help="Model checkpoint directory.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Per-sample offline outputs go here as <lang>/<sample_id>.json.")
    parser.add_argument("--use-detailed-sys", action="store_true",
                        help="Use the persona prompt; otherwise use the minimal baseline prompt.")
    parser.add_argument("--rank", type=int, default=None,
                        help="Rank for sharding (default: $RANK or 0).")
    parser.add_argument("--world-size", type=int, default=None,
                        help="World size for sharding (default: $WORLD_SIZE or 1).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many samples on this rank (0 = all).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if the output JSON already exists.")


def _resolve_rank_world(args: argparse.Namespace) -> tuple[int, int]:
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", 0))
    world = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", 1))
    return rank, world


def output_path_for(args: argparse.Namespace, sample: dataset.Sample) -> Path:
    """Canonical per-sample output path. Aligns with benchmark.offline_path_for."""
    return args.output_dir / sample.language.value / f"{sample.target_speaker}.json"


def iter_samples_for_rank(args: argparse.Namespace) -> Iterable[dataset.Sample]:
    """Yield samples assigned to this rank, in dataset order, honoring --limit."""
    root = config.data_root(args.data_root)
    samples = dataset.load_samples(root)
    rank, world = _resolve_rank_world(args)
    yielded = 0
    for index, sample in enumerate(samples):
        if index % world != rank:
            continue
        if args.limit and yielded >= args.limit:
            return
        yield sample
        yielded += 1


def run_loop(
    args: argparse.Namespace,
    *,
    run_one: Callable[[dataset.Sample, str, Path, argparse.Namespace], None],
) -> int:
    """Drive ``run_one`` over the samples assigned to this rank.

    ``run_one`` writes ``<output-dir>/<lang>/<sample_id>.json`` (and any sibling
    audio file). Adapters need not handle sharding, skip-on-exists, or prompt
    resolution.
    """
    rank, world = _resolve_rank_world(args)
    print(f"[adapter rank={rank}/{world}] data_root={config.data_root(args.data_root)} "
          f"ckpt_dir={args.ckpt_dir} output_dir={args.output_dir} "
          f"use_detailed_sys={args.use_detailed_sys}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_done = n_skipped = n_failed = 0

    for sample in iter_samples_for_rank(args):
        out = output_path_for(args, sample)
        if out.exists() and not args.force:
            n_skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            sys_prompt = sample.system_prompt(detailed=args.use_detailed_sys)
        except Exception as exc:
            print(f"[adapter] {sample.sample_id}: failed to load persona ({exc})", flush=True)
            n_failed += 1
            continue

        try:
            run_one(sample, sys_prompt, out, args)
            n_done += 1
        except Exception as exc:
            print(f"[adapter] {sample.sample_id}: inference failed: {exc}", flush=True)
            n_failed += 1
            if args.force:
                # Don't bury cascading errors when the user is explicitly retrying.
                raise

    print(f"[adapter rank={rank}/{world}] done={n_done} skipped={n_skipped} "
          f"failed={n_failed}", flush=True)
    return 0 if n_failed == 0 else 2


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    add_standard_args(p)
    return p
