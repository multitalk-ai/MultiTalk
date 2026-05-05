#!/usr/bin/env python3
"""End-to-end MultiTalkBench evaluation orchestrator.

Stages
------
  1) inference: each rank runs the configured model adapter (``--inference-script``),
     sharded by ``RANK``/``WORLD_SIZE``.
  2) benchmark: rank 0 streams evaluation in parallel with stage 1 — a watcher thread
     polls the offline_root, queues newly produced samples to a ``ThreadPoolExecutor``
     and reuses ``benchmark.run_or_resume_sample``.
  3) report: rank 0 calls ``report_results.main`` to aggregate across model runs.

Multi-rank coordination is via a small file barrier:
  ``$RESULTS_ROOT/$MODEL_NAME/.barrier/rank_<i>.done``

Direct invocation::

    RANK=0 WORLD_SIZE=1 python -u eval.py \\
        --ckpt-dir   <path> \\
        --inference  models/moshi_multitalk.py \\
        --data-root  ./data \\
        --stages     1,2

See ``eval.sh`` for a thin wrapper that picks defaults out of the environment.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Set

import benchmark
import config
import report_results
from prompt_builder import (
    Language,
    PromptBuilder,
    ROLE_CANONICAL,
    ScoreCalculator,
    merge_scores,
)


HERE = Path(__file__).resolve().parent


# ─────────────────────── argparse ────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MultiTalkBench end-to-end orchestrator.")
    p.add_argument("--ckpt-dir", type=Path, required=True,
                   help="Model checkpoint directory passed to the inference adapter.")
    p.add_argument("--inference-script", type=Path, required=True,
                   help="Path to a model adapter under ./models/, e.g. "
                        "models/moshi_multitalk.py. The adapter must accept "
                        "--data-root, --ckpt-dir, --output-dir, [--use-detailed-sys] "
                        "and write <output-dir>/<lang>/<sample_id>.json files.")
    p.add_argument("--model-name", default=None,
                   help="Override results/<model_name>/. Default: <ckpt parent name>_<ckpt name>[_detailed_sys].")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Local clone of the MultiTalk/MultiTalkBench dataset. "
                        "Falls back to $MULTITALKBENCH_DATA.")
    p.add_argument("--results-root", type=Path, default=None,
                   help="Output root for run artifacts. Default: ./results "
                        "(or $MULTITALKBENCH_RESULTS).")
    p.add_argument("--use-detailed-sys", dest="use_detailed_sys", action="store_true", default=True)
    p.add_argument("--no-detailed-sys", dest="use_detailed_sys", action="store_false")
    p.add_argument("--judge-endpoints", type=str, default=None,
                   help="Comma-separated judge LLM base URLs (overrides endpoints.json/env).")
    p.add_argument("--stages", default="1,2",
                   help="Comma-separated stage IDs, e.g. '1,2,3' or '2'.")
    p.add_argument("--poll-sec", type=float, default=10.0,
                   help="Seconds between watcher polls and progress lines.")
    p.add_argument("--eval-workers", type=int, default=16,
                   help="Concurrent samples evaluated by stage 2.")
    p.add_argument("--min-file-age", type=float, default=2.0,
                   help="Skip offline_*.json younger than this (seconds) to avoid mid-write reads.")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gap-seconds", type=float, default=1.2)
    p.add_argument("--max-transcript-chars", type=int, default=0)
    p.add_argument("--call-retries", type=int, default=2)
    p.add_argument("--retry-delay-seconds", type=float, default=2.0)
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate samples even if a completed result exists.")
    p.add_argument("--run-ts", default=None,
                   help="Run timestamp suffix; auto-generated if omitted.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse the run_dir pointed to by latest_run.txt instead of "
                        "creating a new run_<ts>/. Lets stage 2 skip samples already "
                        "evaluated. Ignored if --run-ts is set.")
    p.add_argument("--rescore-only", action="store_true",
                   help="Skip stages 1/2/3. Recompute score_summary in-place for "
                        "every completed sample using the current ScoreCalculator "
                        "(no judge calls), then rebuild aggregate outputs.")
    # Unknown args are forwarded to the inference script — adapters can take
    # model-specific flags (--moshi-repo, --tokenizer, --ref-audio, …) without
    # eval.py needing to know about them.
    args, args.inference_extras = p.parse_known_args(argv)
    args.data_root = config.data_root(args.data_root)
    args.results_root = config.results_root(args.results_root)
    return args


# ─────────────────────── helpers ─────────────────────────

def derive_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    base = f"{args.ckpt_dir.parent.parent.name}_{args.ckpt_dir.name}"
    return base + ("_detailed_sys" if args.use_detailed_sys else "")


def parse_stages(spec: str) -> Set[int]:
    return {int(x.strip()) for x in spec.replace(",", " ").split() if x.strip()}


def offline_root_for(args: argparse.Namespace) -> Path:
    """Where the model adapter writes per-sample offline outputs for this run."""
    sub = "offline_output_detailed_sys" if args.use_detailed_sys else "offline_output"
    return args.results_root / derive_model_name(args) / sub


def make_benchmark_ns(args: argparse.Namespace, run_dir: Path) -> argparse.Namespace:
    """Build a Namespace compatible with benchmark.run_or_resume_sample / discover_samples."""
    return argparse.Namespace(
        data_root=args.data_root,
        offline_root=offline_root_for(args),
        output_dir=run_dir / "full",
        limit=0,
        meetings="",
        dry_run=False,
        force=args.force,
        fail_fast=False,
        call_retries=args.call_retries,
        retry_delay_seconds=args.retry_delay_seconds,
        gap_seconds=args.gap_seconds,
        max_transcript_chars=args.max_transcript_chars,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


def log(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────── stage 1 ─────────────────────────

def run_stage1(args: argparse.Namespace, rank: int, world_size: int, capture_log: Optional[Path]) -> int:
    cmd = [
        sys.executable, "-u", str(args.inference_script),
        "--data-root", str(args.data_root),
        "--ckpt-dir", str(args.ckpt_dir),
        "--output-dir", str(offline_root_for(args)),
        "--rank", str(rank),
        "--world-size", str(world_size),
    ]
    if args.use_detailed_sys:
        cmd += ["--use-detailed-sys"]
    # Forward any extra args (e.g. --moshi-repo, --tokenizer, --ref-audio).
    cmd += list(getattr(args, "inference_extras", []))

    log(f"[stage 1 rank={rank}/{world_size}] {' '.join(cmd)}")
    if capture_log:
        log(f"[stage 1 rank={rank}] log → {capture_log}")
        capture_log.parent.mkdir(parents=True, exist_ok=True)
        with capture_log.open("w", buffering=1) as out:
            return subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)
    return subprocess.call(cmd)


# ─────────────────────── stage 2 ─────────────────────────

class StreamingEvaluator:
    def __init__(self, args: argparse.Namespace, run_dir: Path, barrier_dir: Path, world_size: int):
        self.args = args
        self.run_dir = run_dir
        self.barrier_dir = barrier_dir
        self.world_size = world_size
        self.ns = make_benchmark_ns(args, run_dir)
        self.builder = PromptBuilder()
        self.submitted: Dict[str, "concurrent.futures.Future"] = {}
        self.results: Dict[str, dict] = {}
        self.samples: Dict[str, "benchmark.Sample"] = {}
        self._lock = threading.Lock()
        self.start_ts = time.time()

    def _evaluate_one(self, sample) -> dict:
        try:
            return benchmark.run_or_resume_sample(sample, self.ns, builder=self.builder)
        except Exception as exc:
            return {
                "status": "failed",
                "dimension_schema_version": benchmark.SCHEMA_VERSION,
                "sample_id": sample.sample_id,
                "language": sample.language.value,
                "meeting_id": sample.meeting_id,
                "target_speaker": sample.target_speaker,
                "offline_json": str(sample.offline_json),
                "audio_path": str(sample.audio_path),
                "calls": {}, "scores": {}, "score_summary": {},
                "errors": [{"stage": "sample", "message": str(exc),
                            "traceback": traceback.format_exc()}],
            }

    def _discover_ready(self) -> List["benchmark.Sample"]:
        if not self.ns.offline_root.exists():
            return []
        try:
            samples = benchmark.discover_samples(self.ns)
        except Exception as exc:
            log(f"[stage 2] discover_samples error: {exc}")
            return []
        if self.args.min_file_age <= 0:
            return samples
        cutoff = time.time() - self.args.min_file_age
        ready = []
        for s in samples:
            try:
                if s.offline_json.stat().st_mtime <= cutoff:
                    ready.append(s)
            except OSError:
                pass
        return ready

    def _all_ranks_done(self) -> bool:
        return self.barrier_dir.exists() and len(list(self.barrier_dir.iterdir())) >= self.world_size

    def run(self) -> bool:
        (self.run_dir / "full").mkdir(parents=True, exist_ok=True)
        (self.args.results_root / self.args.model_name).mkdir(parents=True, exist_ok=True)
        (self.args.results_root / self.args.model_name / "latest_run.txt").write_text(
            str(self.run_dir) + "\n"
        )
        log(f"[stage 2] watcher start, output_dir={self.run_dir / 'full'}, "
            f"workers={self.args.eval_workers}, poll={self.args.poll_sec}s")

        with ThreadPoolExecutor(max_workers=self.args.eval_workers,
                                thread_name_prefix="eval") as pool:
            final_pass_after = None  # epoch time after which to do the last sweep
            while True:
                # Discover and submit new samples.
                for s in self._discover_ready():
                    if s.sample_id in self.submitted:
                        continue
                    self.samples[s.sample_id] = s
                    self.submitted[s.sample_id] = pool.submit(self._evaluate_one, s)

                # Drain completed futures.
                for sid, fut in list(self.submitted.items()):
                    if fut.done() and sid not in self.results:
                        self.results[sid] = fut.result()

                self._print_progress()

                inflight = sum(1 for f in self.submitted.values() if not f.done())
                if final_pass_after and time.time() >= final_pass_after and inflight == 0:
                    break
                if final_pass_after is None and self._all_ranks_done() and inflight == 0:
                    # All ranks have finished writing offline outputs and the in-flight queue
                    # is empty — schedule one more discover+drain after a short delay so
                    # any sample produced moments before the sentinel still gets picked up.
                    final_pass_after = time.time() + max(self.args.poll_sec, self.args.min_file_age + 1.0)
                time.sleep(self.args.poll_sec)

        self._write_aggregate()
        return all(r.get("status") != "failed" for r in self.results.values())

    def _print_progress(self) -> None:
        total = len(self.submitted)
        done = len(self.results)
        ok = sum(1 for r in self.results.values() if r.get("status") == "completed")
        fail = sum(1 for r in self.results.values() if r.get("status") == "failed")
        inflight = total - done
        elapsed = int(time.time() - self.start_ts)
        eta = ""
        if done > 0 and total > done:
            eta = f"  ETA={int((total - done) * elapsed / done)}s"
        log(f"[stage 2 {elapsed}s] queued={total} done={done}/{total} "
            f"ok={ok} fail={fail} inflight={inflight}{eta}")

    def _write_aggregate(self) -> None:
        ordered_samples = sorted(self.samples.values(), key=lambda s: s.sample_id)
        ordered_results = [self.results[s.sample_id] for s in ordered_samples
                           if s.sample_id in self.results]
        benchmark.write_aggregate_outputs(self.run_dir / "full", ordered_samples, ordered_results)
        log(f"[stage 2] done. summary: {self.run_dir / 'full' / 'summary.json'}")


# ─────────────────────── stage 3 ─────────────────────────

def run_stage3(args: argparse.Namespace) -> int:
    log(f"[stage 3] aggregate report under {args.results_root}")
    return report_results.main(["--results-root", str(args.results_root)]) or 0


# ─────────────────────── rescore-only ────────────────────

def _sample_proxy_from_result(r: dict) -> SimpleNamespace:
    """Reconstruct a Sample-like object from a saved result dict."""
    return SimpleNamespace(
        sample_id=r["sample_id"],
        language=Language(r["language"]),
        meeting_id=r["meeting_id"],
        target_speaker=r["target_speaker"],
        offline_json=Path(r.get("offline_json", "")),
        audio_path=Path(r.get("audio_path", "")),
    )


def rescore_run(run_dir: Path) -> int:
    """Recompute score_summary in place under run_dir/full/samples/ using the
    current ScoreCalculator. No judge calls. Rebuilds aggregate outputs."""
    full_dir = run_dir / "full"
    samples_dir = full_dir / "samples"
    if not samples_dir.exists():
        log(f"[rescore] no samples dir at {samples_dir}")
        return 1

    sample_files = sorted(samples_dir.rglob("*.json"))
    log(f"[rescore] scanning {len(sample_files)} sample file(s) under {samples_dir}")

    calc = ScoreCalculator()
    rescored = 0
    skipped_incomplete = 0
    failed = 0
    results: List[dict] = []

    for path in sample_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            log(f"[rescore] {path.name}: load error: {exc}")
            failed += 1
            continue

        if data.get("status") != "completed":
            skipped_incomplete += 1
            results.append(data)
            continue

        try:
            primary_role = data["calls"]["call1"]["parsed"]["primary_role"]
            role_key = ROLE_CANONICAL[primary_role]
        except (KeyError, TypeError) as exc:
            log(f"[rescore] {path.name}: missing primary_role ({exc})")
            failed += 1
            results.append(data)
            continue

        scores = data.get("scores")
        if not scores:
            try:
                from benchmark import extract_score_call_results
                scores = merge_scores(*extract_score_call_results(data))
            except Exception as exc:
                log(f"[rescore] {path.name}: cannot merge scores: {exc}")
                failed += 1
                results.append(data)
                continue

        old_final = data.get("score_summary", {}).get("final_score")
        ratio = (
            data.get("transcript_stats", {}).get("participation_ratio")
            or data.get("score_summary", {}).get("participation_ratio")
            or 0.0
        )
        summary = calc.calculate(scores, role_key=role_key,
                                 participation_ratio=float(ratio))
        data["scores"] = scores
        data["score_summary"] = summary

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        rescored += 1
        results.append(data)
        if old_final != summary["final_score"]:
            log(f"[rescore] {data['sample_id']}: {old_final} → {summary['final_score']} "
                f"(grade={summary['grade']})")

    log(f"[rescore] rescored={rescored} skipped_incomplete={skipped_incomplete} failed={failed}")

    # Rebuild aggregate outputs from the in-memory results.
    sample_proxies = [_sample_proxy_from_result(r) for r in results
                      if r.get("sample_id")]
    benchmark.write_aggregate_outputs(full_dir, sample_proxies, results)
    log(f"[rescore] aggregate refreshed: {full_dir / 'summary.json'}, "
        f"{full_dir / 'samples.csv'}, {full_dir / 'meetings.csv'}")
    return 0 if failed == 0 else 2


# ─────────────────────── main ────────────────────────────

def touch_sentinel(barrier_dir: Path, rank: int) -> None:
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / f"rank_{rank}.done").touch()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.model_name = derive_model_name(args)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    stages = parse_stages(args.stages)

    # Stage 2 needs the judge endpoint pool. Resolve once at startup so a
    # misconfiguration fails fast with a clear message.
    if 2 in stages or args.rescore_only:
        explicit = (
            [u.strip() for u in args.judge_endpoints.split(",") if u.strip()]
            if args.judge_endpoints
            else None
        )
        from judge import gemma4 as _gemma4
        _gemma4.configure(endpoints=config.judge_endpoints(explicit))

    if args.run_ts:
        run_ts = args.run_ts
    elif args.resume:
        latest_file = args.results_root / args.model_name / "latest_run.txt"
        if latest_file.is_file():
            latest_path = Path(latest_file.read_text().strip())
            run_ts = latest_path.name.removeprefix("run_")
            log(f"[resume] reusing {latest_path} (run_ts={run_ts})")
        else:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log(f"[resume] no latest_run.txt at {latest_file}; starting fresh run_{run_ts}")
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root / args.model_name / f"run_{run_ts}"
    barrier_dir = args.results_root / args.model_name / ".barrier"

    log(f"[eval rank={rank}/{world_size}] model_name={args.model_name} "
        f"stages={sorted(stages)} use_detailed_sys={args.use_detailed_sys}")
    log(f"[eval rank={rank}] offline_root={offline_root_for(args)}")
    log(f"[eval rank={rank}] run_dir={run_dir}")

    # Rescore-only short-circuit: rank 0 only, ignore stages, exit after rebuild.
    if args.rescore_only:
        if rank != 0:
            return 0
        if not args.run_ts:
            log("[rescore] WARNING: --run-ts not provided; using auto-generated "
                f"timestamp '{run_ts}', which probably doesn't match an existing run. "
                "Pass RUN_TS=<existing_ts> (or --run-ts) to target a specific run.")
        return rescore_run(run_dir)

    # Rank 0 prepares barrier and run dir up front.
    if rank == 0 and 1 in stages and 2 in stages:
        if barrier_dir.exists():
            shutil.rmtree(barrier_dir)
        barrier_dir.mkdir(parents=True)
        run_dir.mkdir(parents=True, exist_ok=True)

    watcher_thread: Optional[threading.Thread] = None
    watcher_ok = {"value": True}
    evaluator: Optional[StreamingEvaluator] = None

    # Start streaming watcher on rank 0 in parallel with stage 1.
    if rank == 0 and 1 in stages and 2 in stages:
        evaluator = StreamingEvaluator(args, run_dir, barrier_dir, world_size)

        def _watch():
            try:
                watcher_ok["value"] = evaluator.run()
            except Exception:
                traceback.print_exc()
                watcher_ok["value"] = False

        watcher_thread = threading.Thread(target=_watch, name="stage2-watcher", daemon=False)
        watcher_thread.start()

    # Stage 1 — every rank runs inference (sharded internally by RANK/WORLD_SIZE).
    if 1 in stages:
        # When watcher prints to console, redirect stage 1 to a logfile to keep stdout clean.
        capture_log = run_dir / f"stage1_rank{rank}.log" if watcher_thread else None
        rc = run_stage1(args, rank, world_size, capture_log)
        touch_sentinel(barrier_dir, rank)
        if rc != 0:
            log(f"[stage 1 rank={rank}] inference exited with code {rc}")
            if watcher_thread:
                watcher_thread.join()
            return rc

    # Wait for watcher to drain.
    if watcher_thread:
        watcher_thread.join()
        if not watcher_ok["value"]:
            return 2

    if rank != 0:
        return 0

    # Stage 2 watcher when stage 1 ran in a separate invocation (pipelined mode)
    # or skipped entirely. Touch only OUR rank's sentinel; the watcher polls
    # barrier_dir until all WORLD_SIZE sentinels appear naturally — required for
    # multi-rank correctness so we don't false-positive "all done" when other
    # ranks' stage 1 are still producing samples elsewhere.
    if watcher_thread is None and 2 in stages:
        run_dir.mkdir(parents=True, exist_ok=True)
        touch_sentinel(barrier_dir, rank)
        evaluator = StreamingEvaluator(args, run_dir, barrier_dir, world_size)
        if not evaluator.run():
            return 2

    if 3 in stages:
        run_stage3(args)

    log("[eval] all requested stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
