#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import benchmark
import config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full MultiTalkBench test suite and save artifacts."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Local clone of MultiTalk/MultiTalkBench dataset.")
    parser.add_argument("--offline-root", type=Path, required=True,
                        help="Where the model adapter wrote per-sample <lang>/<sample_id>.json.")
    parser.add_argument("--gap-seconds", type=float, default=1.2)
    parser.add_argument("--max-transcript-chars", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--stage-retries",
        type=int,
        default=1,
        help="Number of retries for each orchestration stage after the first attempt.",
    )
    parser.add_argument(
        "--stage-retry-delay-seconds",
        type=float,
        default=3.0,
        help="Delay between stage retries.",
    )
    parser.add_argument(
        "--call-retries",
        type=int,
        default=2,
        help="Pass-through benchmark per-call retry count.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="Pass-through benchmark per-call retry delay.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def stage_log(run_dir: Path, stage_name: str, suffix: str) -> Path:
    return run_dir / "logs" / f"{stage_name}.{suffix}.log"


def stage_attempt_log(run_dir: Path, stage_name: str, attempt: int, suffix: str) -> Path:
    return run_dir / "logs" / f"{stage_name}.attempt{attempt}.{suffix}.log"


def append_orchestrator_log(run_dir: Path, message: str) -> None:
    path = run_dir / "logs" / "orchestrator.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def benchmark_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    meetings: str = "",
    limit: int = 0,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        "benchmark.py",
        "--data-root", str(args.data_root) if args.data_root else "",
        "--offline-root", str(args.offline_root),
        "--output-dir", str(output_dir),
        "--gap-seconds", str(args.gap_seconds),
        "--max-transcript-chars", str(args.max_transcript_chars),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--call-retries", str(args.call_retries),
        "--retry-delay-seconds", str(args.retry_delay_seconds),
    ]
    # Drop the empty --data-root sentinel if user relies on env var.
    if not args.data_root:
        i = cmd.index("--data-root")
        del cmd[i:i + 2]
    if meetings:
        cmd.extend(["--meetings", meetings])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if dry_run:
        cmd.append("--dry-run")
    if fail_fast:
        cmd.append("--fail-fast")
    if args.force:
        cmd.append("--force")
    return cmd


def env_summary() -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": str(Path.cwd()),
        "timestamp": datetime.now().isoformat(),
        "judge_model": config.judge_model(),
        "judge_endpoints_set": bool(os.getenv("MULTITALKBENCH_JUDGE_ENDPOINTS")),
    }


def discover_all_samples(args: argparse.Namespace):
    ns = argparse.Namespace(
        data_root=args.data_root,
        offline_root=args.offline_root,
        meetings="",
        limit=0,
    )
    return benchmark.discover_samples(ns)


def select_smoke_sample(samples, language: str):
    matching = [sample for sample in samples if sample.language.value == language]
    if not matching:
        raise RuntimeError(f"No samples discovered for language={language}")
    return matching[0]


def validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise AssertionError(f"Missing {label}: {path}")


def validate_dry_run_output(stage_dir: Path, language: str, meeting_id: str, target_speaker: str) -> None:
    summary_path = stage_dir / "summary.json"
    sample_path = stage_dir / "samples" / language / meeting_id / f"{target_speaker}.json"
    validate_file_exists(summary_path, "dry-run summary")
    validate_file_exists(sample_path, "dry-run sample result")

    summary = load_json(summary_path)
    sample = load_json(sample_path)
    assert summary["dry_run_count"] == 1, summary
    assert sample["status"] == "dry_run", sample["status"]
    assert sample["dimension_schema_version"] == "v2_progress", sample
    assert set(sample["prompts"]) == {
        "call1",
        "call2a",
        "call2b",
        "call2c",
        "call3a",
        "call3b",
    }, sample["prompts"].keys()
    assert "G4" in sample["prompts"]["call3a"]["system"], sample["prompts"]["call3a"]["system"]


def validate_smoke_output(stage_dir: Path, language: str, meeting_id: str, target_speaker: str) -> None:
    summary_path = stage_dir / "summary.json"
    sample_path = stage_dir / "samples" / language / meeting_id / f"{target_speaker}.json"
    validate_file_exists(summary_path, "smoke summary")
    validate_file_exists(sample_path, "smoke sample result")

    summary = load_json(summary_path)
    sample = load_json(sample_path)
    assert summary["completed_count"] == 1, summary
    assert summary["failed_count"] == 0, summary
    assert sample["status"] == "completed", sample["status"]
    assert set(sample["calls"]) == {
        "call1",
        "call2a",
        "call2b",
        "call2c",
        "call3a",
        "call3b",
    }, sample["calls"].keys()
    assert all(dim in sample["scores"] for dim in ("G1", "G2", "G3", "G4")), sample["scores"]
    assert all(dim in sample["scores"] for dim in ("M1", "M2", "M3", "M4", "M5", "M6")), sample["scores"]
    assert sample["score_summary"]["final_score"] is not None, sample["score_summary"]
    assert sample["score_summary"]["grade"], sample["score_summary"]


def validate_full_output(stage_dir: Path, expected_sample_count: int) -> None:
    summary_path = stage_dir / "summary.json"
    samples_csv = stage_dir / "samples.csv"
    meetings_csv = stage_dir / "meetings.csv"
    manifest_path = stage_dir / "manifest.jsonl"

    validate_file_exists(summary_path, "full summary")
    validate_file_exists(samples_csv, "samples.csv")
    validate_file_exists(meetings_csv, "meetings.csv")
    validate_file_exists(manifest_path, "manifest.jsonl")

    summary = load_json(summary_path)
    assert summary["sample_count"] == expected_sample_count, summary
    assert summary["completed_count"] == expected_sample_count, summary
    assert summary["failed_count"] == 0, summary
    assert summary["dry_run_count"] == 0, summary
    assert line_count(manifest_path) == expected_sample_count, manifest_path

    with samples_csv.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    expected_cols = {
        "status",
        "language",
        "meeting_id",
        "target_speaker",
        "primary_role",
        "final_score",
        "grade",
        "G1",
        "G2",
        "G3",
        "G4",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
    }
    missing = expected_cols - set(header)
    assert not missing, f"Missing columns in samples.csv: {sorted(missing)}"


def run_stage(
    run_dir: Path,
    report: dict,
    stage_name: str,
    cmd: List[str],
    args: argparse.Namespace,
    validator,
) -> None:
    stage_record = {
        "name": stage_name,
        "command": cmd,
        "status": "running",
        "attempts": [],
    }
    max_attempts = max(1, 1 + args.stage_retries)

    for attempt in range(1, max_attempts + 1):
        stdout_path = stage_attempt_log(run_dir, stage_name, attempt, "stdout")
        stderr_path = stage_attempt_log(run_dir, stage_name, attempt, "stderr")
        if attempt == 1:
            append_orchestrator_log(run_dir, f"[start] {stage_name}: {' '.join(cmd)}")
        else:
            append_orchestrator_log(
                run_dir,
                f"[retry] {stage_name} attempt {attempt}/{max_attempts}: {' '.join(cmd)}",
            )

        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent,
            text=True,
            capture_output=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

        attempt_record = {
            "attempt": attempt,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "returncode": proc.returncode,
            "duration_sec": round(time.time() - started, 3),
            "status": "running",
        }

        if proc.returncode == 0:
            try:
                validator()
            except Exception as exc:
                attempt_record["status"] = "failed"
                attempt_record["error"] = str(exc)
                stage_record["attempts"].append(attempt_record)
                if attempt < max_attempts:
                    time.sleep(max(0.0, args.stage_retry_delay_seconds))
                    continue
                stage_record["status"] = "failed"
                stage_record["error"] = str(exc)
                report["stages"].append(stage_record)
                report["status"] = "failed"
                report["failed_stage"] = stage_name
                atomic_write_json(run_dir / "report.json", report)
                append_orchestrator_log(run_dir, f"[failed] {stage_name}: validation={exc}")
                raise

            attempt_record["status"] = "completed"
            stage_record["attempts"].append(attempt_record)
            stage_record["status"] = "completed"
            stage_record["stdout_log"] = str(stdout_path)
            stage_record["stderr_log"] = str(stderr_path)
            stage_record["returncode"] = proc.returncode
            stage_record["duration_sec"] = attempt_record["duration_sec"]
            report["stages"].append(stage_record)
            atomic_write_json(run_dir / "report.json", report)
            append_orchestrator_log(run_dir, f"[ok] {stage_name}")
            return

        attempt_record["status"] = "failed"
        attempt_record["error"] = f"command exited with {proc.returncode}"
        stage_record["attempts"].append(attempt_record)
        if attempt < max_attempts:
            time.sleep(max(0.0, args.stage_retry_delay_seconds))
            continue

        stage_record["status"] = "failed"
        stage_record["error"] = attempt_record["error"]
        stage_record["stdout_log"] = str(stdout_path)
        stage_record["stderr_log"] = str(stderr_path)
        stage_record["returncode"] = proc.returncode
        stage_record["duration_sec"] = attempt_record["duration_sec"]
        report["stages"].append(stage_record)
        report["status"] = "failed"
        report["failed_stage"] = stage_name
        atomic_write_json(run_dir / "report.json", report)
        append_orchestrator_log(run_dir, f"[failed] {stage_name}: returncode={proc.returncode}")
        raise RuntimeError(stage_record["error"])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.results_root = config.results_root(args.results_root)

    if not args.offline_root.exists():
        raise SystemExit(f"offline_root does not exist: {args.offline_root}")

    model_root = args.results_root / args.model_name
    run_dir = model_root / f"run_{now_ts()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(model_root / "latest_run.txt", str(run_dir) + "\n")

    report = {
        "status": "running",
        "model_name": args.model_name,
        "run_dir": str(run_dir),
        "started_at": datetime.now().isoformat(),
        "stages": [],
    }
    atomic_write_json(run_dir / "report.json", report)
    atomic_write_json(run_dir / "env.json", env_summary())

    samples = discover_all_samples(args)
    en_sample = select_smoke_sample(samples, "en")
    zh_sample = select_smoke_sample(samples, "zh")
    run_config = {
        "model_name": args.model_name,
        "results_root": str(args.results_root),
        "run_dir": str(run_dir),
        "offline_root": str(args.offline_root),
        "data_root": str(args.data_root) if args.data_root else None,
        "gap_seconds": args.gap_seconds,
        "max_transcript_chars": args.max_transcript_chars,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stage_retries": args.stage_retries,
        "stage_retry_delay_seconds": args.stage_retry_delay_seconds,
        "call_retries": args.call_retries,
        "retry_delay_seconds": args.retry_delay_seconds,
        "force": args.force,
        "sample_count": len(samples),
        "english_smoke_sample": {
            "sample_id": en_sample.sample_id,
            "meeting_id": en_sample.meeting_id,
            "target_speaker": en_sample.target_speaker,
        },
        "chinese_smoke_sample": {
            "sample_id": zh_sample.sample_id,
            "meeting_id": zh_sample.meeting_id,
            "target_speaker": zh_sample.target_speaker,
        },
    }
    atomic_write_json(run_dir / "run_config.json", run_config)
    append_orchestrator_log(
        run_dir,
        f"discovered {len(samples)} samples; en={en_sample.sample_id}; zh={zh_sample.sample_id}",
    )

    try:
        dry_run_en_dir = run_dir / "dry_run" / "en"
        run_stage(
            run_dir,
            report,
            "dry_run_en",
            benchmark_command(
                args,
                output_dir=dry_run_en_dir,
                meetings=en_sample.meeting_id,
                limit=1,
                dry_run=True,
            ),
            args,
            lambda: validate_dry_run_output(
                dry_run_en_dir,
                "en",
                en_sample.meeting_id,
                en_sample.target_speaker,
            ),
        )

        dry_run_zh_dir = run_dir / "dry_run" / "zh"
        run_stage(
            run_dir,
            report,
            "dry_run_zh",
            benchmark_command(
                args,
                output_dir=dry_run_zh_dir,
                meetings=zh_sample.meeting_id,
                limit=1,
                dry_run=True,
            ),
            args,
            lambda: validate_dry_run_output(
                dry_run_zh_dir,
                "zh",
                zh_sample.meeting_id,
                zh_sample.target_speaker,
            ),
        )

        smoke_en_dir = run_dir / "smoke" / "en"
        run_stage(
            run_dir,
            report,
            "smoke_en",
            benchmark_command(
                args,
                output_dir=smoke_en_dir,
                meetings=en_sample.meeting_id,
                limit=1,
                fail_fast=True,
            ),
            args,
            lambda: validate_smoke_output(
                smoke_en_dir,
                "en",
                en_sample.meeting_id,
                en_sample.target_speaker,
            ),
        )

        smoke_zh_dir = run_dir / "smoke" / "zh"
        run_stage(
            run_dir,
            report,
            "smoke_zh",
            benchmark_command(
                args,
                output_dir=smoke_zh_dir,
                meetings=zh_sample.meeting_id,
                limit=1,
                fail_fast=True,
            ),
            args,
            lambda: validate_smoke_output(
                smoke_zh_dir,
                "zh",
                zh_sample.meeting_id,
                zh_sample.target_speaker,
            ),
        )

        full_dir = run_dir / "full"
        run_stage(
            run_dir,
            report,
            "full",
            benchmark_command(
                args,
                output_dir=full_dir,
                fail_fast=True,
            ),
            args,
            lambda: validate_full_output(full_dir, len(samples)),
        )
    except Exception as exc:
        report["status"] = "failed"
        report["finished_at"] = datetime.now().isoformat()
        report["error"] = str(exc)
        atomic_write_json(run_dir / "report.json", report)
        append_orchestrator_log(run_dir, f"run failed: {exc}")
        return 1

    report["status"] = "completed"
    report["finished_at"] = datetime.now().isoformat()
    atomic_write_json(run_dir / "report.json", report)
    append_orchestrator_log(run_dir, "run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
