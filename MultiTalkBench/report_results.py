#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import config

BASE_SAMPLE_COLUMNS = {
    "status",
    "language",
    "meeting_id",
    "target_speaker",
    "primary_role",
    "final_score",
    "grade",
    "char_count",
    "target_utterance_count",
}

DIMENSION_GROUPS = {
    "G_general": {"G1", "G2", "G3", "G4"},
    "M_multiparty": {"M1", "M2", "M3", "M4", "M5", "M6"},
    "C_collaborator": {"C1", "C2", "C3", "C4"},
    "F_facilitator": {"F1", "F2", "F3", "F4", "F5"},
    "D_driver": {"D1", "D2", "D3", "D4"},
    "E_evaluator": {"E1", "E2", "E3", "E4"},
    "role_specific_all": {
        "C1",
        "C2",
        "C3",
        "C4",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "D1",
        "D2",
        "D3",
        "D4",
        "E1",
        "E2",
        "E3",
        "E4",
    },
}

BUCKETS = ("overall", "en", "zh")


@dataclass(frozen=True)
class RunFiles:
    model_name: str
    model_dir: Path
    run_dir: Path
    full_dir: Path
    summary_path: Path
    samples_path: Path
    meetings_path: Path
    latest_hint: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and compare MultiTalkBench results across models."
    )
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Default: ./results (or $MULTITALKBENCH_RESULTS).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-root>/model_report_<timestamp>",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="MODEL=/abs/path/to/run_dir",
        help="Override a model run directory. Can be used multiple times.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any discovered model is missing a complete full run.",
    )
    return parser.parse_args(argv)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dimension_sort_key(dim: str) -> Tuple[int, str]:
    prefix_order = {"G": 0, "M": 1, "F": 2, "D": 3, "C": 4, "E": 5}
    return prefix_order.get(dim[:1], 9), dim


def to_float(value: str | float | int | None) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "N/A":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_complete_run(run_dir: Path) -> Tuple[bool, str]:
    full_dir = run_dir / "full"
    summary_path = full_dir / "summary.json"
    samples_path = full_dir / "samples.csv"
    meetings_path = full_dir / "meetings.csv"
    if not summary_path.exists():
        return False, "missing full/summary.json"
    if not samples_path.exists():
        return False, "missing full/samples.csv"
    if not meetings_path.exists():
        return False, "missing full/meetings.csv"
    try:
        summary = load_json(summary_path)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"invalid summary.json: {exc}"
    sample_count = summary.get("sample_count")
    completed_count = summary.get("completed_count")
    failed_count = summary.get("failed_count")
    if not isinstance(sample_count, int) or not isinstance(completed_count, int):
        return False, "summary.json missing sample_count/completed_count"
    if completed_count != sample_count:
        return False, f"incomplete run {completed_count}/{sample_count}"
    if failed_count != 0:
        return False, f"failed_count={failed_count}"
    return True, "ok"


def ignored_model_dir(path: Path) -> bool:
    name = path.name
    return name.startswith("_debug") or name.startswith("comparison_")


def looks_like_model_results_dir(path: Path) -> bool:
    if (path / "latest_run.txt").exists():
        return True
    return any(path.glob("run_*"))


def discover_candidate_runs(model_dir: Path) -> List[Tuple[Path, bool]]:
    candidates: List[Tuple[Path, bool]] = []
    latest_path = model_dir / "latest_run.txt"
    if latest_path.exists():
        latest_value = latest_path.read_text(encoding="utf-8").strip()
        if latest_value:
            candidates.append((Path(latest_value), True))
    for run_dir in sorted(model_dir.glob("run_*"), reverse=True):
        candidates.append((run_dir, False))
    deduped: List[Tuple[Path, bool]] = []
    seen = set()
    for run_dir, hinted in candidates:
        key = str(run_dir.resolve()) if run_dir.exists() else str(run_dir)
        if key in seen:
            continue
        deduped.append((run_dir, hinted))
        seen.add(key)
    return deduped


def select_latest_complete_run(model_dir: Path) -> Tuple[Optional[RunFiles], List[str]]:
    diagnostics: List[str] = []
    for run_dir, latest_hint in discover_candidate_runs(model_dir):
        ok, reason = is_complete_run(run_dir)
        if ok:
            full_dir = run_dir / "full"
            return (
                RunFiles(
                    model_name=model_dir.name,
                    model_dir=model_dir,
                    run_dir=run_dir,
                    full_dir=full_dir,
                    summary_path=full_dir / "summary.json",
                    samples_path=full_dir / "samples.csv",
                    meetings_path=full_dir / "meetings.csv",
                    latest_hint=latest_hint,
                ),
                diagnostics,
            )
        diagnostics.append(f"{model_dir.name}: skip {run_dir} ({reason})")
    diagnostics.append(f"{model_dir.name}: no complete full run found")
    return None, diagnostics


def parse_run_override(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --run value {value!r}, expected MODEL=/abs/path/to/run_dir"
        )
    model_name, run_path = value.split("=", 1)
    model_name = model_name.strip()
    run_dir = Path(run_path.strip())
    if not model_name:
        raise argparse.ArgumentTypeError(f"Invalid --run value {value!r}: missing model name")
    return model_name, run_dir


def resolve_runs(
    results_root: Path,
    overrides: Sequence[str],
) -> Tuple[List[RunFiles], List[str]]:
    diagnostics: List[str] = []
    runs: Dict[str, RunFiles] = {}
    override_map = dict(parse_run_override(item) for item in overrides)

    for model_name, run_dir in override_map.items():
        ok, reason = is_complete_run(run_dir)
        if ok:
            full_dir = run_dir / "full"
            runs[model_name] = RunFiles(
                model_name=model_name,
                model_dir=results_root / model_name,
                run_dir=run_dir,
                full_dir=full_dir,
                summary_path=full_dir / "summary.json",
                samples_path=full_dir / "samples.csv",
                meetings_path=full_dir / "meetings.csv",
                latest_hint=False,
            )
        else:
            diagnostics.append(f"{model_name}: override invalid at {run_dir} ({reason})")

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir() or ignored_model_dir(model_dir):
            continue
        if not looks_like_model_results_dir(model_dir):
            continue
        if model_dir.name in runs:
            continue
        selected, extra = select_latest_complete_run(model_dir)
        diagnostics.extend(extra)
        if selected is not None:
            runs[selected.model_name] = selected

    return sorted(runs.values(), key=lambda item: item.model_name), diagnostics


def sample_bucket(row: Dict[str, str], bucket: str) -> bool:
    return row["status"] == "completed" and (bucket == "overall" or row["language"] == bucket)


def compute_numeric_stats(values: Iterable[float]) -> Optional[Dict[str, float]]:
    seq = list(values)
    if not seq:
        return None
    return {
        "count": len(seq),
        "mean": round(sum(seq) / len(seq), 3),
        "min": round(min(seq), 3),
        "max": round(max(seq), 3),
    }


def rows_for_bucket(rows: Sequence[Dict[str, str]], bucket: str) -> List[Dict[str, str]]:
    return [row for row in rows if sample_bucket(row, bucket)]


def collect_dimensions(rows: Sequence[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    dims = [col for col in rows[0].keys() if col not in BASE_SAMPLE_COLUMNS]
    return sorted(dims, key=dimension_sort_key)


def role_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        key = f"{row['language']}:{row['primary_role']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_model_summary(
    run: RunFiles,
) -> Tuple[dict, List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    summary = load_json(run.summary_path)
    sample_rows = read_csv_rows(run.samples_path)
    meeting_rows = read_csv_rows(run.meetings_path)
    diagnostics: List[str] = []

    dims = collect_dimensions(sample_rows)
    buckets: Dict[str, dict] = {}
    for bucket in BUCKETS:
        subset = rows_for_bucket(sample_rows, bucket)
        final_scores = [to_float(row["final_score"]) for row in subset]
        final_score_values = [score for score in final_scores if score is not None]
        dim_stats: Dict[str, dict] = {}
        for dim in dims:
            values = [to_float(row[dim]) for row in subset]
            numeric_values = [value for value in values if value is not None]
            stats = compute_numeric_stats(numeric_values)
            if stats is not None:
                dim_stats[dim] = stats

        group_stats: Dict[str, dict] = {}
        for group_name, group_dims in DIMENSION_GROUPS.items():
            values: List[float] = []
            for row in subset:
                for dim in dims:
                    if dim in group_dims:
                        value = to_float(row.get(dim))
                        if value is not None:
                            values.append(value)
            stats = compute_numeric_stats(values)
            if stats is not None:
                group_stats[group_name] = stats

        buckets[bucket] = {
            "sample_count": len(subset),
            "final_mean": round(sum(final_score_values) / len(final_score_values), 3)
            if final_score_values
            else None,
            "dims": dim_stats,
            "groups": group_stats,
        }

    meeting_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in meeting_rows:
        meeting_map[(row["language"], row["meeting_id"])] = row

    zh_subset = rows_for_bucket(sample_rows, "zh")
    zh_scored = sorted(
        (
            {
                "score": to_float(row["final_score"]),
                "meeting_id": row["meeting_id"],
                "target_speaker": row["target_speaker"],
            }
            for row in zh_subset
        ),
        key=lambda item: item["score"] if item["score"] is not None else -1.0,
        reverse=True,
    )
    extremes = {
        "top_samples": sorted(
            (
                row
                for row in sample_rows
                if row["status"] == "completed" and to_float(row["final_score"]) is not None
            ),
            key=lambda row: to_float(row["final_score"]) or -1.0,
            reverse=True,
        )[:5],
        "bottom_samples": sorted(
            (
                row
                for row in sample_rows
                if row["status"] == "completed" and to_float(row["final_score"]) is not None
            ),
            key=lambda row: to_float(row["final_score"]) or 999.0,
        )[:5],
        "top_zh_samples": zh_scored[:5],
    }

    payload = {
        "model_name": run.model_name,
        "run_dir": str(run.run_dir),
        "full_dir": str(run.full_dir),
        "summary_path": str(run.summary_path),
        "samples_path": str(run.samples_path),
        "meetings_path": str(run.meetings_path),
        "summary": summary,
        "buckets": buckets,
        "role_counts": role_counts(sample_rows),
        "meeting_map": meeting_map,
        "extremes": extremes,
    }
    if summary.get("failed_count", 0):
        diagnostics.append(f"{run.model_name}: failed_count={summary['failed_count']}")
    return payload, sample_rows, meeting_rows, diagnostics


def serializable_model_payload(payload: dict) -> dict:
    serialized = dict(payload)
    meeting_map = serialized.pop("meeting_map", {})
    serialized["meetings"] = [
        {
            "language": language,
            "meeting_id": meeting_id,
            **meeting_row,
        }
        for (language, meeting_id), meeting_row in sorted(meeting_map.items())
    ]
    return serialized


def best_model_and_delta(score_map: Dict[str, float]) -> Tuple[str, Optional[float]]:
    ordered = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return "", None
    if len(ordered) == 1:
        return ordered[0][0], None
    return ordered[0][0], round(ordered[0][1] - ordered[1][1], 3)


def build_model_overview_rows(model_payloads: Dict[str, dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name, payload in sorted(model_payloads.items()):
        summary = payload["summary"]
        buckets = payload["buckets"]
        rows.append(
            {
                "model_name": model_name,
                "run_dir": payload["run_dir"],
                "sample_count": summary.get("sample_count", ""),
                "completed_count": summary.get("completed_count", ""),
                "failed_count": summary.get("failed_count", ""),
                "mean_score": summary.get("mean_score", ""),
                "overall_mean": buckets["overall"]["final_mean"],
                "en_mean": buckets["en"]["final_mean"],
                "zh_mean": buckets["zh"]["final_mean"],
                "dimension_schema_version": summary.get("dimension_schema_version", ""),
            }
        )
    return rows


def build_dimension_rows(model_payloads: Dict[str, dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    models = sorted(model_payloads)
    for bucket in BUCKETS:
        all_dims = sorted(
            {
                dim
                for payload in model_payloads.values()
                for dim in payload["buckets"][bucket]["dims"].keys()
            },
            key=dimension_sort_key,
        )
        for dim in all_dims:
            score_map: Dict[str, float] = {}
            row: Dict[str, object] = {"bucket": bucket, "dim": dim}
            for model_name in models:
                stats = model_payloads[model_name]["buckets"][bucket]["dims"].get(dim)
                mean_value = stats["mean"] if stats else ""
                count_value = stats["count"] if stats else 0
                row[f"{model_name}_mean"] = mean_value
                row[f"{model_name}_count"] = count_value
                if stats:
                    score_map[model_name] = stats["mean"]
            best_model, delta = best_model_and_delta(score_map)
            row["best_model"] = best_model
            row["best_margin"] = delta if delta is not None else ""
            rows.append(row)
    return rows


def build_group_rows(model_payloads: Dict[str, dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    models = sorted(model_payloads)
    for bucket in BUCKETS:
        for group_name in DIMENSION_GROUPS:
            score_map: Dict[str, float] = {}
            row: Dict[str, object] = {"bucket": bucket, "group": group_name}
            for model_name in models:
                stats = model_payloads[model_name]["buckets"][bucket]["groups"].get(group_name)
                mean_value = stats["mean"] if stats else ""
                count_value = stats["count"] if stats else 0
                row[f"{model_name}_mean"] = mean_value
                row[f"{model_name}_count"] = count_value
                if stats:
                    score_map[model_name] = stats["mean"]
            if score_map:
                best_model, delta = best_model_and_delta(score_map)
                row["best_model"] = best_model
                row["best_margin"] = delta if delta is not None else ""
                rows.append(row)
    return rows


def build_meeting_rows(model_payloads: Dict[str, dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    models = sorted(model_payloads)
    all_keys = sorted(
        {
            key
            for payload in model_payloads.values()
            for key in payload["meeting_map"].keys()
        }
    )
    for language, meeting_id in all_keys:
        score_map: Dict[str, float] = {}
        row: Dict[str, object] = {"language": language, "meeting_id": meeting_id}
        for model_name in models:
            meeting = model_payloads[model_name]["meeting_map"].get((language, meeting_id))
            mean_value = to_float(meeting["mean_score"]) if meeting else None
            row[f"{model_name}_mean"] = mean_value if mean_value is not None else ""
            row[f"{model_name}_completed_count"] = (
                meeting["completed_count"] if meeting else 0
            )
            if mean_value is not None:
                score_map[model_name] = mean_value
        best_model, delta = best_model_and_delta(score_map)
        row["best_model"] = best_model
        row["best_margin"] = delta if delta is not None else ""
        rows.append(row)
    return rows


def build_extreme_rows(model_payloads: Dict[str, dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name, payload in sorted(model_payloads.items()):
        for bucket_name, sample_list in (
            ("top_samples", payload["extremes"]["top_samples"]),
            ("bottom_samples", payload["extremes"]["bottom_samples"]),
        ):
            for index, row in enumerate(sample_list, start=1):
                rows.append(
                    {
                        "model_name": model_name,
                        "category": bucket_name,
                        "rank": index,
                        "language": row["language"],
                        "meeting_id": row["meeting_id"],
                        "target_speaker": row["target_speaker"],
                        "final_score": row["final_score"],
                        "grade": row["grade"],
                        "primary_role": row["primary_role"],
                    }
                )
        for index, row in enumerate(payload["extremes"]["top_zh_samples"], start=1):
            rows.append(
                {
                    "model_name": model_name,
                    "category": "top_zh_samples",
                    "rank": index,
                    "language": "zh",
                    "meeting_id": row["meeting_id"],
                    "target_speaker": row["target_speaker"],
                    "final_score": row["score"],
                    "grade": "",
                    "primary_role": "",
                }
            )
    return rows


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def top_delta_rows(
    rows: Sequence[Dict[str, object]],
    key_name: str,
    top_n: int = 8,
) -> List[Dict[str, object]]:
    scored = [
        row
        for row in rows
        if isinstance(row.get("best_margin"), (int, float))
    ]
    return sorted(scored, key=lambda row: float(row["best_margin"]), reverse=True)[:top_n]


def build_report_markdown(
    model_payloads: Dict[str, dict],
    overview_rows: List[Dict[str, object]],
    dimension_rows: List[Dict[str, object]],
    group_rows: List[Dict[str, object]],
    meeting_rows: List[Dict[str, object]],
    extreme_rows: List[Dict[str, object]],
    diagnostics: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# MultiTalkBench Multi-Model Report")
    lines.append("")
    lines.append(f"Generated at `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    overview_columns = [
        "model_name",
        "mean_score",
        "en_mean",
        "zh_mean",
        "sample_count",
        "completed_count",
        "failed_count",
        "dimension_schema_version",
    ]
    lines.append(markdown_table(overview_rows, overview_columns))
    lines.append("")

    lines.append("## Runs")
    lines.append("")
    for model_name, payload in sorted(model_payloads.items()):
        lines.append(f"- `{model_name}`: `{payload['run_dir']}`")
    lines.append("")

    lines.append("## Group Comparison")
    lines.append("")
    for bucket in BUCKETS:
        bucket_rows = [row for row in group_rows if row["bucket"] == bucket]
        lines.append(f"### {bucket}")
        lines.append("")
        columns = ["group"]
        for model_name in sorted(model_payloads):
            columns.extend([f"{model_name}_mean", f"{model_name}_count"])
        columns.extend(["best_model", "best_margin"])
        lines.append(markdown_table(bucket_rows, columns))
        lines.append("")

    lines.append("## Top Dimension Gaps")
    lines.append("")
    for bucket in BUCKETS:
        bucket_rows = [row for row in dimension_rows if row["bucket"] == bucket]
        lines.append(f"### {bucket}")
        lines.append("")
        top_rows = top_delta_rows(bucket_rows, "dim", top_n=10)
        columns = ["dim"]
        for model_name in sorted(model_payloads):
            columns.extend([f"{model_name}_mean", f"{model_name}_count"])
        columns.extend(["best_model", "best_margin"])
        lines.append(markdown_table(top_rows, columns))
        lines.append("")

    lines.append("## Meeting Comparison")
    lines.append("")
    meeting_columns = ["language", "meeting_id"]
    for model_name in sorted(model_payloads):
        meeting_columns.extend([f"{model_name}_mean", f"{model_name}_completed_count"])
    meeting_columns.extend(["best_model", "best_margin"])
    lines.append(markdown_table(meeting_rows, meeting_columns))
    lines.append("")

    lines.append("## Extremes")
    lines.append("")
    lines.append(
        markdown_table(
            extreme_rows,
            [
                "model_name",
                "category",
                "rank",
                "language",
                "meeting_id",
                "target_speaker",
                "final_score",
                "grade",
                "primary_role",
            ],
        )
    )
    lines.append("")

    lines.append("## Role Counts")
    lines.append("")
    role_rows: List[Dict[str, object]] = []
    role_keys = sorted(
        {
            role
            for payload in model_payloads.values()
            for role in payload["role_counts"].keys()
        }
    )
    for role_key in role_keys:
        row: Dict[str, object] = {"role_key": role_key}
        for model_name in sorted(model_payloads):
            row[model_name] = model_payloads[model_name]["role_counts"].get(role_key, 0)
        role_rows.append(row)
    role_columns = ["role_key", *sorted(model_payloads)]
    lines.append(markdown_table(role_rows, role_columns))
    lines.append("")

    lines.append("## Diagnostics")
    lines.append("")
    if diagnostics:
        for item in diagnostics:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def ensure_parent_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = config.results_root(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"results root does not exist: {results_root}")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else results_root / f"model_report_{now_ts()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, diagnostics = resolve_runs(results_root, args.run)
    if not runs:
        atomic_write_text(output_dir / "report.md", "# MultiTalkBench Multi-Model Report\n\nNo valid runs found.\n")
        if args.strict:
            return 1
        return 0

    model_payloads: Dict[str, dict] = {}
    for run in runs:
        payload, _sample_rows, _meeting_rows, extra = build_model_summary(run)
        model_payloads[run.model_name] = payload
        diagnostics.extend(extra)

    if args.strict and any("no complete full run found" in item or "override invalid" in item for item in diagnostics):
        strict_failure = True
    else:
        strict_failure = False

    overview_rows = build_model_overview_rows(model_payloads)
    dimension_rows = build_dimension_rows(model_payloads)
    group_rows = build_group_rows(model_payloads)
    meeting_rows = build_meeting_rows(model_payloads)
    extreme_rows = build_extreme_rows(model_payloads)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "runs": {
            model_name: {
                "run_dir": model_payloads[model_name]["run_dir"],
                "summary_path": model_payloads[model_name]["summary_path"],
                "samples_path": model_payloads[model_name]["samples_path"],
                "meetings_path": model_payloads[model_name]["meetings_path"],
            }
            for model_name in sorted(model_payloads)
        },
        "models": {
            model_name: serializable_model_payload(model_payloads[model_name])
            for model_name in sorted(model_payloads)
        },
        "tables": {
            "model_overview": overview_rows,
            "dimension_comparison": dimension_rows,
            "group_comparison": group_rows,
            "meeting_comparison": meeting_rows,
            "sample_extremes": extreme_rows,
        },
        "diagnostics": diagnostics,
    }

    atomic_write_json(output_dir / "report.json", payload)

    model_overview_fields = list(overview_rows[0].keys()) if overview_rows else []
    write_csv(output_dir / "model_overview.csv", overview_rows, model_overview_fields)

    dimension_fields = list(dimension_rows[0].keys()) if dimension_rows else []
    write_csv(output_dir / "dimension_comparison.csv", dimension_rows, dimension_fields)

    group_fields = list(group_rows[0].keys()) if group_rows else []
    write_csv(output_dir / "group_comparison.csv", group_rows, group_fields)

    meeting_fields = list(meeting_rows[0].keys()) if meeting_rows else []
    write_csv(output_dir / "meeting_comparison.csv", meeting_rows, meeting_fields)

    extreme_fields = list(extreme_rows[0].keys()) if extreme_rows else []
    write_csv(output_dir / "sample_extremes.csv", extreme_rows, extreme_fields)

    report_md = build_report_markdown(
        model_payloads=model_payloads,
        overview_rows=overview_rows,
        dimension_rows=dimension_rows,
        group_rows=group_rows,
        meeting_rows=meeting_rows,
        extreme_rows=extreme_rows,
        diagnostics=diagnostics,
    )
    atomic_write_text(output_dir / "report.md", report_md)
    print(output_dir)
    return 1 if strict_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
