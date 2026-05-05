#!/usr/bin/env python3
"""Re-score every completed sample under a results root with the current
``ScoreCalculator``. Each sample's participation coefficient is measured
against the human ground-truth ratio for that same (meeting, target speaker),
computed on-the-fly from the dataset's alignment files. P1 ∈ [1, 5] is
rederived as ``1 + 4·coef`` so it shares the 1–5 scale with the LLM-judged
dimensions. Rebuilds samples.csv / meetings.csv / summary.json per run.

Usage::

    python rescore_all.py [--results-root <path>] [--gap-seconds 1.2] [--data-root <path>]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark
import config
import data as dataset
from prompt_builder import Language, ROLE_CANONICAL, ScoreCalculator, merge_scores


def participation_for(
    result: dict,
    gap_seconds: float,
    by_id: dict,
) -> Tuple[Optional[float], Optional[dict]]:
    """Use cached transcript_stats.participation_ratio if present; otherwise
    reconstruct the runtime sample from the dataset and recompute."""
    cached = result.get("transcript_stats", {}).get("participation_ratio")
    if cached is not None:
        return float(cached), None

    sid = result.get("sample_id")
    ds_sample = by_id.get(sid)
    offline = Path(result.get("offline_json", ""))
    if ds_sample is None or not offline.exists():
        return None, None

    peers = [s for s in by_id.values()
             if s.language == ds_sample.language and s.meeting_id == ds_sample.meeting_id]
    proxy = benchmark.make_runtime_sample(ds_sample, peers, offline)
    try:
        _, _, stats = benchmark.build_transcript(proxy, gap_seconds=gap_seconds)
    except Exception as exc:
        print(f"  [warn] build_transcript failed for {sid}: {exc}", flush=True)
        return None, None
    return float(stats.get("participation_ratio") or 0.0), stats


def gt_ratio_for(sid: str, by_id: dict) -> Optional[float]:
    """Compute GT participation ratio for a sample, using its own GT alignment
    plus all peer alignments from the dataset."""
    ds_sample = by_id.get(sid)
    if ds_sample is None:
        return None
    peers = [s for s in by_id.values()
             if s.language == ds_sample.language and s.meeting_id == ds_sample.meeting_id]
    proxy = benchmark.make_runtime_sample(ds_sample, peers, ds_sample.alignment_path)
    try:
        return benchmark.compute_gt_participation_ratio(proxy)
    except Exception:
        return None


def rescore_sample(path: Path, gap_seconds: float, by_id: dict) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "completed":
        return "skipped_incomplete"

    try:
        primary_role = data["calls"]["call1"]["parsed"]["primary_role"]
        role_key = ROLE_CANONICAL[primary_role]
    except (KeyError, TypeError):
        return "no_role"

    scores = data.get("scores")
    if not scores:
        try:
            scores = merge_scores(*benchmark.extract_score_call_results(data))
        except Exception:
            return "no_scores"

    ratio, fresh_stats = participation_for(data, gap_seconds, by_id)
    if ratio is None:
        return "no_participation"

    gt_ratio = gt_ratio_for(data.get("sample_id", ""), by_id)
    if gt_ratio is None:
        coef = float(ratio)
    else:
        coef = max(0.0, 1.0 - abs(float(ratio) - float(gt_ratio)))
    # P1 ∈ [1, 5] is the "match-the-GT" score (5 = matches human, 1 = totally off).
    scores["P1"] = round(1 + 4 * coef, 2)

    summary = ScoreCalculator().calculate(
        scores,
        role_key=role_key,
        participation_ratio=float(ratio),
        gt_participation_ratio=gt_ratio,
    )
    data["scores"] = scores
    data["score_summary"] = summary
    if fresh_stats:
        data["transcript_stats"] = fresh_stats

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return "ok"


def rescore_run(run_dir: Path, gap_seconds: float, by_id: dict) -> Optional[dict]:
    samples_dir = run_dir / "full" / "samples"
    if not samples_dir.exists():
        return None

    counts = {"ok": 0, "skipped_incomplete": 0,
              "no_role": 0, "no_scores": 0, "no_participation": 0, "error": 0}
    sample_files = sorted(samples_dir.rglob("*.json"))
    for path in sample_files:
        try:
            counts[rescore_sample(path, gap_seconds, by_id)] += 1
        except Exception:
            traceback.print_exc()
            counts["error"] += 1

    results = []
    for path in sample_files:
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    sample_proxies = [
        SimpleNamespace(
            sample_id=r["sample_id"],
            language=Language(r["language"]),
            meeting_id=r["meeting_id"],
            target_speaker=r["target_speaker"],
            offline_json=Path(r.get("offline_json", "")),
            audio_path=Path(r.get("audio_path", "")),
        )
        for r in results if r.get("sample_id")
    ]
    benchmark.write_aggregate_outputs(run_dir / "full", sample_proxies, results)
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None,
                   help="Required only when reconstructing transcripts from scratch.")
    p.add_argument("--gap-seconds", type=float, default=1.2)
    args = p.parse_args()

    results_root = config.results_root(args.results_root)

    by_id: dict = {}
    try:
        by_id = dataset.load_samples_dict(config.data_root(args.data_root))
    except Exception as exc:
        print(f"[rescore_all] dataset not configured ({exc}); will only rescore "
              "samples that already cached participation_ratio.", flush=True)

    runs = sorted(results_root.glob("*/run_*"))
    print(f"[rescore_all] {results_root}: {len(runs)} run dir(s)", flush=True)

    overall = {"ok": 0, "skipped_incomplete": 0,
               "no_role": 0, "no_scores": 0, "no_participation": 0, "error": 0}
    for run in runs:
        counts = rescore_run(run, args.gap_seconds, by_id)
        tag = f"{run.parent.name}/{run.name}"
        if counts is None:
            print(f"  - {tag}: skipped (no full/samples)", flush=True)
            continue
        for k in overall:
            overall[k] += counts[k]
        print(f"  - {tag}: {counts}", flush=True)

    print(f"[rescore_all] totals: {overall}", flush=True)
    return 0 if overall["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
