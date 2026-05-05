from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import config
import data as dataset
from judge import gemma4
from prompt_builder import (
    Language,
    PipelineInput,
    PromptPair,
    PromptBuilder,
    ROLE_CANONICAL,
    ScoreCalculator,
    merge_scores,
    parse_call1_result,
    parse_score_result,
    score_dims_for_call,
)


SCHEMA_VERSION = "v2_progress"
CALL_ORDER = ("call1", "call2a", "call2b", "call2c", "call3a", "call3b")
SCORE_CALLS = ("call2a", "call2b", "call2c", "call3a", "call3b")
PAD_TOKENS = {"PAD", "EPAD"}
CALL_MAX_TOKENS = {
    "call1": 8192,
    "call2a": 8192,
    "call2b": 8192,
    "call2c": 8192,
    "call3a": 8192,
    "call3b": 8192,
}
CONTEXT_ERROR_RE = re.compile(
    r"maximum context length is (\d+) tokens.*requested (\d+) output tokens.*"
    r"prompt contains at least (\d+) input tokens",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    """A runtime evaluation sample.

    The dataset-level fields come from ``data.Sample`` (driven by
    ``metadata.jsonl``); ``offline_json`` and ``peer_alignments`` are populated
    at discovery time once the model has produced output.
    """
    sample_id: str
    language: Language
    meeting_id: str
    target_speaker: str
    offline_json: Path
    audio_path: Path
    target_alignment: Path
    # peer_alignments[other_speaker] -> path to that speaker's *_processed.json
    peer_alignments: Dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker: str
    text: str
    source: str


class CallAttemptError(RuntimeError):
    def __init__(self, message: str, attempts: List[dict]):
        super().__init__(message)
        self.attempts = attempts


def speaker_alias(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"S{index + 1}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MultiTalkBench Gemma-based benchmark over offline outputs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to a local clone of the MultiTalk/MultiTalkBench dataset "
        "(falls back to $MULTITALKBENCH_DATA).",
    )
    parser.add_argument(
        "--offline-root",
        type=Path,
        required=True,
        help="Directory written by a model adapter; contains <lang>/<sample_id>.json files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples.")
    parser.add_argument(
        "--meetings",
        default="",
        help="Comma-separated meeting ids to include, e.g. EN2002a,R8002_M8002.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed sample and write partial aggregate outputs.",
    )
    parser.add_argument(
        "--call-retries",
        type=int,
        default=2,
        help="Number of retries for each Gemma call after the first attempt.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="Delay between per-call retries.",
    )
    parser.add_argument("--gap-seconds", type=float, default=1.2)
    parser.add_argument(
        "--max-transcript-chars",
        type=int,
        default=0,
        help="0 disables length guard; otherwise skip samples above this length.",
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--judge-endpoints",
        type=str,
        default=None,
        help="Comma-separated judge LLM base URLs (overrides endpoints.json/env).",
    )
    return parser.parse_args(argv)


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def offline_path_for(offline_root: Path, ds_sample: dataset.Sample) -> Path:
    """Canonical place a model adapter writes the offline output for one sample."""
    return offline_root / ds_sample.language.value / f"{ds_sample.target_speaker}.json"


def make_runtime_sample(
    ds_sample: dataset.Sample,
    peers: List[dataset.Sample],
    offline_json: Path,
) -> Sample:
    peer_alignments = {
        p.target_speaker: p.alignment_path
        for p in peers
        if p.target_speaker != ds_sample.target_speaker
    }
    return Sample(
        sample_id=ds_sample.sample_id,
        language=ds_sample.language,
        meeting_id=ds_sample.meeting_id,
        target_speaker=ds_sample.target_speaker,
        offline_json=offline_json,
        audio_path=ds_sample.audio_path,
        target_alignment=ds_sample.alignment_path,
        peer_alignments=peer_alignments,
    )


def discover_samples(args: argparse.Namespace) -> List[Sample]:
    """Walk metadata.jsonl, return the runtime Samples whose model output exists."""
    root = config.data_root(getattr(args, "data_root", None))
    ds_samples = dataset.filter_meetings(dataset.load_samples(root), args.meetings)
    by_meeting = dataset.group_by_meeting(ds_samples)

    out: List[Sample] = []
    for ds in ds_samples:
        offline_json = offline_path_for(args.offline_root, ds)
        if not offline_json.exists():
            continue
        peers = by_meeting[(ds.language, ds.meeting_id)]
        out.append(make_runtime_sample(ds, peers, offline_json))

    out.sort(key=lambda s: (s.language.value, s.meeting_id, s.target_speaker))
    if args.limit > 0:
        out = out[: args.limit]
    return out


def load_processed_utterances(processed_file: Path, speaker: str) -> List[Utterance]:
    data = load_json(processed_file)
    utterances: List[Utterance] = []
    for seg in data.get("sentence_alignments", []):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        utterances.append(
            Utterance(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", seg.get("start", 0.0))),
                speaker=speaker,
                text=text,
                source="processed",
            )
        )
    return utterances


def load_offline_utterances(
    offline_json: Path,
    *,
    speaker: str,
    gap_seconds: float,
) -> List[Utterance]:
    data = load_json(offline_json)
    alignments = data.get("alignments", [])
    utterances: List[Utterance] = []
    tokens: List[str] = []
    seg_start: Optional[float] = None
    seg_end: Optional[float] = None
    prev_end: Optional[float] = None

    def flush() -> None:
        nonlocal tokens, seg_start, seg_end
        text = "".join(tokens).strip()
        if text and seg_start is not None and seg_end is not None:
            utterances.append(
                Utterance(
                    start=seg_start,
                    end=seg_end,
                    speaker=speaker,
                    text=text,
                    source="offline",
                )
            )
        tokens = []
        seg_start = None
        seg_end = None

    for item in alignments:
        if not isinstance(item, list) or len(item) < 2:
            continue
        token = str(item[0])
        span = item[1]
        if token in PAD_TOKENS or not token.strip():
            continue
        if not isinstance(span, list) or len(span) < 2:
            continue
        start = float(span[0])
        end = float(span[1])
        if tokens and prev_end is not None and start - prev_end > gap_seconds:
            flush()
        if not tokens:
            seg_start = start
        tokens.append(token)
        seg_end = end
        prev_end = end

    flush()
    return utterances


def compute_participation(
    utterances: List[Utterance],
    target_speaker: str,
) -> Dict[str, object]:
    """Mechanical participation metric:
        (target speaker's last utterance end − first utterance start)
        ÷ (overall meeting span across all speakers)
    """
    target_spans = [(u.start, u.end) for u in utterances if u.speaker == target_speaker]
    if not target_spans:
        return {
            "participation_ratio": 0.0,
            "target_first_start": None,
            "target_last_end": None,
            "meeting_total_seconds": 0.0,
        }
    target_first = min(s for s, _ in target_spans)
    target_last = max(e for _, e in target_spans)
    meeting_first = min(u.start for u in utterances)
    meeting_last = max(u.end for u in utterances)
    total = meeting_last - meeting_first
    span = target_last - target_first
    ratio = span / total if total > 0 else 0.0
    return {
        "participation_ratio": round(ratio, 4),
        "target_first_start": round(target_first, 3),
        "target_last_end": round(target_last, 3),
        "meeting_total_seconds": round(total, 3),
    }


def compute_gt_participation_ratio(sample: Sample) -> Optional[float]:
    """Participation ratio if the target speaker were replaced by their own
    ground-truth alignment. Used as the per-sample reference for the
    multiplicative coefficient in ScoreCalculator."""
    utterances: List[Utterance] = []
    utterances.extend(load_processed_utterances(sample.target_alignment, sample.target_speaker))
    if not utterances:
        return None
    for speaker in sample.peer_alignments:
        utterances.extend(load_processed_utterances(sample.peer_alignments[speaker], speaker))
    stats = compute_participation(utterances, sample.target_speaker)
    return stats.get("participation_ratio")


def build_transcript(
    sample: Sample,
    *,
    gap_seconds: float,
) -> Tuple[str, List[str], Dict[str, object]]:
    """Interleave the model's offline output (target speaker) with ground-truth
    sentence alignments for every other speaker in the same meeting."""
    target_utts = load_offline_utterances(
        sample.offline_json,
        speaker=sample.target_speaker,
        gap_seconds=gap_seconds,
    )
    target_count = len(target_utts)

    utterances: List[Utterance] = list(target_utts)
    peer_speakers = sorted(sample.peer_alignments)
    for speaker in peer_speakers:
        utterances.extend(load_processed_utterances(sample.peer_alignments[speaker], speaker))

    all_speakers = [sample.target_speaker, *peer_speakers]
    utterances.sort(key=lambda item: (item.start, item.end, item.speaker))
    aliases = {
        speaker: speaker_alias(index)
        for index, speaker in enumerate(all_speakers)
    }
    transcript = "\n".join(f"{aliases[utt.speaker]}: {utt.text}" for utt in utterances)
    stats: Dict[str, object] = {
        "utterance_count": len(utterances),
        "target_utterance_count": target_count,
        "char_count": len(transcript),
        "speaker_aliases": aliases,
    }
    stats.update(compute_participation(utterances, sample.target_speaker))
    return transcript, all_speakers, stats


def result_path(output_dir: Path, sample: Sample) -> Path:
    return (
        output_dir
        / "samples"
        / sample.language.value
        / sample.meeting_id
        / f"{sample.target_speaker}.json"
    )


def make_base_result(sample: Sample, stats: Dict[str, int], all_speakers: List[str]) -> dict:
    target_alias = stats.get("speaker_aliases", {}).get(sample.target_speaker, "")
    return {
        "status": "running",
        "dimension_schema_version": SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "language": sample.language.value,
        "meeting_id": sample.meeting_id,
        "target_speaker": sample.target_speaker,
        "target_speaker_alias": target_alias,
        "all_speakers": all_speakers,
        "offline_json": str(sample.offline_json),
        "audio_path": str(sample.audio_path),
        "transcript_stats": stats,
        "calls": {},
        "scores": {},
        "score_summary": {},
        "errors": [],
    }


def requested_max_tokens(call_id: str, cli_max_tokens: int) -> int:
    return min(cli_max_tokens, CALL_MAX_TOKENS.get(call_id, cli_max_tokens))


def parse_context_error(message: str) -> Optional[Tuple[int, int, int]]:
    match = CONTEXT_ERROR_RE.search(message)
    if not match:
        return None
    return tuple(int(match.group(i)) for i in range(1, 4))


def extract_score_call_results(result: dict) -> List[Dict[str, object]]:
    score_dicts: List[Dict[str, object]] = []
    for call_id in SCORE_CALLS:
        call = result.get("calls", {}).get(call_id)
        if not call:
            continue
        parsed = call.get("parsed")
        if isinstance(parsed, dict):
            score_dicts.append(parsed)
    return score_dicts


def ensure_call_success(result: dict, call_id: str) -> bool:
    call = result.get("calls", {}).get(call_id)
    return isinstance(call, dict) and call.get("status") == "completed"


def build_prompts_for_call(
    builder: PromptBuilder,
    inp: PipelineInput,
    call_id: str,
):
    if call_id == "call1":
        return builder.build_call1(inp)
    if call_id == "call2a":
        return builder.build_call2a(inp)
    if call_id == "call2b":
        return builder.build_call2b(inp)
    if call_id == "call2c":
        return builder.build_call2c(inp)
    if call_id == "call3a":
        return builder.build_call3a(inp)
    if call_id == "call3b":
        return builder.build_call3b(inp)
    raise ValueError(f"Unknown call id: {call_id}")


def add_target_alias_focus(
    prompt_pair: PromptPair,
    *,
    language: Language,
    target_alias: str,
) -> PromptPair:
    if language == Language.ZH:
        reminder = (
            f"重要：待评估对象是说话人 {target_alias}。"
            f"只能根据以“{target_alias}:”开头的发言进行评分或举证；"
            "不要把其他说话人的内容算到该对象名下。"
        )
    else:
        reminder = (
            f"Important: the participant being evaluated is speaker {target_alias}. "
            f"Only score or cite utterances spoken on lines that begin with '{target_alias}:'. "
            "Do not use other speakers' utterances as evidence for the target participant."
        )
    return PromptPair(
        call_id=prompt_pair.call_id,
        system=prompt_pair.system,
        user=f"{prompt_pair.user}\n\n{reminder}",
        parallel_group=prompt_pair.parallel_group,
        description=prompt_pair.description,
    )


def compact_score_prompt(prompt_pair: PromptPair, expected_dims: Sequence[str]) -> PromptPair:
    dims_json = ", ".join(f'"{dim}"' for dim in expected_dims)
    skeleton = "{\n" + ",\n".join(
        (
            f'  "{dim}": {{'
            '"score": 1, '
            '"positive_evidence": [], '
            '"negative_evidence": [], '
            '"reasoning": "compact retry"'
            "}"
        )
        for dim in expected_dims
    ) + "\n}"
    compact_system = (
        "RETRY FORMAT OVERRIDE: The previous answer was not valid JSON. "
        "For this retry, do not inspect or quote the transcript. "
        "Output only one compact, complete JSON object. "
        "No Markdown fences, no direct quotes, no evidence text, no commentary. "
        "Use [] for every evidence field. "
        f"The top-level keys must be exactly: {dims_json}. "
        "If the target speaker is incoherent or the score is uncertain, use score 1."
    )
    compact_user = (
        "Return compact JSON only using this exact shape, with every requested dimension present:\n"
        f"{skeleton}\n"
        "You may change score and reasoning, but do not add quotes or evidence text."
    )
    return PromptPair(
        call_id=prompt_pair.call_id,
        system=compact_system,
        user=compact_user,
        parallel_group=prompt_pair.parallel_group,
        description=f"{prompt_pair.description} (compact retry)",
    )


def run_call_with_retries(
    *,
    call_id: str,
    prompt_pair,
    args: argparse.Namespace,
    primary_role: Optional[str] = None,
) -> Tuple[str, dict, List[dict]]:
    attempts: List[dict] = []
    max_attempts = max(1, 1 + args.call_retries)
    expected_dims = (
        None
        if call_id == "call1"
        else score_dims_for_call(call_id, primary_role=primary_role)
    )

    for attempt in range(1, max_attempts + 1):
        raw = ""
        active_prompt = (
            prompt_pair
            if attempt == 1 or call_id == "call1"
            else compact_score_prompt(prompt_pair, expected_dims or [])
        )
        retry_mode = "default" if active_prompt is prompt_pair else "compact_json"
        try:
            raw = gemma4.chat(
                active_prompt.to_openai_messages(),
                max_tokens=requested_max_tokens(call_id, args.max_tokens),
                temperature=args.temperature,
            )
            parsed = (
                parse_call1_result(raw)
                if call_id == "call1"
                else parse_score_result(raw, expected_dims=expected_dims)
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "retry_mode": retry_mode,
                    "status": "completed",
                    "raw": raw,
                    "parsed": parsed,
                }
            )
            return raw, parsed, attempts
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "retry_mode": retry_mode,
                    "status": "failed",
                    "raw": raw,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if attempt >= max_attempts:
                raise CallAttemptError(str(exc), attempts) from exc
            time.sleep(max(0.0, args.retry_delay_seconds))

    raise CallAttemptError("unreachable retry state", attempts)


def run_or_resume_sample(
    sample: Sample,
    args: argparse.Namespace,
    *,
    builder: PromptBuilder,
) -> dict:
    out_path = result_path(args.output_dir, sample)
    if out_path.exists() and not args.force:
        existing = load_json(out_path)
        if existing.get("status") == "completed":
            print(f"[skip] {sample.sample_id}")
            return existing
        result = existing
    else:
        result = {}

    transcript, all_speakers, stats = build_transcript(
        sample,
        gap_seconds=args.gap_seconds,
    )
    target_alias = stats["speaker_aliases"][sample.target_speaker]
    eval_speakers = [stats["speaker_aliases"][speaker] for speaker in all_speakers]
    if args.max_transcript_chars and stats["char_count"] > args.max_transcript_chars:
        result = make_base_result(sample, stats, all_speakers)
        result["status"] = "failed"
        result["errors"].append(
            {
                "stage": "transcript",
                "message": (
                    f"transcript char count {stats['char_count']} exceeds "
                    f"--max-transcript-chars {args.max_transcript_chars}"
                ),
            }
        )
        atomic_write_json(out_path, result)
        return result

    base = make_base_result(sample, stats, all_speakers)
    base["calls"] = result.get("calls", {})
    base["scores"] = result.get("scores", {})
    base["score_summary"] = result.get("score_summary", {})
    base["errors"] = result.get("errors", [])
    result = base

    inp = PipelineInput(
        transcript=transcript,
        target_speaker=target_alias,
        all_speakers=eval_speakers,
        language=sample.language,
    )

    if args.dry_run:
        result["status"] = "dry_run"
        result["prompts"] = {}
        for call_id in CALL_ORDER:
            if call_id == "call3b":
                role = "主持人" if sample.language == Language.ZH else "Facilitator"
                inp.primary_role = role
            prompt_pair = add_target_alias_focus(
                build_prompts_for_call(builder, inp, call_id),
                language=sample.language,
                target_alias=target_alias,
            )
            result["prompts"][call_id] = prompt_pair.to_dict()
        atomic_write_json(out_path, result)
        return result

    try:
        for call_id in CALL_ORDER:
            if ensure_call_success(result, call_id):
                if call_id == "call1":
                    parsed = result["calls"]["call1"].get("parsed", {})
                    inp.primary_role = parsed.get("primary_role")
                continue

            if call_id == "call3b":
                call1 = result.get("calls", {}).get("call1", {})
                parsed = call1.get("parsed", {})
                inp.primary_role = parsed.get("primary_role")
                if not inp.primary_role:
                    raise ValueError("Cannot build call3b without call1 primary_role")

            prompt_pair = add_target_alias_focus(
                build_prompts_for_call(builder, inp, call_id),
                language=sample.language,
                target_alias=target_alias,
            )
            raw, parsed, attempts = run_call_with_retries(
                call_id=call_id,
                prompt_pair=prompt_pair,
                args=args,
                primary_role=inp.primary_role,
            )
            if call_id == "call1":
                inp.primary_role = parsed["primary_role"]

            result["calls"][call_id] = {
                "status": "completed",
                "description": prompt_pair.description,
                "raw": raw,
                "parsed": parsed,
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
            atomic_write_json(out_path, result)

        role_key = ROLE_CANONICAL[result["calls"]["call1"]["parsed"]["primary_role"]]
        scores = merge_scores(*extract_score_call_results(result))
        ratio = float(stats.get("participation_ratio") or 0.0)
        gt_ratio = compute_gt_participation_ratio(sample)
        if gt_ratio is None:
            coef = ratio
        else:
            coef = max(0.0, 1.0 - abs(ratio - float(gt_ratio)))
        # P1 ∈ [1, 5] is the "match-the-GT" score, sharing the 1–5 scale with
        # the LLM-judged dimensions (5 = matches human; 1 = totally off).
        scores["P1"] = round(1 + 4 * coef, 2)
        summary = ScoreCalculator().calculate(
            scores,
            role_key=role_key,
            participation_ratio=ratio,
            gt_participation_ratio=gt_ratio,
        )
        result["scores"] = scores
        result["score_summary"] = summary
        result["status"] = "completed"
        atomic_write_json(out_path, result)
        print(
            f"[ok] {sample.sample_id} score={summary['final_score']} "
            f"grade={summary['grade']}"
        )
        return result
    except Exception as exc:
        result["status"] = "failed"
        context_info = parse_context_error(str(exc))
        error_record = {
            "stage": "scoring",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, CallAttemptError):
            error_record["attempts"] = exc.attempts
            error_record["attempt_count"] = len(exc.attempts)
        if context_info is not None:
            limit, requested_output, input_tokens = context_info
            error_record["context_limit"] = limit
            error_record["requested_output_tokens"] = requested_output
            error_record["input_tokens"] = input_tokens
        result["errors"].append(
            error_record
        )
        atomic_write_json(out_path, result)
        print(f"[failed] {sample.sample_id}: {exc}")
        return result


def safe_score(result: dict) -> Optional[float]:
    if result.get("status") != "completed":
        return None
    score = result.get("score_summary", {}).get("final_score")
    return float(score) if score is not None else None


def role_from_result(result: dict) -> str:
    call = result.get("calls", {}).get("call1", {})
    parsed = call.get("parsed", {})
    return str(parsed.get("primary_role", ""))


def dimension_columns(results: Sequence[dict]) -> List[str]:
    dims = set()
    for result in results:
        dims.update(result.get("scores", {}).keys())

    def sort_key(dim: str) -> Tuple[int, str]:
        prefix_order = {"G": 0, "M": 1, "F": 2, "D": 2, "C": 2, "E": 2, "P": 3}
        return prefix_order.get(dim[:1], 9), dim

    return sorted(dims, key=sort_key)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_aggregate_outputs(output_dir: Path, samples: Sequence[Sample], results: Sequence[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dims = dimension_columns(results)
    sample_rows: List[Dict[str, object]] = []
    for result in results:
        row: Dict[str, object] = {
            "status": result.get("status", ""),
            "language": result.get("language", ""),
            "meeting_id": result.get("meeting_id", ""),
            "target_speaker": result.get("target_speaker", ""),
            "primary_role": role_from_result(result),
            "final_score": result.get("score_summary", {}).get("final_score", ""),
            "grade": result.get("score_summary", {}).get("grade", ""),
            "char_count": result.get("transcript_stats", {}).get("char_count", ""),
            "target_utterance_count": result.get("transcript_stats", {}).get(
                "target_utterance_count", ""
            ),
            "participation_ratio": result.get("transcript_stats", {}).get(
                "participation_ratio", ""
            ),
        }
        for dim in dims:
            row[dim] = result.get("scores", {}).get(dim, "")
        sample_rows.append(row)

    sample_fields = [
        "status",
        "language",
        "meeting_id",
        "target_speaker",
        "primary_role",
        "final_score",
        "grade",
        "participation_ratio",
        "char_count",
        "target_utterance_count",
        *dims,
    ]
    write_csv(output_dir / "samples.csv", sample_rows, sample_fields)

    meeting_groups: Dict[Tuple[str, str], List[dict]] = {}
    for result in results:
        key = (str(result.get("language", "")), str(result.get("meeting_id", "")))
        meeting_groups.setdefault(key, []).append(result)

    meeting_rows: List[Dict[str, object]] = []
    for (language, meeting_id), group in sorted(meeting_groups.items()):
        scores = [score for score in (safe_score(item) for item in group) if score is not None]
        meeting_rows.append(
            {
                "language": language,
                "meeting_id": meeting_id,
                "sample_count": len(group),
                "completed_count": len(scores),
                "failed_count": sum(1 for item in group if item.get("status") == "failed"),
                "mean_score": round(sum(scores) / len(scores), 3) if scores else "",
            }
        )
    write_csv(
        output_dir / "meetings.csv",
        meeting_rows,
        ["language", "meeting_id", "sample_count", "completed_count", "failed_count", "mean_score"],
    )

    completed_scores = [
        score for score in (safe_score(result) for result in results) if score is not None
    ]
    summary = {
        "dimension_schema_version": SCHEMA_VERSION,
        "sample_count": len(samples),
        "result_count": len(results),
        "completed_count": len(completed_scores),
        "failed_count": sum(1 for result in results if result.get("status") == "failed"),
        "dry_run_count": sum(1 for result in results if result.get("status") == "dry_run"),
        "mean_score": round(sum(completed_scores) / len(completed_scores), 3)
        if completed_scores
        else None,
        "meetings": meeting_rows,
    }
    atomic_write_json(output_dir / "summary.json", summary)

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "language": sample.language.value,
                        "meeting_id": sample.meeting_id,
                        "target_speaker": sample.target_speaker,
                        "offline_json": str(sample.offline_json),
                        "audio_path": str(sample.audio_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Wire up the judge endpoint pool before any score call.
    explicit = (
        [u.strip() for u in args.judge_endpoints.split(",") if u.strip()]
        if args.judge_endpoints
        else None
    )
    gemma4.configure(endpoints=config.judge_endpoints(explicit))

    samples = discover_samples(args)
    print(f"discovered {len(samples)} samples")

    builder = PromptBuilder()
    results: List[dict] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.sample_id}")
        try:
            result = run_or_resume_sample(sample, args, builder=builder)
        except Exception as exc:
            result = {
                "status": "failed",
                "dimension_schema_version": SCHEMA_VERSION,
                "sample_id": sample.sample_id,
                "language": sample.language.value,
                "meeting_id": sample.meeting_id,
                "target_speaker": sample.target_speaker,
                "offline_json": str(sample.offline_json),
                "audio_path": str(sample.audio_path),
                "calls": {},
                "scores": {},
                "score_summary": {},
                "errors": [
                    {
                        "stage": "sample",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ],
            }
            atomic_write_json(result_path(args.output_dir, sample), result)
            print(f"[failed] {sample.sample_id}: {exc}")
        results.append(result)
        if args.fail_fast and result.get("status") == "failed":
            print(f"stopping early due to --fail-fast after {sample.sample_id}")
            break

    write_aggregate_outputs(args.output_dir, samples, results)
    print(f"wrote outputs to {args.output_dir}")
    return 0 if all(result.get("status") != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
