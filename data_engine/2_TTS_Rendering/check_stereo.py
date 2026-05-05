#!/usr/bin/env python3
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
from tqdm import tqdm


DEFAULT_DATA_PATH = "./tts_output/dialogues_stereo_compressed.jsonl"


def iter_jsonl(path, max_items=None):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if max_items is not None and line_no > max_items:
                break

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, None, f"invalid json: {exc}"
                continue

            yield line_no, item, None


def resolve_audio_path(data_path, audio_path):
    if os.path.isabs(audio_path):
        return audio_path
    return os.path.join(os.path.dirname(data_path), audio_path)


def check_one(args):
    line_no, item, parse_error, data_path, expected_channels = args

    if parse_error is not None:
        return {
            "ok": False,
            "line_no": line_no,
            "status": "invalid_json",
            "error": parse_error,
        }

    rel_path = item.get("path")
    if not rel_path:
        return {
            "ok": False,
            "line_no": line_no,
            "status": "missing_path_field",
            "error": "record has no non-empty `path` field",
        }

    audio_path = resolve_audio_path(data_path, rel_path)
    if not os.path.exists(audio_path):
        return {
            "ok": False,
            "line_no": line_no,
            "status": "missing_file",
            "path": rel_path,
            "audio_path": audio_path,
            "error": "audio file does not exist",
        }

    try:
        info = sf.info(audio_path)
    except Exception as exc:
        return {
            "ok": False,
            "line_no": line_no,
            "status": "read_error",
            "path": rel_path,
            "audio_path": audio_path,
            "error": repr(exc),
        }

    return {
        "ok": info.channels == expected_channels,
        "line_no": line_no,
        "status": "ok" if info.channels == expected_channels else "bad_channels",
        "path": rel_path,
        "audio_path": audio_path,
        "channels": info.channels,
        "expected_channels": expected_channels,
        "samplerate": info.samplerate,
        "frames": info.frames,
        "duration": info.duration,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether every audio file referenced by a jsonl is stereo."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"Input jsonl path. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=2,
        help="Expected channel count. Default: 2",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel file-info readers. Default: 16",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Only check the first N jsonl lines. Default: check all.",
    )
    parser.add_argument(
        "--bad-output",
        default=None,
        help="Optional jsonl path for non-stereo/missing/error records.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Print at most this many bad records. Default: 20",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    items = [
        (line_no, item, error, args.data_path, args.expected_channels)
        for line_no, item, error in iter_jsonl(args.data_path, max_items=args.max_items)
    ]

    bad_records = []
    total_duration = 0.0
    status_counts = {}
    executor = None

    if args.workers <= 1:
        iterator = map(check_one, items)
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        iterator = executor.map(check_one, items)

    try:
        for result in tqdm(iterator, total=len(items), desc="Checking audio channels"):
            status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
            if result.get("duration") is not None:
                total_duration += result["duration"]
            if not result["ok"]:
                bad_records.append(result)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    print("\nCheck complete")
    print(f"  jsonl:             {args.data_path}")
    print(f"  checked records:   {len(items)}")
    print(f"  expected channels: {args.expected_channels}")
    print(f"  total duration:    {total_duration / 3600:.2f} h")
    print(f"  bad records:       {len(bad_records)}")

    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    if bad_records:
        print(f"\nFirst {min(args.show, len(bad_records))} bad records:")
        for record in bad_records[: args.show]:
            print(json.dumps(record, ensure_ascii=False))

    if args.bad_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.bad_output)), exist_ok=True)
        with open(args.bad_output, "w", encoding="utf-8") as f:
            for record in bad_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nSaved bad records to: {args.bad_output}")

    if bad_records:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
