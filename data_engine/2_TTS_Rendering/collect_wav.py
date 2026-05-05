import json
import os
import hashlib
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))


# Root of the per-dialogue rendered audio tree; override via --datasets-dir or env.
DATASETS_DIR = os.environ.get("DATASETS_DIR", "./tts_output/datasets")


def hash_string(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def get_language(text: str):
    text = text.replace(" ", "").replace("\n", "")
    chinese_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
    total = len(text)
    chinese_ratio = len(chinese_chars) / total if total > 0 else 0
    return "zh" if chinese_ratio > 0.3 else "en"


def process_line(item: dict):
    """解析单行并检查文件是否存在，返回结果 dict 或 None。"""
    try:
        request_id  = item["id"]
        # if not "hf" in item["id"]:
        #     return None
        item['audio'] = get_audio_path_from_id(request_id)
        output_path = os.path.join(DATASETS_DIR, item["audio"])
        stem = os.path.splitext(output_path)[0]

        if os.path.exists(stem + '.wav') and os.path.exists(stem + '.json'):
            return {
                "audio": stem + '.wav',
                "metadata": stem + '.json'
            }
        elif os.path.exists(stem + '.flac') and os.path.exists(stem + '.json'):
            return {
                "audio": stem + '.flac',
                "metadata": stem + '.json'
            }
        elif os.path.exists(stem + '_compressed.flac') and os.path.exists(stem + '_compressed.json'):
            return {
                "audio": stem + '_compressed.flac',
                "metadata": stem + '_compressed.json'
            }
        elif os.path.exists(stem + '_compressed.wav') and os.path.exists(stem + '_compressed.json'):
            return {
                "audio": stem + '_compressed.wav',
                "metadata": stem + '_compressed.json'
            }
        return None
    except Exception:
        traceback.print_exc()
        return None

def get_audio_path_from_id(request_id):
    request_id = str(request_id)
    int_request_id = int(request_id.split("_")[0])
    subdir1 = f"{int_request_id // 1000:03d}"
    return f"{subdir1}/{request_id}"


def main(jsonl_paths, out_path, max_tasks=None, num_workers=32):
    # ── 1. 读取所有行（顺序读取，速度本身很快）──────────────────────────
    items = {}
    for jsonl_path in jsonl_paths:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(tqdm(f, desc="Reading lines")):
                if max_tasks is not None and line_idx >= max_tasks:
                    break
                if line.strip():
                    item = json.loads(line.strip())
                    items[item["id"]] = item
    items = list(items.values())

    print(f"Read {len(items)} items, starting parallel file-existence check "
          f"with {num_workers} workers ...")
    
    if os.path.exists(out_path):
        # resume from existing output
        print(f"Output file {out_path} already exists. Loading existing results to avoid re-processing ...")
        existing_results = set()
        with open(out_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Loading existing results from {out_path}"):
                try:
                    item = json.loads(line.strip())
                    existing_results.add(item["audio"].split("/datasets/")[-1].replace('.wav', '').replace('_compressed.flac', '').replace('.flac', ''))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {out_path}: {line.strip()}")
        print(f"Loaded {len(existing_results)} existing results. Will skip these in processing.")
        print(list(existing_results)[:10])  # 打印前10个示例路径
        # Filter out items that have already been processed        
        items = [item for item in tqdm(items, desc="Filtering existing results") if get_audio_path_from_id(item["id"]) not in existing_results]
        print(f"{len(items)} items remain after filtering existing results.")

    # ── 2. 并行检查文件是否存在 ──────────────────────────────────────────
    output = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_line, item): i
                   for i, item in enumerate(tqdm(items, desc="Submitting tasks"))}
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Checking files"):
            result = future.result()
            if result is not None:
                output.append(result)

    # ── 3. 写出结果 ──────────────────────────────────────────────────────
    print(f"Collected {len(output)} valid audio files.")
    deduplicated_output = {}
    for item in output:
        audio_path = item["audio"]
        if audio_path not in deduplicated_output:
            deduplicated_output[audio_path] = item
    output = list(deduplicated_output.values())
    print(f"Deduplicated to {len(output)} unique audio files.")
    with open(out_path, 'a', encoding='utf-8') as f:
        for item in tqdm(output, desc="Writing output"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved to {out_path}")



if __name__ == '__main__':
    # Inputs: one or more dialogue-script JSONLs (output of stage 1).
    # Output: a JSONL listing the rendered {audio, metadata} pairs that were
    # actually found on disk under DATASETS_DIR.
    default_inputs = os.environ.get(
        "DIALOGUE_JSONLS",
        "./tts_output/full_duplex_dialogues.jsonl",
    )
    main(
        jsonl_paths=[p for p in default_inputs.split(",") if p.strip()],
        out_path=os.environ.get(
            "WAV_LIST_OUT",
            "./tts_output/full_duplex_dialogues_wav_lst.jsonl",
        ),
        max_tasks=None,
        num_workers=int(os.environ.get('NUM_WORKERS', 32)),
    )

