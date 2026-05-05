#!/usr/bin/env python3
"""
将句子级别的对齐 manifest 升级为词级别对齐（基于 MFA）。

支持两种输入记录格式（见 parse_audio_record）：
  - legacy: ``audio`` 字段是 ``[audio_info_json, text_info_json]`` 列表；
  - metadata_file: ``audio`` 是音频路径字符串，并配合同级 ``metadata`` 字段
    指向同记录的 JSON sidecar（即 stage 2 ``gen_duration_and_alignments.py``
    的输出）。

优化：
  - batch_size:  每次 MFA 子进程调用处理的片段数，跨样本打包
  - line_batch:  每次从文件读取并放入内存的最大样本行数，避免一次性加载全量数据
  - 同 batch 内相同音频文件只读取和重采样一次
  - worker_init: 每个 worker 进程只初始化一次 MFAAligner
  - 按语言分桶：同一 batch 内只含单一语言，避免模型错配
  - 读取阶段每行只调用一次 parse_audio_record，磁盘 I/O 减半
"""
import json
import sys
import os
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count, Lock as MPLock
import traceback
from typing import List, Dict, Any, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from transcript_mfa_aligner import MFAAligner, detect_language


# ============================================================
# Worker 级别 aligner 复用
# ============================================================

_worker_aligner:    Optional[MFAAligner] = None
_worker_first_done: bool                 = False
_global_init_lock                        = None


def worker_init(init_lock, mfa_root: Optional[str]):
    global _worker_aligner, _global_init_lock, _worker_first_done
    _global_init_lock  = init_lock
    _worker_first_done = False
    _worker_aligner    = MFAAligner(keep_temp=False, mfa_root=mfa_root)


def process_segment_batch_worker(batch_task):
    global _worker_first_done
    if not _worker_first_done:
        with _global_init_lock:
            result = process_segment_batch(batch_task, aligner=_worker_aligner)
        _worker_first_done = True
        return result
    return process_segment_batch(batch_task, aligner=_worker_aligner)


# ============================================================
# JSON 工具函数
# ============================================================

def load_json_maybe_file(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str):
        raise TypeError(f"不支持的 JSON 输入类型: {type(value)}")
    stripped = value.strip()
    if not stripped:
        raise ValueError("JSON 输入为空")
    if stripped.startswith('{') or stripped.startswith('['):
        return json.loads(stripped)
    path = Path(stripped)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(stripped)


def dump_json_or_file(value, obj):
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and not (stripped.startswith('{') or stripped.startswith('[')):
            path = Path(stripped)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return value
    return json.dumps(obj, ensure_ascii=False)


# ============================================================
# 数据解析
# ============================================================

def parse_audio_record(data):
    parsed = {
        'format': None, 'sample_id': None, 'audio_path': None,
        'text_info': None, 'raw_text_field': None, 'metadata_path': None,
    }
    audio_field = data.get('audio')

    if isinstance(audio_field, list) and len(audio_field) >= 2:
        audio_info  = load_json_maybe_file(audio_field[0])
        text_info   = load_json_maybe_file(audio_field[1])
        model_audio = None
        for item in audio_info.get('path', []):
            if item[0] == 'MODEL':
                model_audio = item[1]
                break
        parsed.update({
            'format': 'legacy', 'sample_id': model_audio,
            'audio_path': model_audio, 'text_info': text_info,
            'raw_text_field': audio_field[1],
        })
        return parsed

    if isinstance(audio_field, str) and 'metadata' in data:
        metadata_path = data['metadata']
        text_info     = load_json_maybe_file(metadata_path)
        sample_id     = text_info.get('id') or audio_field
        parsed.update({
            'format': 'metadata_file', 'sample_id': sample_id,
            'audio_path': audio_field, 'text_info': text_info,
            'raw_text_field': metadata_path, 'metadata_path': metadata_path,
        })
        return parsed

    raise ValueError("不支持的数据格式：既不是旧版 audio 列表，也不是新版 audio/metadata 结构")


def extract_alignments(text_info):
    if not isinstance(text_info, dict):
        raise ValueError("文本信息不是字典")

    if 'text' in text_info and isinstance(text_info['text'], dict):
        normalized = []
        for alignment in (text_info['text'].get('alignments') or []):
            if not isinstance(alignment, list) or len(alignment) < 2:
                continue
            text = alignment[0].strip()
            if not text or not text.strip(".。"):
                continue
            time_range = alignment[1]
            if not time_range or len(time_range) < 2:
                continue
            normalized.append({
                'text':    text,
                'start':   time_range[0],
                'end':     time_range[1],
                'speaker': alignment[2] if len(alignment) > 2 else None,
                'raw':     alignment,
            })
        return normalized, 'legacy'

    normalized = []
    for segment in (text_info.get('metadata:transcript') or []):
        if not isinstance(segment, dict):
            continue
        text  = segment.get('transcript', '').strip()
        start = segment.get('start')
        end   = segment.get('end')
        if not text or not text.strip(".。") or start is None or end is None:
            continue
        normalized.append({
            'text':    text,
            'start':   start,
            'end':     end,
            'speaker': 'Assistant',
            'raw':     segment,
        })
    return normalized, 'metadata_file'


def apply_word_alignments(text_info, source_type, segments, aligned_results):
    if source_type == 'legacy':
        new_alignments = []
        for segment, result in zip(segments, aligned_results):
            speaker = segment.get('speaker')
            if result and result.get('words'):
                for word, (ws, we) in result['words']:
                    new_alignments.append([word, [ws, we], speaker])
            else:
                new_alignments.append(segment['raw'])
        text_info.setdefault('text', {})['alignments'] = new_alignments
        return text_info

    if source_type == 'metadata_file':
        updated = []
        for idx, segment in enumerate(text_info.get('metadata:transcript') or []):
            if idx < len(aligned_results):
                result = aligned_results[idx]
                if result and result.get('words'):
                    segment = dict(segment)
                    segment['words'] = [
                        {'word': w, 'start': s, 'end': e}
                        for w, (s, e) in result['words']
                    ]
                    for key in ('start', 'end', 'transcript'):
                        if result.get(key) is not None:
                            segment[key] = result[key]
            updated.append(segment)
        text_info['metadata:transcript'] = updated
        return text_info

    raise ValueError(f"未知的 source_type: {source_type}")


# ============================================================
# 缓存工具
# ============================================================

def load_processed_ids(cache_dir):
    processed_ids = set()
    if not Path(cache_dir).exists():
        return processed_ids
    for cache_file in Path(cache_dir).glob("processed_ids*.txt"):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed_ids.add(line)
        except Exception as e:
            print(f"警告: 加载缓存文件 {cache_file} 失败: {e}")
    return processed_ids


def save_processed_id(cache_file, sample_id):
    with open(cache_file, 'a', encoding='utf-8') as f:
        f.write(f"{sample_id}\n")
        f.flush()


# ============================================================
# Batch 处理
# ============================================================

def process_segment_batch(batch_task, aligner=None):
    batch_idx, batch_items = batch_task

    if aligner is None:
        aligner = MFAAligner(keep_temp=False)

    segment_requests = [item[2] for item in batch_items]

    try:
        align_results = aligner.align_batch(segment_requests)
    except Exception as e:
        print(f"[Batch {batch_idx}] align_batch 异常: {e}\n{traceback.format_exc()}")
        align_results = [None] * len(segment_requests)

    return (
        batch_idx,
        [
            (sample_idx, seg_idx, result)
            for (sample_idx, seg_idx, _), result in zip(batch_items, align_results)
        ],
    )

def add_alignments_to_metadata(metadata):
    # 假设 metadata 中有一个 "segments" 字段是一个列表，每个元素包含 "start_time" 和 "end_time"
    alignments = []
    for segment in metadata["metadata:transcript"]:
        for word_info in segment["words"]:
            alignments.append([
                word_info["word"],
                [word_info["start"], word_info["end"]], 
                "SPEAKER_MAIN"
                ])

    metadata["alignments"] = alignments

    return metadata

def assemble_sample(record):
    data        = record['data']
    parsed      = record['parsed']
    text_info   = record['text_info']
    segments    = record['segments']
    source_type = record['source_type']
    results     = record['results']

    text_info = apply_word_alignments(text_info, source_type, segments, results)
    if len(text_info["metadata:transcript"][0]["words"]) > 0 and "alignments" not in text_info:
        text_info = add_alignments_to_metadata(text_info)
    if parsed['format'] == 'legacy':
        data['audio'][1] = dump_json_or_file(parsed['raw_text_field'], text_info)
    elif parsed['format'] == 'metadata_file':
        data['metadata'] = dump_json_or_file(parsed['raw_text_field'], text_info)
    else:
        raise ValueError(f"未知格式: {parsed['format']}")

    return json.dumps(data, ensure_ascii=False) + '\n'


# ============================================================
# 主流程
# ============================================================

def process_jsonl(
    input_path,
    output_path   = None,
    resume        = True,
    num_workers   = None,
    rank          = None,
    world_size    = None,
    max_num_tasks = None,
    batch_size    = 32,
    line_batch    = 500,
    mfa_root      = None,
):
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    if num_workers is None:
        num_workers = cpu_count()

    input_file = Path(input_path)
    if output_path is None:
        suffix = f"_word_aligned_rank{rank}" if world_size > 1 else "_word_aligned"
        output_path = input_file.parent / f"{input_file.stem}{suffix}{input_file.suffix}"

    cache_dir = input_file.parent / f".cache_{input_file.stem}"
    cache_dir.mkdir(exist_ok=True)
    cache_file = (cache_dir / f"processed_ids_rank{rank}.txt"
                  if world_size > 1 else cache_dir / "processed_ids.txt")

    processed_ids = set()
    if resume:
        processed_ids = load_processed_ids(cache_dir)
        if processed_ids:
            print(f"[Rank {rank}] 从缓存加载了 {len(processed_ids)} 个已处理的ID")

    print(f"[Rank {rank}/{world_size}] 开始处理: {input_path}")
    print(f"[Rank {rank}] workers={num_workers}, batch_size={batch_size}, line_batch={line_batch}")
    print(f"[Rank {rank}] 输出: {output_path}")

    completed_count   = 0
    error_count       = 0
    total_samples     = 0
    global_line_num   = 0
    global_task_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:

        if num_workers > 1:
            _init_lock = MPLock()
            pool = Pool(
                processes=num_workers,
                initializer=worker_init,
                initargs=(_init_lock, mfa_root),
            )
        else:
            pool    = None
            aligner = MFAAligner(keep_temp=False, mfa_root=mfa_root)

        try:
            chunk_num = 0
            with tqdm(desc=f"[Rank {rank}] 已读取样本", unit="sample") as pbar_read:
                while True:
                    if max_num_tasks is not None and global_task_count >= max_num_tasks:
                        break

                    # ── Phase 1: 分批读取至多 line_batch 个新样本 ──────────
                    sample_records: List[Dict[str, Any]] = []
                    all_segments:   List[tuple]          = []
                    reached_limit = False

                    for line in f_in:
                        global_line_num += 1

                        if global_line_num % world_size != rank:
                            continue

                        line = line.strip()
                        if not line:
                            continue

                        # ── 每行只调用一次 parse_audio_record，
                        #    原先先 get_sample_id（内部读文件）再 parse（再读文件）
                        #    导致每行触发两次磁盘 I/O，此处合并为一次 ──────────
                        try:
                            data      = json.loads(line)
                            parsed    = parse_audio_record(data)
                            sample_id = parsed['sample_id']
                        except Exception as e:
                            print(f"[Rank {rank}] Line {global_line_num}: 解析失败 - {e}")
                            continue

                        if sample_id is None:
                            print(f"[Rank {rank}] Line {global_line_num}: 无法提取ID，跳过")
                            continue
                        if sample_id in processed_ids:
                            continue

                        try:
                            text_info             = parsed['text_info']
                            segments, source_type = extract_alignments(text_info)
                        except Exception as e:
                            print(f"[Rank {rank}] Line {global_line_num}: 提取对齐失败 - {e}")
                            continue

                        if not segments:
                            continue

                        # 已含词级对齐则跳过（安全访问，避免 KeyError）
                        transcript = text_info.get("metadata:transcript") or []
                        if transcript and transcript[0].get("words"):
                            print(f"[Rank {rank}] Line {global_line_num}: 已包含词级对齐信息，跳过")
                            continue

                        sample_idx = len(sample_records)
                        sample_records.append({
                            'line_num':    global_line_num,
                            'sample_id':   sample_id,
                            'data':        data,
                            'parsed':      parsed,
                            'text_info':   text_info,
                            'segments':    segments,
                            'source_type': source_type,
                            'results':     [None] * len(segments),
                            'pending':     len(segments),
                        })

                        for seg_idx, seg in enumerate(segments):
                            all_segments.append((sample_idx, seg_idx, {
                                'audio_path': parsed['audio_path'],
                                'text':       seg['text'],
                                'start':      seg['start'],
                                'end':        seg['end'],
                                'language':   None,
                            }))

                        global_task_count += 1
                        pbar_read.update(1)

                        if len(sample_records) >= line_batch:
                            break
                        if max_num_tasks is not None and global_task_count >= max_num_tasks:
                            reached_limit = True
                            break

                    if not sample_records:
                        break

                    chunk_num     += 1
                    total_samples += len(sample_records)

                    # ── Phase 2: 语言检测 + 按语言分桶 + 分 MFA batch ──────
                    for _, _, req in all_segments:
                        if not req.get('language'):
                            req['language'] = detect_language(req.get('text', ''))

                    lang_buckets: Dict[str, List] = defaultdict(list)
                    for item in all_segments:
                        lang_buckets[item[2]['language']].append(item)

                    batches = []
                    for lang_items in lang_buckets.values():
                        for i in range(0, len(lang_items), batch_size):
                            batches.append((len(batches), lang_items[i:i + batch_size]))

                    total_segs    = len(all_segments)
                    total_batches = len(batches)
                    lang_summary  = {lang: len(items) for lang, items in lang_buckets.items()}

                    print(
                        f"\n[Rank {rank}] Chunk {chunk_num}: "
                        f"{len(sample_records)} 个样本 / {total_segs} 个片段 / "
                        f"{total_batches} 个 MFA batch | 语言: {lang_summary}"
                    )

                    # ── Phase 3: 处理 MFA batch，样本完成后立即写出 ─────────
                    def handle_batch_result(batch_result):
                        nonlocal completed_count, error_count
                        _, output_items = batch_result
                        for s_idx, sg_idx, result in output_items:
                            rec                    = sample_records[s_idx]
                            rec['results'][sg_idx] = result
                            rec['pending']        -= 1
                            if rec['pending'] == 0:
                                try:
                                    f_out.write(assemble_sample(rec))
                                    f_out.flush()
                                    save_processed_id(cache_file, rec['sample_id'])
                                    completed_count += 1
                                except Exception as e:
                                    error_count += 1
                                    tqdm.write(
                                        f"[Rank {rank}] 样本 {rec['sample_id']} 组装失败: {e}"
                                    )

                    if pool is not None:
                        with tqdm(
                            total=total_segs,
                            desc=f"[Rank {rank}] Chunk {chunk_num} 片段进度",
                            unit="seg",
                        ) as pbar:
                            for batch_result in pool.imap(
                                process_segment_batch_worker, batches, chunksize=1
                            ):
                                handle_batch_result(batch_result)
                                pbar.update(len(batch_result[1]))
                                pbar.set_postfix({
                                    "样本完成": completed_count,
                                    "失败":     error_count,
                                })
                    else:
                        with tqdm(
                            total=total_segs,
                            desc=f"[Rank {rank}] Chunk {chunk_num} 片段进度",
                            unit="seg",
                        ) as pbar:
                            for batch_task in batches:
                                batch_result = process_segment_batch(batch_task, aligner)
                                handle_batch_result(batch_result)
                                pbar.update(len(batch_task[1]))
                                pbar.set_postfix({
                                    "样本完成": completed_count,
                                    "失败":     error_count,
                                })

                    if reached_limit:
                        break

        finally:
            if pool is not None:
                pool.close()
                pool.join()

    print(f"\n[Rank {rank}] 处理完成:")
    print(f"  处理样本总数: {total_samples}")
    print(f"  成功写出: {completed_count}")
    print(f"  组装失败: {error_count}")
    print(f"  输出文件: {output_path}")
    print(f"  缓存文件: {cache_file}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='升级句子级 manifest 为词级别对齐 (MFA)')
    parser.add_argument(
        'input', nargs='?', default=os.environ.get('INPUT_PATH'),
        help='输入 JSONL 文件路径（默认从环境变量 INPUT_PATH 获取）'
    )
    parser.add_argument('-o', '--output',     help='输出 JSONL 文件路径')
    parser.add_argument('-j', '--jobs',       type=int, default=16,
                        help='并行进程数（默认16）')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='每次 MFA 调用的片段数（默认32）')
    parser.add_argument('-l', '--line-batch', type=int, default=500,
                        help='每次读取并处理的最大样本行数（默认500），控制内存峰值')
    parser.add_argument('--no-resume',        action='store_true')
    parser.add_argument('--rank',             type=int, default=None)
    parser.add_argument('--world-size',       type=int, default=None)
    parser.add_argument('--max-tasks',        type=int, default=None)
    parser.add_argument(
        '--mfa-root', default=None,
        help='包含 pretrained_models/ 的 MFA 根目录（默认 ~/Documents/MFA）'
    )

    args = parser.parse_args()

    process_jsonl(
        args.input,
        args.output,
        resume        = not args.no_resume,
        num_workers   = args.jobs,
        rank          = args.rank,
        world_size    = args.world_size,
        max_num_tasks = args.max_tasks,
        batch_size    = args.batch_size,
        line_batch    = args.line_batch,
        mfa_root      = args.mfa_root,
    )