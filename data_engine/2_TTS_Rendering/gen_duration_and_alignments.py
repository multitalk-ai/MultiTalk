import multiprocessing as mp
import threading
import json
import os
import traceback
from tqdm import tqdm

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

def detect_language(text: str) -> str:
    """
    判断字符串主要语言是中文还是英文。
    返回: 'zh', 'en'
    """

    zh_count = 0
    en_count = 0

    for char in text:
        code = ord(char)
        # 中文字符范围（基本汉字 + 扩展）
        if (0x4E00 <= code <= 0x9FFF or   # 基本汉字
            0x3400 <= code <= 0x4DBF or   # 扩展A
            0x20000 <= code <= 0x2A6DF):  # 扩展B
            zh_count += 1
        # 英文字母
        elif char.isalpha() and code < 128:
            en_count += 1

    total = zh_count + en_count


    zh_ratio = zh_count / total
    en_ratio = en_count / total

    # print(f"Text: {text!r}, zh_count: {zh_count}, en_count: {en_count}, zh_ratio: {zh_ratio:.2f}, en_ratio: {en_ratio:.2f}")
    if zh_ratio > en_ratio:
        return 'zh'
    else:
        return 'en'

def tts_worker(worker_id, task_queue, result_queue):
    """检查 TTS 输出文件是否已存在，并将结果写入 result_queue。"""
    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            output_path = task['audio']
            metadata_path = task['metadata']

            if os.path.exists(output_path) and os.path.exists(metadata_path):
                metadata = json.load(open(metadata_path, 'r', encoding='utf-8'))
                assistant_voice = metadata["voice"]["Assistant"]
                if len(metadata["metadata:transcript"][0]["words"]) > 0:
                    if "alignments" not in metadata:
                        metadata = add_alignments_to_metadata(metadata)
                        print(f"Worker {worker_id}: Added alignments to metadata for {output_path}")
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                    all_text = "".join(transcript["transcript"] for transcript in metadata["metadata:transcript"])
                    result_queue.put({
                        "status":    "found",
                        "output_path": output_path,
                        "duration":   metadata["duration"],
                        "voice":      assistant_voice,
                        "language":   detect_language(all_text),
                    })
                else:
                    result_queue.put({
                        "status":    "missing",
                        "output_path": output_path
                    })
            else:
                result_queue.put({
                    "status":    "missing",
                    "output_path": output_path
                })

        except Exception:
            result_queue.put({
                "status":    "error",
                "output_path": output_path,
                "error":     traceback.format_exc(),
            })


def feed_tasks(task_queue, tasks, num_workers):
    """在后台线程中投递任务，避免 task_queue 满时阻塞主进程。"""
    for task in tasks:
        task_queue.put(task)
    for _ in range(num_workers):
        task_queue.put(None)


def main(jsonl_path, output_dir, num_workers=4, max_tasks=None):
    # 分布式环境变量只读一次
    world_size = int(os.environ.get('WORLD_SIZE_V', 1))
    rank       = int(os.environ.get('RANK_V', 0))

    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line_idx % world_size != rank:
                continue

            tasks.append(json.loads(line))

            if max_tasks is not None and len(tasks) >= max_tasks:
                break

    print(f"Loaded {len(tasks)} tasks")
    if not tasks:
        return []

    task_queue   = mp.Queue(maxsize=num_workers * 2)
    result_queue = mp.Queue()

    workers = []
    for wid in range(num_workers):
        p = mp.Process(target=tts_worker, args=(wid, task_queue, result_queue), daemon=True)
        p.start()
        workers.append(p)

    feeder = threading.Thread(target=feed_tasks, args=(task_queue, tasks, num_workers), daemon=True)
    feeder.start()

    results = []
    for _ in tqdm(range(len(tasks)), desc="Collecting"):
        results.append(result_queue.get())

    feeder.join()
    for p in workers:
        p.join()

    found_count   = sum(1 for r in results if r['status'] == 'found')
    missing_count = sum(1 for r in results if r['status'] == 'missing')
    error_count   = sum(1 for r in results if r['status'] == 'error')

    print(f"\nCollection complete:")
    print(f"  Found:   {found_count}")
    print(f"  Missing: {missing_count}")
    print(f"  Errors:  {error_count}")

    # 只返回已找到的文件列表
    wav_list = [r for r in results if r['status'] == 'found']
    return wav_list


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    input_path  = os.environ.get('INPUT_PATH',  "./tts_output/full_duplex_dialogues_wav_lst.jsonl")
    output_dir  = os.environ.get('OUTPUT_DIR',  "./tts_output/datasets")
    output_jsonl_path = os.environ.get('OUTPUT_JSONL', "./tts_output/durations_compressed.jsonl")
    num_workers = int(os.environ.get('NUM_WORKERS', 32))

    wav_list = main(jsonl_path=input_path, output_dir=output_dir, num_workers=num_workers, max_tasks=None)
    print(f"\nCollected {len(wav_list)} wav files.")

    sum_duration = sum(item["duration"] for item in wav_list)
    print(f"Total duration of collected wav files: {sum_duration:.2f} seconds={sum_duration/3600:.2f} hours")
    system_en = "You are a helpful, knowledgeable assistant. Respond in English."
    system_zh = "你是一个乐于助人、知识渊博的助手。使用中文回答。"
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in tqdm(wav_list, desc="Writing output"):
            item = {
                "path": item["output_path"].split("/datasets/")[-1],
                "duration": item["duration"],
                "voice": item["voice"],
                "system": system_zh if item["language"] == "zh" else system_en,
                }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved collected wav list to {output_jsonl_path}")