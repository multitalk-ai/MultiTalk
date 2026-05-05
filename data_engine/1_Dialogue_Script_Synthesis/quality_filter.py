"""
质量过滤模块
对 LLM 生成的对话进行多层自动过滤
适配两阶段架构：participants 中包含真实名字
"""

import contextlib
import json
import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

# 匹配 silence turn: speaker=="silence", text like "[2s]", "[3.5s]"
SILENCE_DURATION_PATTERN = re.compile(r'^\[\d+(?:\.\d+)?s\]$')

VALID_EMOTIONS = {"happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"}
VALID_SPEEDS = {"very_slow", "slow", "normal", "fast", "very_fast"}
VALID_VOLUMES = {"whisper", "soft", "normal", "loud", "shout"}


def _is_silence_turn(turn: dict) -> bool:
    """判断是否为全局静音 turn"""
    return turn.get("speaker") == "silence" and SILENCE_DURATION_PATTERN.match(turn.get("text", "").strip())


def check_format(record: Dict) -> Tuple[bool, str]:
    """格式检查：JSON 结构完整，字段正确"""
    if "dialogue" not in record or not isinstance(record["dialogue"], list):
        return False, "missing or invalid 'dialogue' field"
    if "participants" not in record or not isinstance(record["participants"], list):
        return False, "missing or invalid 'participants' field"

    for i, turn in enumerate(record["dialogue"]):
        if not isinstance(turn, dict):
            return False, f"turn {i} is not a dict"
        if "speaker" not in turn:
            return False, f"turn {i} missing 'speaker'"
        if "text" not in turn:
            return False, f"turn {i} missing 'text'"
        if not isinstance(turn["text"], str) or not turn["text"].strip():
            return False, f"turn {i} has empty text"
        # overlap 字段如果存在必须是 bool
        if "overlap" in turn and not isinstance(turn["overlap"], bool):
            return False, f"turn {i} has invalid 'overlap' field (must be bool)"

        # voice 字段校验（silence turn 不需要）
        if not (turn.get("speaker") == "silence"):
            if "voice" not in turn or not isinstance(turn["voice"], dict):
                return False, f"turn {i} missing 'voice' field"
            voice = turn["voice"]
            if voice.get("emotion") not in VALID_EMOTIONS:
                return False, f"turn {i} invalid emotion: '{voice.get('emotion')}'"
            intensity = voice.get("emotion_intensity")
            if not isinstance(intensity, (int, float)) or intensity < 0 or intensity > 1:
                return False, f"turn {i} invalid emotion_intensity: {intensity}"
            if voice.get("speed") not in VALID_SPEEDS:
                return False, f"turn {i} invalid speed: '{voice.get('speed')}'"
            if voice.get("volume") not in VALID_VOLUMES:
                return False, f"turn {i} invalid volume: '{voice.get('volume')}'"
            if "dialect" not in voice or not isinstance(voice["dialect"], str):
                return False, f"turn {i} missing or invalid 'dialect' in voice"

    for i, p in enumerate(record["participants"]):
        if not isinstance(p, dict):
            return False, f"participant {i} is not a dict"
        if "name" not in p or "role" not in p:
            return False, f"participant {i} missing 'name' or 'role'"
        if p.get("gender") not in ("male", "female"):
            return False, f"participant {i} missing or invalid 'gender' (must be 'male' or 'female')"

    return True, ""


def check_length(record: Dict, min_turns: int = 4, max_turns: int = 60,
                 max_words_per_turn: int = 200) -> Tuple[bool, str]:
    """长度检查（适配口语对话：允许短回应、[silence]、连续发言）"""
    dialogue = record["dialogue"]

    if len(dialogue) < min_turns:
        return False, f"too few turns: {len(dialogue)} < {min_turns}"
    if len(dialogue) > max_turns:
        return False, f"too many turns: {len(dialogue)} > {max_turns}"

    for i, turn in enumerate(dialogue):
        text = turn["text"].strip()
        # silence turn 是合法的特殊标记
        if _is_silence_turn(turn):
            continue
        # 英文按空格分词，中文按字符计数
        word_count = len(text.split()) + len(re.findall(r'[\u4e00-\u9fff]', text))
        if word_count > max_words_per_turn:
            return False, f"turn {i} too long: {word_count} words"

    # 连续同一人发言不超过 6 轮（口语中短连续发言很常见）
    consecutive = 1
    for i in range(1, len(dialogue)):
        if _is_silence_turn(dialogue[i]):
            consecutive = 1
            continue
        if dialogue[i]["speaker"] == dialogue[i - 1]["speaker"] and not _is_silence_turn(dialogue[i - 1]):
            consecutive += 1
            if consecutive > 6:
                return False, f"speaker '{dialogue[i]['speaker']}' speaks {consecutive} times in a row"
        else:
            consecutive = 1

    return True, ""


def check_roles(record: Dict) -> Tuple[bool, str]:
    """角色检查：speaker 名字与 participants 匹配，每个角色发言次数合理"""
    dialogue = record["dialogue"]
    participants = record["participants"]

    # 从 participants 提取合法名字
    user_names = set()
    assistant_name = None
    for p in participants:
        if p["role"] == "user":
            user_names.add(p["name"])
        elif p["role"] == "assistant":
            assistant_name = p["name"]

    if assistant_name is None:
        return False, "no assistant in participants"

    all_valid_names = user_names | {assistant_name}
    # 统计发言次数时排除 silence turn
    speaker_counts = Counter(turn["speaker"] for turn in dialogue if not _is_silence_turn(turn))

    # 检查 speaker 标签合法性
    for speaker in speaker_counts:
        if speaker not in all_valid_names:
            return False, f"unexpected speaker: '{speaker}' (valid: {all_valid_names})"

    # 每个用户至少说 2 轮
    for name in user_names:
        if speaker_counts.get(name, 0) < 2:
            return False, f"'{name}' speaks only {speaker_counts.get(name, 0)} times (min 2)"

    # AI 助手至少说 1 轮
    if speaker_counts.get(assistant_name, 0) < 1:
        return False, f"assistant '{assistant_name}' never speaks"

    return True, ""


def check_content(record: Dict) -> Tuple[bool, str]:
    """内容检查：重复、套话、AI身份泄露等"""
    dialogue = record["dialogue"]

    # 检查相邻发言是否重复（跳过 silence turn）
    for i in range(1, len(dialogue)):
        if _is_silence_turn(dialogue[i - 1]) or _is_silence_turn(dialogue[i]):
            continue
        prev_text = dialogue[i - 1]["text"].strip().lower()
        curr_text = dialogue[i]["text"].strip().lower()
        if prev_text == curr_text:
            return False, f"turn {i} is identical to turn {i-1}"

    # 找出 assistant 的名字
    assistant_name = None
    for p in record.get("participants", []):
        if p["role"] == "assistant":
            assistant_name = p["name"]
            break

    # 检查 AI 套话比例
    ai_turns = [t for t in dialogue
                if t["speaker"] == assistant_name and not _is_silence_turn(t)]
    if ai_turns:
        cliche_patterns = [
            r"^as an ai",
            r"^i understand your feelings",
            r"^i'm here for you",
            r"^that's a great question",
            r"^i appreciate you sharing",
            r"^thank you for sharing",
            r"^i hear you",
        ]
        cliche_count = 0
        for turn in ai_turns:
            text_lower = turn["text"].strip().lower()
            for pattern in cliche_patterns:
                if re.match(pattern, text_lower):
                    cliche_count += 1
                    break

        cliche_ratio = cliche_count / len(ai_turns)
        if cliche_ratio > 0.5:
            return False, f"AI cliche ratio too high: {cliche_ratio:.2f}"

    # 检查是否有明显的拒绝/免责声明/AI身份泄露
    refusal_patterns = [
        r"i cannot",
        r"i'm not able to",
        r"as a language model",
        r"i don't have personal",
        r"作为一个AI",
        r"作为语言模型",
        r"作为AI助手",
        r"我是一个AI",
        r"i'm an ai",
        r"as an ai assistant",
    ]
    for turn in dialogue:
        text_lower = turn["text"].strip().lower()
        if _is_silence_turn(turn):
            continue
        for pattern in refusal_patterns:
            if re.search(pattern, text_lower):
                return False, f"contains refusal/disclaimer/AI identity: '{pattern}'"

    return True, ""


def filter_single(record: Dict) -> Tuple[bool, List[str]]:
    """
    对单条记录执行所有过滤检查

    Returns:
        (通过与否, 失败原因列表)
    """
    reasons = []

    ok, reason = check_format(record)
    if not ok:
        reasons.append(f"format: {reason}")

    if ok:  # 格式通过才继续检查
        ok2, reason2 = check_length(record)
        if not ok2:
            reasons.append(f"length: {reason2}")

        ok3, reason3 = check_roles(record)
        if not ok3:
            reasons.append(f"roles: {reason3}")

        ok4, reason4 = check_content(record)
        if not ok4:
            reasons.append(f"content: {reason4}")

    return len(reasons) == 0, reasons


def filter_batch(
    input_path: str,
    output_path: str,
    rejected_path: Optional[str] = None,
) -> Dict[str, int]:
    """
    批量过滤 JSONL 文件

    Args:
        input_path: 输入 JSONL 路径（api_caller 的输出）
        output_path: 过滤后的输出 JSONL 路径
        rejected_path: 被拒绝记录的输出路径（可选，用于分析）

    Returns:
        统计信息
    """
    stats = {"total": 0, "passed": 0, "rejected": 0, "reject_reasons": Counter()}

    with contextlib.ExitStack() as stack:
        f_in = stack.enter_context(open(input_path, "r"))
        f_out = stack.enter_context(open(output_path, "w"))
        f_rejected = (
            stack.enter_context(open(rejected_path, "w")) if rejected_path else None
        )

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["rejected"] += 1
                stats["reject_reasons"]["json_parse_error"] += 1
                continue

            passed, reasons = filter_single(record)

            if passed:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["passed"] += 1
            else:
                stats["rejected"] += 1
                for r in reasons:
                    category = r.split(":")[0]
                    stats["reject_reasons"][category] += 1
                if f_rejected:
                    record["_reject_reasons"] = reasons
                    f_rejected.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"Filter results: total={stats['total']}, "
        f"passed={stats['passed']}, rejected={stats['rejected']}"
    )
    logger.info(f"Reject reasons: {dict(stats['reject_reasons'])}")

    return stats
