"""
Claude API 批量调用模块（两阶段）

阶段一：生成场景 + 角色画像
阶段二：基于阶段一输出生成对话
支持并发调用、重试、断点续传、结果保存
"""

import os
import random
import json
import time
import logging
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from anthropic import Anthropic
from dataclasses import asdict
from scenario_config import ScenarioConfig
from prompt_builder import (
    STAGE1_SYSTEM_PROMPT, STAGE2_SYSTEM_PROMPT,
    build_stage1_messages, build_stage2_messages,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 每百万 token 的价格（美元）
MODEL_PRICING = {
    "claude-opus-4-6-20250514": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # aliases
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """计算单次调用的费用（美元）"""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # 未知模型，尝试模糊匹配
        for key, val in MODEL_PRICING.items():
            if key in model or model in key:
                pricing = val
                break
    if pricing is None:
        logger.warning(f"Unknown model pricing for '{model}', using sonnet pricing as default")
        pricing = {"input": 3.0, "output": 15.0}
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def _make_request_id(config: ScenarioConfig) -> str:
    """为每个请求生成唯一 ID"""
    content = json.dumps(asdict(config), sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:20]


def _parse_json_response(raw_text: str) -> Optional[any]:
    """解析 LLM 返回的 JSON，处理 markdown 代码块包裹、尾部多余文本等"""
    text = raw_text.strip()

    # 去掉 markdown 代码块包裹
    if text.startswith("```"):
        first_newline = text.index("\n")
        last_fence = text.rfind("```")
        if last_fence > first_newline:
            text = text[first_newline + 1:last_fence].strip()
        else:
            text = text[first_newline + 1:].strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个完整的 JSON 对象或数组
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        break

    # 兜底失败
    raise json.JSONDecodeError("Cannot extract valid JSON from response", text[:200], 0)

def _call_llm(
    client: Anthropic,
    system: str,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple:
    """单次 LLM 调用，返回 (原始文本, input_tokens, output_tokens)"""
    # logger.info(f"Calling LLM with model={model}, max_tokens={max_tokens}, temperature={temperature}")
    # logger.info(f"System prompt:\n{system}")
    # logger.info(f"Messages:\n{json.dumps(messages, indent=2, ensure_ascii=False)}")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )
    usage = response.usage
    logger.info(f"LLM call usage: input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}")
    # logger.info(f"LLM response: {response}")  # 调试输出原始文本
    return response.content[0].text, usage.input_tokens, usage.output_tokens

def _call_llm_stream(
    client: Anthropic,
    system: str,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple:
    """单次 LLM 调用（流式），返回 (原始文本, input_tokens, output_tokens)"""
    full_text = ""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            full_text += text
        final_message = stream.get_final_message()
    usage = final_message.usage
    logger.info(f"LLM call usage: input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}")
    full_text = full_text.strip()
    return full_text, usage.input_tokens, usage.output_tokens


_CLAUDE_CODE_TEMPERATURE_WARNED = False
_GEMMA4_TEMPERATURE_WARNED = False


def _call_llm_gemma4(
    client,
    system: str,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple:
    """通过 vLLM OpenAI 兼容接口调用本地 Gemma4 模型，返回 (原始文本, input_tokens, output_tokens)。

    Anthropic Messages API 的 system 参数转为 OpenAI role="system" 消息。
    采样参数固定为 temperature=1.0, top_p=0.95, top_k=64（参考 gemma4_api_long.py，
    跨用例统一），忽略调用方传入的 temperature。
    """
    global _GEMMA4_TEMPERATURE_WARNED
    if not _GEMMA4_TEMPERATURE_WARNED and abs(temperature - 1.0) > 1e-6:
        logger.warning(
            f"gemma4 backend uses standardized sampling (temperature=1.0, top_p=0.95, "
            f"top_k=64); ignoring caller temperature={temperature}"
        )
        _GEMMA4_TEMPERATURE_WARNED = True

    openai_messages = [{"role": "system", "content": system}] + list(messages)
    response = client.chat.completions.create(
        model=model,
        messages=openai_messages,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=0.95,
        extra_body={"top_k": 64},
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    logger.info(f"LLM call usage (gemma4): input_tokens={in_tok}, output_tokens={out_tok}")
    return text.strip(), in_tok, out_tok


def _call_llm_claude_code(
    system: str,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple:
    """通过 Claude Code CLI (`claude -p`) 调用，返回 (原始文本, input_tokens, output_tokens)

    用户 prompt 通过 stdin 传入避免命令行参数长度限制；system prompt 通过 --system-prompt
    完全替换默认 system（这意味着 CLAUDE.md/cwd 等动态段也不会注入）；--tools "" 禁用所有
    工具；--max-turns 1 防止 agent 循环。
    注意：不使用 --bare，因为 --bare 会强制走 ANTHROPIC_API_KEY 路径并禁用 OAuth/keychain，
    会让已通过 `claude /login` 登录的用户出现 "Not logged in"。
    Claude Code CLI 不支持 temperature；max_tokens 也无对应参数。
    """
    global _CLAUDE_CODE_TEMPERATURE_WARNED
    if not _CLAUDE_CODE_TEMPERATURE_WARNED:
        logger.warning(
            f"claude-code backend ignores temperature={temperature} and max_tokens "
            f"(CLI does not expose these); generation will use Claude Code defaults"
        )
        _CLAUDE_CODE_TEMPERATURE_WARNED = True

    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(
            f"claude-code backend expects a single user message; got {len(messages)} messages"
        )
    user_prompt = messages[0]["content"]

    cmd = [
        "claude", "-p",
        "--system-prompt", system,
        "--output-format", "json",
        "--model", model,
        "--max-turns", "1",
        "--tools", "",
        "--no-session-persistence",
    ]
    proc = subprocess.run(
        cmd,
        input=user_prompt,
        capture_output=True,
        text=True,
        check=False,
    )
    # CLI 可能在出错时仍输出 JSON 到 stdout（is_error=true）或直接非零退出，
    # 所以失败诊断信息要同时看 stdout 和 stderr。
    if proc.returncode != 0:
        # 如果 stdout 里就是结构化错误 JSON，优先解析它来给出可读信息
        try:
            err_data = json.loads(proc.stdout) if proc.stdout.strip() else None
        except json.JSONDecodeError:
            err_data = None
        if err_data and err_data.get("is_error"):
            raise RuntimeError(
                f"claude CLI failed (rc={proc.returncode}): {err_data.get('result', '')[:500]}"
            )
        raise RuntimeError(
            f"claude CLI failed (rc={proc.returncode}); "
            f"stderr={proc.stderr[:500]!r}; stdout={proc.stdout[:500]!r}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude CLI output is not valid JSON: {proc.stdout[:1000]}"
        ) from e

    if data.get("is_error"):
        raise RuntimeError(f"claude CLI returned error: {data.get('result', '')[:500]}")

    text = data.get("result", "")
    usage = data.get("usage") or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    logger.info(f"LLM call usage (claude-code): input_tokens={in_tok}, output_tokens={out_tok}")
    return text.strip(), in_tok, out_tok


def _call_llm_route(
    client,
    system: str,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    use_claude_code: bool,
    use_gemma4: bool = False,
) -> tuple:
    """根据后端开关路由到不同 LLM 实现。"""
    if use_claude_code:
        return _call_llm_claude_code(system, messages, model, max_tokens, temperature)
    if use_gemma4:
        return _call_llm_gemma4(client, system, messages, model, max_tokens, temperature)
    return _call_llm_stream(client, system, messages, model, max_tokens, temperature)


def call_two_stage_single(
    client,
    config: ScenarioConfig,
    model: str = "claude-sonnet-4-20250514",
    max_tokens_stage1: int = 16384,
    max_tokens_stage2: int = 16384,
    temperature: float = 0.85,
    s2_max_retries: int = 3,
    use_claude_code: bool = False,
    use_gemma4: bool = False,
) -> Optional[Dict]:
    """
    两阶段调用：先生成场景+角色，再生成对话

    Returns:
        成功时返回完整记录，失败时返回 None
    """
    request_id = _make_request_id(config)
    total_input_tokens = 0
    total_output_tokens = 0
    raw_stage1, raw_stage2 = "", ""

    # ---- 阶段一：场景 + 角色 ----
    try:
        stage1_messages = build_stage1_messages(config)
        raw_stage1, s1_in, s1_out = _call_llm_route(
            client, STAGE1_SYSTEM_PROMPT, stage1_messages,
            model, max_tokens_stage1, temperature, use_claude_code,
            use_gemma4=use_gemma4,
        )
        total_input_tokens += s1_in
        total_output_tokens += s1_out
        stage1_output = _parse_json_response(raw_stage1)

        if not isinstance(stage1_output, dict):
            logger.warning(f"[{request_id}] Stage 1: response is not a dict")
            return None
        if "scenario" not in stage1_output or "participants" not in stage1_output:
            logger.warning(f"[{request_id}] Stage 1: missing required fields")
            return None

        # 基本校验：参与者数量
        # num_users 是总人数（含 assistant 角色），assistant 也被描述为人类
        users = [p for p in stage1_output["participants"] if p.get("role") == "user"]
        assistants = [p for p in stage1_output["participants"] if p.get("role") == "assistant"]
        total_participants = len(users) + len(assistants)
        if total_participants != config.num_users + 1:
            logger.warning(
                f"[{request_id}] Stage 1: expected {config.num_users + 1} total participants "
                f"(num_users={config.num_users} + 1 assistant), got {total_participants}"
            )
            return None
        if len(assistants) != 1:
            logger.warning(f"[{request_id}] Stage 1: expected 1 assistant, got {len(assistants)}")
            return None

    except json.JSONDecodeError as e:
        logger.warning(f"[{request_id}] Stage 1 JSON parse error: {e}")
        logger.warning(f"[{request_id}] Raw stage 1 output:\n{raw_stage1}")
        return None
    except Exception as e:
        logger.error(f"[{request_id}] Stage 1 API call failed: {e}")
        logger.error(traceback.format_exc())
        return None
    logger.info(f"[{request_id}] Stage 1 successful: scenario='{stage1_output['scenario']}', "
          f"participants={len(stage1_output['participants'])}, "
          f"input_tokens={s1_in}, output_tokens={s1_out}")

    # ---- 阶段二：对话生成 ----
    for attempt in range(s2_max_retries):  # 单次失败后不放回队列重试，而是直接在这里重试，避免过多重复调用阶段一
        try:
            stage2_messages = build_stage2_messages(config, stage1_output)
            raw_stage2, s2_in, s2_out = _call_llm_route(
                client, STAGE2_SYSTEM_PROMPT, stage2_messages,
                model, max_tokens_stage2, temperature, use_claude_code,
                use_gemma4=use_gemma4,
            )
            if raw_stage2.startswith("```") and not raw_stage2.endswith("```"):
                logger.warning(f"[{request_id}] Stage 2: response starts with code fence but does not end with it, trying to fix...")
                raw_stage2 = raw_stage2.rsplit('{"speaker":', 1)[0].strip()
                raw_stage2 = raw_stage2.rsplit(',', 1)[0].strip()
                raw_stage2 = raw_stage2 + "\n]\n```"
            total_input_tokens += s2_in
            total_output_tokens += s2_out
            dialogue = _parse_json_response(raw_stage2)

            if not isinstance(dialogue, list):
                logger.warning(f"[{request_id}] Stage 2: response is not a list")
                logger.warning(f"[{request_id}] Raw stage 2 output:\n{raw_stage2}")
                time.sleep(5)
                continue
            else:
                logger.info(f"[{request_id}] Stage 2: successfully parsed dialogue with {len(dialogue)} turns")

        except json.JSONDecodeError as e:
            logger.error(f"[{request_id}] Stage 2 JSON parse error: {e}")
            logger.error(f"Retry stage2 {attempt + 1}/{s2_max_retries} after {5}s...")
            time.sleep(5) 
            continue
        except Exception as e:
            logger.error(f"[{request_id}] Stage 2 API call failed: {e}")
            logger.error(f"Retry stage2 {attempt + 1}/{s2_max_retries} after {10}s...")
            time.sleep(10)
            continue
        logger.info(f"[{request_id}] Stage 2 successful: dialogue turns={len(dialogue)}, "
              f"input_tokens={s2_in}, output_tokens={s2_out}")
        break  # 阶段二成功，跳出重试循环
    else:
        logger.error(f"[{request_id}] Stage 2 failed after {s2_max_retries} attempts")
        return None

    # ---- 组装结果 ----
    # 本地 Gemma4 模型不计费
    cost = 0.0 if use_gemma4 else _calc_cost(model, total_input_tokens, total_output_tokens)
    logger.info(f"[{request_id}] tokens: in={total_input_tokens}, out={total_output_tokens}, cost=${cost:.4f}")
    return {
        "request_id": request_id,
        "config": {
            "seed_sentence": config.seed_sentence,
            "seed_emotion": config.seed_emotion,
            "num_users": config.num_users,
            "turn_count": config.turn_count,
            "language": config.language,
            "ai_involvement": config.ai_involvement,
            "predefined_assistant_role_id": config.predefined_assistant_role["id"] if config.predefined_assistant_role else None,
            "industry": config.industry,
            "topic": config.topic,
        },
        "design": stage1_output["design"],
        "scenario": stage1_output["scenario"],
        "participants": stage1_output["participants"],
        "dialogue": dialogue,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost_usd": round(cost, 6),
        },
        "raw_stage1": raw_stage1,
        "raw_stage2": raw_stage2,
    }


def call_two_stage_batch(
    client,
    configs: List[ScenarioConfig],
    output_path: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens_stage1: int = 16384,
    max_tokens_stage2: int = 16384,
    temperature: float = 0.85,
    max_workers: int = 5,
    max_retries: int = 5,
    retry_delay: float = 5.0,
    max_cost: float = None,
    rejected_path: str = None,
    use_claude_code: bool = False,
    use_gemma4: bool = False,
) -> Dict[str, int]:
    """
    批量两阶段调用

    Args:
        client: Anthropic 客户端
        configs: 场景配置列表
        output_path: 输出 JSONL 文件路径
        model: 模型名称
        max_tokens_stage1: 阶段一最大 token 数
        max_tokens_stage2: 阶段二最大 token 数
        temperature: 温度
        max_workers: 并发数
        max_retries: 最大重试次数
        retry_delay: 重试间隔基数（秒，指数退避）

    Returns:
        统计信息
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载已完成的 seed_sentence（断点续传）
    completed_sentence = set()
    completed_roles = set()
    completed_topics = {}
    previous_tokens_in = 0
    previous_tokens_out = 0
    previous_cost = 0.0

    # 从成功文件加载
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in tqdm(f, desc="Loading completed sentences from output JSONL"):
                try:
                    data = json.loads(line)
                    completed_sentence.add(data["config"]["seed_sentence"])
                    completed_roles.add(data["config"].get("predefined_assistant_role_id"))
                    usage = data.get("usage", {})
                    previous_tokens_in += usage.get("input_tokens", 0)
                    previous_tokens_out += usage.get("output_tokens", 0)
                    previous_cost += usage.get("cost_usd", 0.0)
                    if data["config"].get("industry") and data["config"].get("topic"):
                        topic_key = f"{data['config']['industry']} | {data['config']['topic']}"
                        completed_topics[topic_key] = completed_topics.get(topic_key, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    continue
    logger.info(f"Loaded {len(completed_sentence)} completed sentences from {output_path}, "
            f"total input tokens={previous_tokens_in}, total output tokens={previous_tokens_out}, "
            f"total cost=${previous_cost:.4f}")
    logger.info(f"Completed predefined assistant roles: {len(completed_roles)}")
    logger.info(f"Completed industry-topic combinations: {len(completed_topics)}")

    # 过滤已完成的（基于 seed_sentence ，成功+失败都跳过）
    pending_configs = []
    skipped_configs_due_to_role = 0
    skipped_configs_due_to_sentence = 0
    for config in configs:
        rid = config.seed_sentence
        role_id = config.predefined_assistant_role["id"] if config.predefined_assistant_role else None
        role_name = config.predefined_assistant_role["name"] if config.predefined_assistant_role else None
        if rid not in completed_sentence:
            if role_name is None or role_id not in completed_roles:
                pending_configs.append(config)
            else:
                skipped_configs_due_to_role += 1
                # logger.info(f"Skipping request for {config.seed_sentence} due to completed predefined assistant role {role_id}: {role_name}")
        else:
            skipped_configs_due_to_sentence += 1
            # logger.info(f"Skipping already completed request for {config.seed_sentence}, predefined assistant role {role_id}: {role_name}")
    logger.info(f"Total configs: {len(configs)}, pending: {len(pending_configs)}, "
            f"skipped due to completed sentence: {skipped_configs_due_to_sentence}, "
            f"skipped due to completed predefined assistant role: {skipped_configs_due_to_role}")
    stats = {
        "total": len(configs),
        "previously_success": len(completed_sentence),
        "previously_failed": 0,
        "pending": len(pending_configs),
        "new_success": 0,
        "new_failed": 0,
        "total_input_tokens": previous_tokens_in,
        "total_output_tokens": previous_tokens_out,
        "total_cost_usd": previous_cost,
    }

    if not pending_configs:
        logger.info("All requests already completed.")
        return stats

    logger.info(f"Processing {len(pending_configs)} pending requests with {max_workers} workers...")

    if pending_configs[0].is_work_meeting:
        from collections import defaultdict
        from itertools import zip_longest

        # 按 topic 聚合
        topic_items = defaultdict(list)
        topic_count = {}
        for c in pending_configs:
            key = f"{c.industry} | {c.topic}"
            topic_items[key].append(c)
            topic_count[key] = completed_topics.get(key, 0)

        max_count = max(topic_count.values(), default=0)

        catch_up = []  # 各 topic 需要"补差"的部分
        uniform  = []  # 追平后剩余的部分（各 topic 已等高）

        for key, items in topic_items.items():
            deficit = max_count - topic_count[key]
            n = min(deficit, len(items))
            if n:
                catch_up.append(items[:n])   # 只补够追平即可
            if n < len(items):
                uniform.append(items[n:])    # 剩余进入均匀阶段

        catch_up.sort(key=len, reverse=True)
        uniform.sort(key=len, reverse=True)

        # 阶段一：追平 —— deficit 大的 topic 自然在这里占更多位置
        result = [
            c
            for batch in zip_longest(*catch_up)
            for c in batch if c is not None
        ]

        # 阶段二：均匀 —— 所有 topic 虚拟 count 已齐平，公平 round-robin
        result += [
            c
            for batch in zip_longest(*uniform)
            for c in batch if c is not None
        ]

        pending_configs = result
        logger.info("Sorted pending configs by least completed industry-topic combinations to prioritize underrepresented scenarios")
        cnt_zero = sum(1 for c in pending_configs if completed_topics.get(f"{c.industry} | {c.topic}", 0) == 0)
        logger.info(f"{cnt_zero}/{len(pending_configs)} pending configs are from industry-topic combinations with zero completed dialogues")

    def _call_with_retry(config: ScenarioConfig) -> Optional[Dict]:
        for attempt in range(max_retries):
            result = call_two_stage_single(
                client, config, model=model,
                max_tokens_stage1=max_tokens_stage1,
                max_tokens_stage2=max_tokens_stage2,
                temperature=temperature,
                s2_max_retries=max_retries*2,
                use_claude_code=use_claude_code,
                use_gemma4=use_gemma4,
            )
            if result is not None:
                logger.info(f"Request for seed_sentence='{config.seed_sentence}' succeeded on attempt {attempt + 1}")
                return result
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.warning(f"Retry stage1 {attempt + 1}/{max_retries} after {wait}s...")
                time.sleep(wait)
        else:
            logger.error(f"Failed after {max_retries} attempts for seed_sentence: {config.seed_sentence}")
        return None

    # 顺序 / 并发执行
    budget_exceeded = False

    with open(output_path, "a") as f_out:

        # ── 公共结果处理（两个分支共用） ──────────────────────────
        def _handle_result(i: int, result) -> None:
            if result is not None:
                logger.info(
                    f"Successfully processed request {i + 1}/{len(pending_configs)}: "
                    f"seed_sentence='{result['config']['seed_sentence']}'"
                )
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                stats["new_success"] += 1
                usage = result.get("usage", {})
                stats["total_input_tokens"] += usage.get("input_tokens", 0)
                stats["total_output_tokens"] += usage.get("output_tokens", 0)
                stats["total_cost_usd"] += usage.get("cost_usd", 0.0)
            else:
                stats["new_failed"] += 1

            if (i + 1) % 10 == 0:
                total_success = stats["previously_success"] + stats["new_success"]
                total_failed  = stats["previously_failed"]  + stats["new_failed"]
                logger.info(
                    f"Progress: {i + 1}/{len(pending_configs)} "
                    f"(total_success={total_success}, total_failed={total_failed}, "
                    f"cost=${stats['total_cost_usd']:.4f})"
                )

        # ── 分支：单线程顺序执行（max_workers=1） ─────────────────
        if max_workers == 1:
            for i, config in enumerate(
                tqdm(pending_configs, total=len(pending_configs), desc="Processing requests")
            ):
                if budget_exceeded:
                    break
                result = _call_with_retry(config)
                _handle_result(i, result)

        # ── 分支：多线程并发执行（max_workers>1） ─────────────────
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {
                    executor.submit(_call_with_retry, config): config
                    for config in pending_configs
                }
                for i, future in enumerate(
                    tqdm(as_completed(future_to_config), total=len(pending_configs), desc="Processing requests")
                ):
                    if budget_exceeded:
                        future.cancel()
                        continue
                    result = future.result()
                    _handle_result(i, result)

    total_success = stats["previously_success"] + stats["new_success"]
    total_failed  = stats["previously_failed"]  + stats["new_failed"]
    logger.info(
        f"Done. Total={stats['total']}, Success={total_success} "
        f"(previous={stats['previously_success']}, new={stats['new_success']}), "
        f"Failed={total_failed} (previous={stats['previously_failed']}, new={stats['new_failed']}), "
        f"Pending={stats['pending']}, "
        f"Tokens(in={stats['total_input_tokens']}, out={stats['total_output_tokens']}), "
        f"Cost=${stats['total_cost_usd']:.4f}"
    )
    return stats