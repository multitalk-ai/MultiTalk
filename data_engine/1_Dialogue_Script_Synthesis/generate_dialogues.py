"""
多人对话文本数据构建 - 主流程脚本（两阶段架构）

阶段一：LLM 生成场景描述 + 角色画像（含真实名字）
阶段二：LLM 基于阶段一输出生成对话

用法:
    # 生成全量数据
    python generate_dialogues.py --output_dir ./output --num_per_seed 1

    # 小规模测试
    python generate_dialogues.py --output_dir ./output --max_seeds 10 --num_per_seed 1

    # 只做过滤（跳过生成）
    python generate_dialogues.py --output_dir ./output --filter_only

    # 预览 prompt（不调用 API）
    python generate_dialogues.py --output_dir ./output --preview --max_seeds 3
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import pandas as pd
from anthropic import Anthropic
from typing import List, Dict, Optional
from scenario_config import (
    ScenarioConfig,
    sample_scenario,
    INDUSTRY_TOPICS_LIST,
    load_roles_from_jsonl,
    random_qa_sample,
)
import httpx

from api_caller import call_two_stage_batch, call_two_stage_single
from quality_filter import filter_batch, filter_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class _Gemma4LoadBalancer:
    """跨多个 OpenAI 兼容端点随机分发请求的极简负载均衡器。

    暴露 OpenAI Python SDK 在本项目实际用到的属性（.chat, .models）。每次属性访问都会
    随机挑一个底层客户端，因此一次完整调用链 `client.chat.completions.create(...)`
    会落在同一个被选中的端点上。CPython 下 random.choice 是原子的，OpenAI 客户端本身亦线程安全。
    """

    def __init__(self, clients):
        self._clients = list(clients)
        if not self._clients:
            raise ValueError("Gemma4 load balancer requires at least one client")

    def __len__(self):
        return len(self._clients)

    def _pick(self):
        return random.choice(self._clients)

    @property
    def chat(self):
        return self._pick().chat

    @property
    def models(self):
        return self._pick().models


def _build_gemma4_client(api_bases, api_key: Optional[str] = None):
    """构造调用 Gemma4 的 OpenAI 兼容客户端。

    api_bases 接受单个 URL 字符串或 URL 列表（后者启用负载均衡）。
    """
    from openai import OpenAI

    if isinstance(api_bases, str):
        api_bases = [api_bases]
    api_bases = [b for b in api_bases if b]
    if not api_bases:
        raise ValueError("--api_base or --gemma4_list must be provided when using --gemma4")

    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
    clients = [OpenAI(api_key=resolved_key, base_url=b) for b in api_bases]
    if len(clients) == 1:
        logger.info(f"[Gemma4] connecting to OpenAI-compatible endpoint at {api_bases[0]}")
        return clients[0]
    logger.info(f"[Gemma4] load-balancing across {len(clients)} endpoints:")
    for b in api_bases:
        logger.info(f"  - {b}")
    return _Gemma4LoadBalancer(clients)


def _load_gemma4_api_bases() -> List[str]:
    """从同目录下的 gemma4_list.py 读取 api_bases 列表。"""
    from gemma4_list import api_bases as _api_bases
    return list(_api_bases)


def _resolve_gemma4_model(client, requested_model: str) -> str:
    """若 --model 仍是 Claude 默认值，则向 vLLM /v1/models 询问真正加载的模型名，避免 404。"""
    if requested_model and not requested_model.startswith("claude-"):
        return requested_model
    try:
        listing = client.models.list()
        served = [m.id for m in listing.data]
    except Exception as e:
        logger.error(
            f"[Gemma4] --model='{requested_model}' looks like a Claude default and auto-resolve "
            f"via /v1/models failed: {e}. Pass --model explicitly with the served model name."
        )
        raise
    if not served:
        raise RuntimeError("[Gemma4] /v1/models returned an empty list at the configured endpoint")
    resolved = served[0]
    logger.info(f"[Gemma4] auto-resolved --model '{requested_model}' -> '{resolved}'")
    if len(served) > 1:
        logger.info(f"[Gemma4] endpoint serves {len(served)} models; picked first. Others: {served[1:]}")
    return resolved


def load_emotion_dataset(data_path: str) -> pd.DataFrame:
    """加载 emotion 数据集"""
    if "," in data_path:
        paths = data_path.split(",")
        dfs = []
        for p in paths:
            df_part = pd.read_parquet(p)
            dfs.append(df_part)
            logger.info(f"Loaded {len(df_part)} samples from {p}")
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset has {len(df)} samples")
    else:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def generate_scenario_configs(
    df: pd.DataFrame,
    num_per_seed: int = 1,
    max_seeds: int = None,
    seed: int = 42,
    roles: List[Dict] = None,
    is_work_meeting: bool = False,
    is_ai_assistant: bool = False,
    qa_path: Optional[str] = None,
) -> list:
    """
    为每条种子句子生成场景配置（只含维度参数，不含角色画像）

    Args:
        df: emotion 数据集
        num_per_seed: 每条种子生成几段对话（如果提供了 roles，则表示每个角色生成几段对话）
        max_seeds: 最多使用多少条种子（None=全部）。如果提供了 roles，此参数被忽略
        seed: 随机种子
        roles: 预定义角色列表（可选）。如果提供，将为每个角色生成 num_per_seed 段对话
        is_work_meeting: 是否生成工作场景
    """
    random.seed(seed)

    configs = []

    if roles:
        # 基于角色生成配置：为每个角色生成 num_per_seed 段对话
        logger.info(f"Generating configs based on {len(roles)} roles, {num_per_seed} dialogues per role")

        for role in tqdm(roles, desc="Generating configs for predefined roles"):
            for idx in range(num_per_seed):
                # 从数据集中随机选择一个 seed_sentence
                row = df.sample(n=1, random_state=None).iloc[0]
                sentence = row["text"]
                label = row["label"]

                config = sample_scenario(sentence, label, assigned_role=True, is_work_meeting=is_work_meeting, is_ai_assistant=is_ai_assistant)
                config.predefined_assistant_role = role
                configs.append(config)
    else:
        # 基于 seed_sentence 生成配置
        if max_seeds is not None:
            df = df.sample(n=min(max_seeds, len(df)), random_state=seed)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating configs for seed sentences"):
            sentence = row["text"]
            label = row["label"]
            if is_work_meeting:
                # 工作场景强制使用会议相关的行业+话题
                industry, topic = INDUSTRY_TOPICS_LIST[idx % len(INDUSTRY_TOPICS_LIST)]
            else:
                industry, topic = None, None
            for _ in range(num_per_seed):
                config = sample_scenario(sentence, label, assigned_role=False, is_work_meeting=is_work_meeting, is_ai_assistant=is_ai_assistant, industry=industry, topic=topic)
                configs.append(config)

    random.shuffle(configs)
    logger.info(f"Generated {len(configs)} scenario configs")

    # --seed-qa：从外部 Q&A 数据集（默认 extracted_data.jsonl）抽一条作为
    # 对话话题"种子"。LLM 不必逐字复述，只把它当作灵感。为 None 时不注入。
    if qa_path:
        if not os.path.exists(qa_path):
            raise RuntimeError(f"--seed-qa requires JSONL at {qa_path!r}")
        qa_rng = random.Random(seed)
        qa_pairs = random_qa_sample(qa_path, len(configs), qa_rng)
        logger.info(f"--seed-qa: sampled {len(qa_pairs)}/{len(configs)} Q&A pairs from {qa_path}")
        for cfg, rec in zip(configs, qa_pairs):
            cfg.qa_question = rec.get("question")
            cfg.qa_answer = rec.get("answer")

    # 统计维度分布
    lang_dist = Counter(c.language for c in configs)
    user_dist = Counter(c.num_users for c in configs)
    ai_dist = Counter(c.ai_involvement for c in configs)
    logger.info(f"Language distribution: {dict(lang_dist)}")
    logger.info(f"User count distribution: {dict(user_dist)}")
    logger.info(f"AI involvement distribution: {dict(ai_dist)}")

    if roles:
        logger.info(f"Using {len(roles)} predefined assistant roles, total configs: {len(configs)}")

    return configs


def split_dataset(input_path: str, output_dir: str, train_ratio=0.9, val_ratio=0.05, seed=42):
    """将过滤后的数据划分为 train/val/test"""
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(line)

    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }

    for split_name, split_records in splits.items():
        split_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(split_path, "w") as f:
            for record in split_records:
                f.write(record + "\n")
        logger.info(f"{split_name}: {len(split_records)} records -> {split_path}")


def _setup_gemma4_backend(args):
    """构造 Gemma4 客户端并按需把 args.model 解析为服务端实际加载的模型名。

    返回构造好的 OpenAI 客户端（可能是 _Gemma4LoadBalancer）。
    """
    if args.gemma4_list:
        api_bases = _load_gemma4_api_bases()
        logger.info(
            f"Using Gemma4 backend with --gemma4_list: {len(api_bases)} endpoints from gemma4_list.py"
        )
    else:
        api_bases = args.api_base
        logger.info("Using Gemma4 backend with --api_base (single endpoint)")
    client = _build_gemma4_client(api_bases=api_bases, api_key=args.api_key)
    args.model = _resolve_gemma4_model(client, args.model)
    return client


def run_preview_mode(args, output_dir: Path):
    """运行预览模式：生成一条样例数据并展示"""
    if args.claude_code:
        logger.info("Using Claude Code CLI as generation backend (--claude-code)")
        client = None
    elif args.gemma4:
        client = _setup_gemma4_backend(args)
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key,
        timeout=httpx.Timeout(
            connect=10.0,    # 连接超时
            read=600.0,      # 读取超时（10分钟）
            write=30.0,      # 写入超时
            pool=10.0        # 连接池超时
        ))
        if args.api_base:
            client.base_url = args.api_base

    # 加载角色数据（如果提供）
    roles = None
    if args.role_path:
        logger.info(f"Loading roles from {args.role_path}")
        roles = load_roles_from_jsonl(args.role_path)
        logger.info(f"Loaded {len(roles)} roles")

    df = load_emotion_dataset(args.data_path)
    configs = generate_scenario_configs(
        df, num_per_seed=1, max_seeds=1, seed=args.seed, roles=roles,
        is_work_meeting=args.is_work_meeting, is_ai_assistant=args.ai_assistant,
        qa_path=(args.qa_path if args.seed_qa else None),
    )
    config = configs[0]

    print(f"\n{'='*70}")
    print(f"Preview: emotion={config.seed_emotion}, users={config.num_users}, "
          f"lang={config.language}, "
          f"ai_inv={config.ai_involvement}")
    print(f"Seed: {config.seed_sentence}")
    print(f"Model: {args.model}")
    if config.predefined_assistant_role:
        print(f"Predefined Role: {config.predefined_assistant_role.get('name', 'Unknown')}")
    print(f"{'='*70}")

    logger.info("Calling API (two-stage)...")
    result = call_two_stage_single(
        client, config,
        model=args.model,
        temperature=args.temperature,
        use_claude_code=args.claude_code,
        use_gemma4=args.gemma4,
    )

    if result is None:
        logger.error("Generation failed. Check logs above for details.")
        sys.exit(1)

    # 展示结果
    print(f"\n--- STAGE 1 OUTPUT: Scenario + Participants ---")
    print(f"Scenario: {result['scenario']}")
    print(f"Participants:")
    for p in result["participants"]:
        summary = p.get('summary', p.get('description', ''))
        print(f"  [{p['role']}] {p['name']}: {summary}")
        if 'roleplay_guidelines' in p:
            print(f"    Speech: {p['roleplay_guidelines'][:100]}...")

    print(f"\n--- STAGE 2 OUTPUT: Dialogue ({len(result['dialogue'])} turns) ---")
    for turn in result["dialogue"]:
        overlap_tag = " [OVERLAP]" if turn.get("overlap") else ""
        voice = turn.get("voice", {})
        if voice:
            voice_tag = f" ({voice.get('emotion', '?')}:{voice.get('emotion_intensity', '?')} {voice.get('speed', '?')} {voice.get('volume', '?')})"
        else:
            voice_tag = ""
        print(f"  {turn['speaker']}: {turn['text']}{overlap_tag}{voice_tag}")

    # 费用信息
    usage = result.get("usage", {})
    if usage:
        print(f"\n--- COST ---")
        print(f"  Tokens: in={usage.get('input_tokens', 0)}, out={usage.get('output_tokens', 0)}")
        print(f"  Cost: ${usage.get('cost_usd', 0):.4f}")

    # 质量过滤检查
    passed, reasons = filter_single(result)
    print(f"\n--- QUALITY CHECK ---")
    if passed:
        print("PASSED all filters")
    else:
        print(f"FAILED: {reasons}")

    # 保存到文件
    preview_path = output_dir / "preview_sample.json"
    with open(preview_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nFull result saved to {preview_path}")


def run_generation_stage(args, output_dir: Path, raw_path: Path, rejected_path: Path):
    """运行生成阶段：批量生成对话数据"""
    if args.claude_code:
        logger.info("Using Claude Code CLI as generation backend (--claude-code)")
        client = None
    elif args.gemma4:
        client = _setup_gemma4_backend(args)
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("No API key provided. Use --api_key or set ANTHROPIC_API_KEY")
            sys.exit(1)

        client = Anthropic(api_key=api_key)
        if args.api_base:
            client.base_url = args.api_base

    # 加载角色数据（如果提供）
    roles = None
    if args.role_path:
        logger.info(f"Loading roles from {args.role_path}")
        roles = load_roles_from_jsonl(args.role_path)
        logger.info(f"Loaded {len(roles)} roles")

    df = load_emotion_dataset(args.data_path)
    configs = generate_scenario_configs(
        df,
        num_per_seed=args.num_per_seed,
        max_seeds=args.max_seeds,
        seed=args.seed,
        roles=roles,
        is_work_meeting=args.is_work_meeting,
        is_ai_assistant=args.ai_assistant,
        qa_path=(args.qa_path if args.seed_qa else None),
    )

    if args.max_samples is not None and len(configs) > args.max_samples:
        logger.info(f"Truncating configs from {len(configs)} to {args.max_samples} (--max_samples)")
        configs = configs[:args.max_samples]

    # 保存配置快照
    if args.claude_code:
        backend = "claude-code"
    elif args.gemma4:
        backend = "gemma4-vllm"
    else:
        backend = "anthropic-api"
    config_snapshot = {
        "num_seeds": len(configs),
        "num_per_seed": args.num_per_seed,
        "model": args.model,
        "temperature": args.temperature,
        "max_workers": args.max_workers,
        "architecture": "two-stage",
        "role_path": args.role_path if args.role_path else None,
        "num_roles": len(roles) if roles else 0,
        "backend": backend,
        "is_ai_assistant": args.ai_assistant,
        "is_work_meeting": args.is_work_meeting,
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    logger.info(f"Starting generation for {len(configs)} configs with model {args.model}...")
    stats = call_two_stage_batch(
        client=client,
        configs=configs,
        output_path=str(raw_path),
        model=args.model,
        temperature=args.temperature,
        max_workers=args.max_workers,
        max_cost=args.max_cost,
        rejected_path=str(rejected_path),
        use_claude_code=args.claude_code,
        use_gemma4=args.gemma4,
    )
    logger.info(f"Generation stats: {stats}")
    return stats


def run_filter_stage(raw_path: Path, filtered_path: Path, rejected_path: Path):
    """运行过滤阶段：质量过滤"""
    if not raw_path.exists():
        logger.error(f"Raw output not found: {raw_path}")
        sys.exit(1)

    filter_stats = filter_batch(
        input_path=str(raw_path),
        output_path=str(filtered_path),
        rejected_path=str(rejected_path),
    )
    logger.info(f"Filter stats: {filter_stats}")
    return filter_stats


def run_split_stage(filtered_path: Path, output_dir: Path, seed: int):
    """运行划分阶段：train/val/test 划分"""
    split_dataset(
        str(filtered_path),
        str(output_dir),
        train_ratio=0.98,
        val_ratio=0.01,
        seed=seed
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-user dialogue data generation (two-stage)")
    parser.add_argument("--data_path", type=str,
                        default="seed_data/emotion_train.parquet,seed_data/emotion_unsplit.parquet",
                        help="Path(s) to emotion dataset parquet file(s), comma-separated. "
                             "Download from https://huggingface.co/datasets/dair-ai/emotion "
                             "(splits: train and unsplit).")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--role_path", type=str, default=None,
                        help="Path to predefined roles JSONL file. If provided, output_dir will have '_role' suffix added automatically.")
    parser.add_argument("--num_per_seed", type=int, default=1,
                        help="Number of dialogues to generate per seed sentence")
    parser.add_argument("--max_seeds", type=int, default=None,
                        help="Max number of seed sentences to process (None=all)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max total number of dialogues to generate (None=all). "
                             "Applied after config expansion, truncates the config list.")
    parser.add_argument("--max_cost", type=float, default=1e10,
                        help="Max total cost in USD. Generation stops when this budget is exceeded.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Model name to use. For Anthropic backend, one of "
                             "claude-opus-4-6 / claude-sonnet-4-6 / claude-sonnet-4-5-20250929 / "
                             "claude-haiku-4-5-20251001. For --gemma4, the local model name or "
                             "HF path served by vLLM (e.g. google/gemma-3-27b-it).")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of concurrent API calls")
    parser.add_argument("--temperature", type=float, default=0.85,
                        help="Generation temperature")
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.")
    parser.add_argument("--api_base", type=str,
                        default=os.environ.get("ANTHROPIC_BASE_URL"),
                        help="Override Anthropic base URL. Defaults to ANTHROPIC_BASE_URL env var.")
    parser.add_argument("--filter_only", action="store_true",
                        help="Skip generation, only run filtering on existing raw output")
    parser.add_argument("--preview", action="store_true",
                        help="Preview mode: print sample prompts for both stages and exit")
    parser.add_argument("--seed-industry", dest="is_work_meeting", action="store_true",
                        help="Seed scenarios from INDUSTRY_TOPICS (round-robin over "
                             "industry x topic pairs); produces workplace-meeting style "
                             "dialogues. Output dir gains the '_work_meeting' suffix.")
    parser.add_argument("--ai-assistant", action="store_true",
                        help="AI assistant mode: the assistant participant is a real AI voice "
                             "assistant (Siri/Alexa/Claude-voice style) rather than a disguised human. "
                             "Mutually exclusive with --role_path and --seed-industry.")
    parser.add_argument("--claude-code", action="store_true",
                        help="Use the Claude Code CLI (`claude -p`) as generation backend "
                             "instead of the Anthropic API. Skips API key/base; ignores "
                             "temperature (CLI does not expose it).")
    parser.add_argument("--gemma4", action="store_true",
                        help="Use a local Gemma4 model served via a vLLM OpenAI-compatible "
                             "endpoint (specified by --api_base) as generation backend. "
                             "Sampling is fixed to temperature=1.0, top_p=0.95, top_k=64. "
                             "Mutually exclusive with --claude-code.")
    parser.add_argument("--gemma4_list", action="store_true",
                        help="With --gemma4: load-balance requests across the multiple "
                             "OpenAI-compatible endpoints listed in gemma4_list.py. "
                             "When set, --api_base is ignored.")
    parser.add_argument("--seed-qa", dest="seed_qa", action="store_true",
                        help="Sample one Q&A from an external dataset (default "
                             "extracted_data.jsonl, override via --qa_path) and inject "
                             "it as a TOPIC SEED into stage1 + stage2 prompts. The LLM is "
                             "instructed not to recite it verbatim — only borrow the "
                             "topic angle.")
    parser.add_argument("--qa_path", type=str, default="seed_data/extracted_data.jsonl",
                        help="Path to the Q&A JSONL used by --seed-qa. Each line should "
                             "be a JSON object with at least 'idx', 'question', 'answer' "
                             "fields.")

    args = parser.parse_args()

    # AI assistant 模式与 role / work-meeting 模式互斥（前者 assistant 是 AI，后者是人类）
    if args.ai_assistant and (args.role_path or args.is_work_meeting):
        logger.error("--ai-assistant is mutually exclusive with --role_path and --seed-industry")
        sys.exit(1)

    # 后端开关互斥
    if args.claude_code and args.gemma4:
        logger.error("--claude-code and --gemma4 are mutually exclusive")
        sys.exit(1)
    if args.gemma4_list and not args.gemma4:
        logger.error("--gemma4_list requires --gemma4")
        sys.exit(1)
    if args.gemma4 and not args.gemma4_list and not args.api_base:
        logger.error("--gemma4 requires --api_base (single endpoint) or --gemma4_list (load-balanced from gemma4_list.py)")
        sys.exit(1)
    if args.gemma4 and (not args.model or "gemma" not in args.model.lower()):
        logger.warning("--gemma4 is enabled but model name does not contain 'gemma'. "
                       "Make sure --model points to the correct Gemma model served by vLLM, "
                       "or rely on auto-resolve from /v1/models if the endpoint serves only one.")

    # 如果提供了角色文件，自动在输出目录添加 _role 后缀
    output_dir = Path(args.output_dir)
    if args.role_path:
        output_dir = Path(str(output_dir) + "_role")
        logger.info(f"Using predefined roles, output directory changed to: {output_dir}")
    if args.is_work_meeting:
        output_dir = Path(str(output_dir) + "_work_meeting")
        logger.info(f"Using work meeting scenarios, output directory changed to: {output_dir}")
    if args.ai_assistant:
        output_dir = Path(str(output_dir) + "_ai_assistant")
        logger.info(f"Using AI assistant mode, output directory changed to: {output_dir}")
    if args.claude_code:
        output_dir = Path(str(output_dir) + "_cc")
        logger.info(f"Using Claude Code backend, output directory changed to: {output_dir}")
    if args.gemma4:
        output_dir = Path(str(output_dir) + "_gemma4")
        logger.info(f"Using Gemma4 backend, output directory changed to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_dialogues.jsonl"
    filtered_path = output_dir / "filtered_dialogues.jsonl"
    rejected_path = output_dir / "rejected_dialogues.jsonl"

    # Preview 模式
    if args.preview:
        run_preview_mode(args, output_dir)
        return

    # 生成阶段
    if not args.filter_only:
        run_generation_stage(args, output_dir, raw_path, rejected_path)

    # 过滤阶段
    run_filter_stage(raw_path, filtered_path, rejected_path)

    # 划分阶段
    run_split_stage(filtered_path, output_dir, args.seed)

    logger.info(f"All done. Output in {output_dir}")


if __name__ == "__main__":
    main()