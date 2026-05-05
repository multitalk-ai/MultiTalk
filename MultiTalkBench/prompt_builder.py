"""
prompt_builder.py
─────────────────────────────────────────────────────────────────────────
Multi-Person IQ (MPIQ) Meeting Evaluation Pipeline — Prompt Builder

6-call pipeline:
  CALL 1            : Role identification         (sequential, must run first)
  CALL 2a │ 2b │ 2c : MPIQ dimensions M1–M6       (parallel group "mpiq")
  CALL 3a │ 3b      : General + role-specific     (parallel group "dim")

Supported languages: Language.ZH (Chinese), Language.EN (English)

Quick start:
    from prompt_builder import PromptBuilder, PipelineInput, Language

    inp = PipelineInput(
        transcript="Alice: Let's discuss the budget...",
        target_speaker="Alice",
        all_speakers=["Alice", "Bob", "Carol"],
        language=Language.EN,
    )
    builder = PromptBuilder()

    call1 = builder.build_call1(inp)
    # → send call1 to your LLM, parse the JSON result
    inp.primary_role = json.loads(llm_response)["primary_role"]

    for prompt in builder.build_remaining(inp):
        # parallel_group == "mpiq" → run 2a/2b/2c concurrently
        # parallel_group == "dim"  → run 3a/3b concurrently
        print(prompt.call_id, prompt.parallel_group)
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


# ══════════════════════════════════════════════════════════════════════════
# ❶  Enums & Core Data Structures
# ══════════════════════════════════════════════════════════════════════════

class Language(str, Enum):
    ZH = "zh"
    EN = "en"


#: Maps every accepted role string → canonical lowercase key
ROLE_CANONICAL: Dict[str, str] = {
    "主持人": "facilitator",    "Facilitator": "facilitator",
    "推动者": "driver",         "Driver":       "driver",
    "协作者": "collaborator",   "Collaborator": "collaborator",
    "评议者": "evaluator",      "Evaluator":    "evaluator",
}

#: Human-readable role labels per language
ROLE_LABELS: Dict[str, Dict[str, str]] = {
    "facilitator":  {"zh": "主持人",  "en": "Facilitator"},
    "driver":       {"zh": "推动者",  "en": "Driver"},
    "collaborator": {"zh": "协作者",  "en": "Collaborator"},
    "evaluator":    {"zh": "评议者",  "en": "Evaluator"},
}

#: Dimension codes for each CALL 2x
_CALL2_DIMS: Dict[str, List[str]] = {
    "call2a": ["M1", "M2"],
    "call2b": ["M3", "M4"],
    "call2c": ["M5", "M6"],
}

#: Role-specific dimension codes
_ROLE_DIM_CODES: Dict[str, List[str]] = {
    "facilitator":  ["F1", "F2", "F3", "F4", "F5"],
    "driver":       ["D1", "D2", "D3", "D4"],
    "collaborator": ["C1", "C2", "C3", "C4"],
    "evaluator":    ["E1", "E2", "E3", "E4"],
}


@dataclass
class PromptPair:
    """A (system, user) prompt pair for one LLM call."""
    call_id:        str
    system:         str
    user:           str
    parallel_group: Optional[str] = None  # same group → can run concurrently
    description:    str = ""

    def to_dict(self) -> dict:
        return {
            "call_id":         self.call_id,
            "parallel_group":  self.parallel_group,
            "description":     self.description,
            "system":          self.system,
            "user":            self.user,
        }

    def to_openai_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI-style messages list."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user",   "content": self.user},
        ]

    def to_anthropic_kwargs(self) -> Dict:
        """Convert to keyword arguments for Anthropic client.messages.create()."""
        return {
            "system":   self.system,
            "messages": [{"role": "user", "content": self.user}],
        }


@dataclass
class PipelineInput:
    """All data needed to assemble every prompt in the 6-call pipeline."""
    transcript:     str
    target_speaker: str
    all_speakers:   List[str]
    language:       Language = Language.ZH
    primary_role:   Optional[str] = None   # set after CALL 1 is parsed
    secondary_role: Optional[str] = None

    @property
    def speakers_str(self) -> str:
        sep = "、" if self.language == Language.ZH else ", "
        return sep.join(self.all_speakers)

    def role_key(self) -> Optional[str]:
        """Return canonical lowercase role key (e.g. 'facilitator')."""
        if self.primary_role is None:
            return None
        return ROLE_CANONICAL.get(self.primary_role)

    def validate(self) -> None:
        if not self.transcript.strip():
            raise ValueError("transcript is empty")
        if not self.target_speaker.strip():
            raise ValueError("target_speaker is empty")
        if not self.all_speakers:
            raise ValueError("all_speakers is empty")
        if self.target_speaker not in self.all_speakers:
            raise ValueError(
                f"target_speaker '{self.target_speaker}' not in all_speakers"
            )


# ══════════════════════════════════════════════════════════════════════════
# ❷  Template Rendering Helper
# ══════════════════════════════════════════════════════════════════════════

def _render(template: str, **kwargs: str) -> str:
    """
    Replace ``%%KEY%%`` placeholders in *template* with provided values.

    We deliberately avoid Python f-strings / str.format() because the
    templates contain JSON literal braces { } that would cause conflicts.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"%%{key}%%", value)
    unresolved = re.findall(r"%%[A-Z_]+%%", result)
    if unresolved:
        warnings.warn(f"Unresolved template placeholders: {unresolved}", stacklevel=3)
    return result


def _build_json_block(dims: List[str], lang: Language) -> str:
    """Build the multi-dimension JSON output format block dynamically."""
    if lang == Language.ZH:
        requirements = (
            "补充要求：必须只输出一个完整 JSON 对象；必须覆盖全部维度键；"
            "每个 evidence 数组最多 1 条；每条证据必须是极短原文摘录，不超过 20 个字符；"
            "没有证据就输出 []；不要输出 Markdown 或任何额外说明。"
        )
        inner = (
            '  "{DIM}": {{\n'
            '    "score": 1至5的整数或"N/A",\n'
            '    "positive_evidence": ["最多1条极短正面证据"],\n'
            '    "negative_evidence": ["最多1条极短负面证据"],\n'
            '    "reasoning": "评分理由，40字以内"\n'
            "  }}"
        )
        header = "## 输出格式（严格 JSON）"
    else:
        requirements = (
            "Requirements: output exactly one complete JSON object and include every dimension key; "
            "each evidence array may contain at most one item; each quote must be a very short "
            "verbatim snippet no longer than 12 words; use [] when there is no evidence; "
            "do not output Markdown or any extra text."
        )
        inner = (
            '  "{DIM}": {{\n'
            '    "score": integer 1\u20135 or "N/A",\n'
            '    "positive_evidence": ["at most 1 very short positive quote"],\n'
            '    "negative_evidence": ["at most 1 very short negative quote"],\n'
            '    "reasoning": "scoring rationale, within 40 words"\n'
            "  }}"
        )
        header = "## Output Format (strict JSON)"

    body = ",\n".join(inner.format(DIM=d) for d in dims)
    return f"{header}\n{requirements}\n{{\n{body}\n}}"


# ══════════════════════════════════════════════════════════════════════════
# ❸  Shared Scoring Rubric Fragments
# ══════════════════════════════════════════════════════════════════════════

_RUBRIC_ZH = """\
## 统一评分标准
5分：表现突出，有明确可引用的正面证据，无明显失误
4分：整体良好，偶有小瑕疵但不影响整体效果
3分：基本达标，无明显亮点也无重大失误
2分：有明显问题，影响了会议效果，但并非完全缺失
1分：几乎未体现，或存在明显的负面表现
N/A：有效发言片段不足5条，无法评估"""

_RUBRIC_EN = """\
## Scoring Rubric
5 – Outstanding: clear positive evidence, no notable flaws
4 – Good: generally solid, minor imperfections with no significant impact
3 – Adequate: meets the baseline, no obvious strengths or major failures
2 – Below average: noticeable problems that affected meeting outcomes, not entirely absent
1 – Poor: almost no evidence of the behavior, or clear negative performance
N/A – Not enough valid speaking turns (< 5) to evaluate"""


# ══════════════════════════════════════════════════════════════════════════
# ❹  Chinese Prompt Templates
# ══════════════════════════════════════════════════════════════════════════

_CALL1_SYS_ZH = """\
你是一名专业的会议行为分析专家，负责识别会议参与者的角色类型。

## 角色定义

**主持人（Facilitator）**
核心特征：开场或收尾发言、主动切换议题（3次以上）、分配他人发言权、宣布或确认决议
典型语言："咱们来讨论下一个"、"你来说说"、"那就定这个方案"、"今天先到这"

**推动者（Driver）**
核心特征：主动提出新观点占比高、引领讨论方向、提出具体方案或行动建议，多为主动发起而非回应他人
典型语言："我建议…"、"可以这样做…"、"我觉得方向应该是…"

**协作者（Collaborator）**
核心特征：在他人观点上延伸构建、整合不同意见、促进共识形成，延伸他人次数 > 独立发起次数
典型语言："在你说的基础上…"、"这两点可以结合…"、"我觉得大家说的其实都有道理"

**评议者（Evaluator）**
核心特征：质疑现有方案、指出风险或限制条件、提出反例，质疑类发言占比超过35%
典型语言："但这里有个问题…"、"这样做会不会…"、"有一个前提我们还没讨论"

## 识别优先级规则
主持人 > 推动者 > 评议者 > 协作者（默认兜底）

同时满足两种角色明显特征时，记录主要角色与次要角色。

## 输出格式（严格 JSON，不输出任何额外说明）
{
  "primary_role": "主持人|推动者|协作者|评议者",
  "secondary_role": "主持人|推动者|协作者|评议者|null",
  "confidence": "high|medium|low",
  "turn_count": 目标说话人总发言轮数（整数）,
  "key_evidence": [
    "证据1（直接引用原文片段 + 说明属于哪类特征）",
    "证据2",
    "证据3"
  ],
  "reasoning": "角色判断的核心理由，100字以内"
}"""

_CALL1_USER_ZH = """\
请分析以下多人会议记录，识别【%%TARGET_SPEAKER%%】的主导角色。

参会人员：%%ALL_SPEAKERS_LIST%%

会议记录：
%%MEETING_TRANSCRIPT%%

请严格按照 JSON 格式输出对【%%TARGET_SPEAKER%%】的角色识别结果。"""

# ── CALL 2a ─────────────────────────────────────────────────────────────

_CALL2A_SYS_ZH = f"""\
你是一名多人会议能力评测专家，专门评估参与者在多人会议场景中的特定能力表现。

{_RUBRIC_ZH}

---

## 维度 M1：多说话人状态追踪

【定义】
随着会议推进，参与者能否始终准确区分每个人说过什么、持有什么立场，不随时间推移产生混淆或错误归因。

【评估要点】
- 引用他人发言时，是否存在张冠李戴（把 A 说的归给 B）
- 在会议后段引用前段观点时，归因是否与原文一致
- 多人连续发言后，再描述某人立场时是否仍然准确
- 参与者人数≥4人时，追踪精度是否出现下降

【高分信号】有明确归因的引用且与原文一致；能清晰区分多人的不同立场
【低分信号】错误归因、立场混淆、遗漏某人曾说过的关键内容

---

## 维度 M2：差异化回应能力

【定义】
面对立场、需求或背景各不相同的多位参与者，能否给出有区分度的回应，而非对全体发出无差别消息。

【评估要点】
- 是否针对不同立场的参与者给出了不同的具体回应
- 是否存在忽略弱势方或沉默方的情况
- 面对分歧时是否用"大家都说得有道理"式无差别中性表态来规避真实差异化

【高分信号】明确对不同人说不同的话；在认可 A 的同时具体回应 B 的疑虑
【低分信号】对全场发出同一条消息；只回应发言量最大的人；沉默方被持续忽略

---

%%JSON_BLOCK%%"""

_CALL2A_USER_ZH = """\
请根据以下多人会议记录，评估【%%TARGET_SPEAKER%%】在 M1（多说话人状态追踪）和 M2（差异化回应能力）上的表现。

参会人员：%%ALL_SPEAKERS_LIST%%
待评估参与者：%%TARGET_SPEAKER%%

会议记录：
%%MEETING_TRANSCRIPT%%"""

# ── CALL 2b ─────────────────────────────────────────────────────────────

_CALL2B_SYS_ZH = f"""\
你是一名多人会议能力评测专家。

{_RUBRIC_ZH}

---

## 维度 M3：跨说话人信息整合

【定义】
关键信息分散在多个人的发言中时，参与者能否将碎片拼合成完整图景，识别出跨人的隐性共识或矛盾。

【评估要点】
- 总结或决策时是否覆盖了来自不同人的全部关键信息，还是只依赖最近几条
- 是否发现了分散在不同人、不同时间点发言中的逻辑矛盾
- 是否注意到发言较少者提供的关键信息（而非仅整合主要发言者的内容）
- 是否能识别出多人在不同措辞下表达的实质相同的观点

【高分信号】总结时能溯源到不同人的具体发言；发现跨轮次的矛盾或潜在共识
【低分信号】总结只反映了后段或发言量大者的观点；忽略早期或边缘发言者的信息

---

## 维度 M4：多方分歧处理

【定义】
当存在三方以上立场分歧时，参与者能否识别各方的核心诉求，找到多方可接受的整合出路，而非简单选边或搁置。

【评估要点】
- 当 A、B、C 各有不同立场时，是否只处理了最显著的那对冲突，遗漏第三方
- 是否能识别出某方案实际上包含多方诉求，将其作为整合点提出
- 给出的结论是否是有实质内容的收敛，还是"我们需要进一步讨论"式的无效搁置
- 是否因某方发言量更大而产生明显的立场偏袒

【高分信号】明确识别三方以上的各自核心诉求；找到包容多方诉求的整合方向
【低分信号】只处理两方冲突；给出无实质内容的搁置结论；偏袒发言量最大的一方

---

%%JSON_BLOCK%%"""

_CALL2B_USER_ZH = """\
请根据以下多人会议记录，评估【%%TARGET_SPEAKER%%】在 M3（跨说话人信息整合）和 M4（多方分歧处理）上的表现。

参会人员：%%ALL_SPEAKERS_LIST%%
待评估参与者：%%TARGET_SPEAKER%%

会议记录：
%%MEETING_TRANSCRIPT%%"""

# ── CALL 2c ─────────────────────────────────────────────────────────────

_CALL2C_SYS_ZH = f"""\
你是一名多人会议能力评测专家。

{_RUBRIC_ZH}

---

## 维度 M5：发言的群体意识

【定义】
参与者是否意识到自己的每一句话在多人场域中对所有在场者可见，并相应地考虑自己的表达对不同在场者的影响。

【评估要点】
- 在回应某人时，是否有意无意地忽视了其他在场者的立场或感受
- 当着持不同立场的人的面，是否会高度肯定某一方而不加任何平衡性说明
- 发言中是否使用了大量只有特定人才有背景的默认前提，使其他在场者无从理解
- 是否能利用多人在场的特性，借助一次发言同时向多方传递有价值的信息

【高分信号】回应 A 时主动照顾到 B、C 的感受或信息需求；利用公开发言推动全场共识
【低分信号】当着 B 的面强烈背书 A 的有争议方案；大量使用只有 A 才懂的内部信息

---

## 维度 M6：抗干扰与输入优先级判断

【定义】
当多位参与者同时或连续发出彼此竞争的输入时，参与者能否做出合理的优先级判断，而不是简单执行最近或最响亮的那一条。

【评估要点】
- 被最后一条消息完全覆盖，遗忘之前更重要的指令或上下文的情况
- 对主持人的议程指令和普通参与者的临时插话，处理优先级是否合理区分
- 某一方连续施压后，参与者是否在没有实质理由的情况下改变了立场
- 面对矛盾输入时，是否能显式指出冲突而非悄悄选边

【高分信号】明确说明处理优先级的理由；面对矛盾输入时主动澄清而非单方面执行
【低分信号】被最后一条消息覆盖历史指令；反复施压后无理由改变立场

---

%%JSON_BLOCK%%"""

_CALL2C_USER_ZH = """\
请根据以下多人会议记录，评估【%%TARGET_SPEAKER%%】在 M5（发言的群体意识）和 M6（抗干扰与输入优先级判断）上的表现。

参会人员：%%ALL_SPEAKERS_LIST%%
待评估参与者：%%TARGET_SPEAKER%%

会议记录：
%%MEETING_TRANSCRIPT%%"""

# ── CALL 3a ─────────────────────────────────────────────────────────────

_CALL3A_SYS_ZH = f"""\
你是一名会议表现评测专家，负责评估参与者在四个通用维度上的表现。这些维度适用于所有角色，不考虑参与者的专业知识是否准确，只评价会议行为本身。

{_RUBRIC_ZH}

---

## 维度 G1：表达清晰与可执行性

【定义】发言能否让听者准确理解，并据此知道下一步该如何讨论或行动。既看表述是否清楚，也看信息是否足够具体、可被执行或继续推进。

【评估要点】
- 是否说清楚了自己的意图、结论、建议或问题
- 是否存在大量模糊措辞，使他人难以接话或执行
- 发言是否提供了推进讨论所需的关键信息，而非只给情绪或碎片化反应

---

## 维度 G2：回应相关性与理解准确性

【定义】对他人发言的回应是否准确抓住了原意，回应内容是否紧扣当前问题，而不是答非所问、误解、空泛附和或机械重复。

【评估要点】
- 回应是否针对对方实际表达的核心点，而不是自己想象的版本
- 是否存在曲解、忽略关键限定条件、只挑容易回应部分的情况
- 回应是否提供了实质内容，帮助问题更清楚或更接近解决

---

## 维度 G3：议题贴合与节奏感知

【定义】发言是否与当前会议议题、当前阶段和讨论节奏相匹配，既不过早跳题，也不在需要收敛时继续无效扩散。

【评估要点】
- 是否围绕正在讨论的议题发言，而不是突然切换到无关内容
- 是否能感知会议处于探索、对齐、决策或收尾阶段，并给出匹配贡献
- 是否存在在已收敛时重启无关争论，或在需要回应时长期失联的情况

---

## 维度 G4：会议推进贡献

【定义】发言是否帮助会议向前推进，包括澄清问题、缩小分歧、形成阶段结论、明确下一步，而不是造成停滞、重复或无效循环。

【评估要点】
- 发言后，讨论是更清楚、更聚焦，还是更混乱、更发散
- 是否帮助把模糊观点转成可讨论选项、可执行动作或明确分歧
- 是否只是重复已有内容、表达态度，缺少实际推进价值

---

%%JSON_BLOCK%%"""

_CALL3A_USER_ZH = """\
请根据以下多人会议记录，评估【%%TARGET_SPEAKER%%】在 G1（表达清晰与可执行性）、G2（回应相关性与理解准确性）、G3（议题贴合与节奏感知）、G4（会议推进贡献）四个通用维度上的表现。

参会人员：%%ALL_SPEAKERS_LIST%%
待评估参与者：%%TARGET_SPEAKER%%

会议记录：
%%MEETING_TRANSCRIPT%%"""

# ── CALL 3b ─────────────────────────────────────────────────────────────

_CALL3B_SYS_ZH_BASE = f"""\
你是一名会议表现评测专家，负责评估参与者在其主导角色的专属维度上的表现。不考虑其专业知识的正确性，只评价会议行为本身。

{_RUBRIC_ZH}

---

%%ROLE_BLOCK%%"""

_CALL3B_USER_ZH = """\
请根据以下多人会议记录，评估【%%TARGET_SPEAKER%%】作为【%%PRIMARY_ROLE%%】在角色专属维度上的表现。

参会人员：%%ALL_SPEAKERS_LIST%%
待评估参与者：%%TARGET_SPEAKER%%
识别角色：%%PRIMARY_ROLE%%

会议记录：
%%MEETING_TRANSCRIPT%%

请严格按照 JSON 格式输出评估结果。"""

_ROLE_BLOCKS_ZH: Dict[str, str] = {
    "facilitator": """\
## F1：议程推进力
是否有效推动议题向前发展；议题切换是否自然且有逻辑；会议在有效时间内是否覆盖了预期议题。
【高分信号】主动宣布议题切换并说明理由；确保每个议题有结论后再推进
【低分信号】议题长时间原地打转；跳跃切换没有交代；议题遗漏

## F2：参与激活度
是否主动邀请不同成员发言；是否防止某人垄断发言时间；是否将沉默成员纳入讨论。
【高分信号】点名邀请沉默方发言；在某人发言过多时自然切换
【低分信号】只与活跃发言者互动；有人全程沉默未被关注

## F3：决策收敛能力
面对分歧时是否能有效归纳共识；是否能做出明确结论而非搁置；结论是否获得全场接受。
【高分信号】将分歧归纳为可决策的选项；明确宣布结论并确认接受度
【低分信号】将分歧留给"以后再说"；结论含糊无法执行

## F4：总结与闭环质量
阶段性总结是否准确；行动项是否明确到人和时间；会议结尾是否有下一步清单。
【高分信号】总结包含"谁在何时做什么"；与之前发言内容一致
【低分信号】总结内容与实际讨论有出入；行动项没有负责人或时间

## F5：中立与公平性
是否平等对待不同意见；是否有明显偏袒；能否在不同观点间保持客观立场。
【高分信号】对不同意见用相近措辞处理；不预设结论地主持讨论
【低分信号】对某人的意见明显更有热情；在结论形成前已暗示倾向""",

    "driver": """\
## D1：新方向提出能力
是否能提出此前未被充分讨论的新方向、新方案或新切入点，而非只重复他人意见。
【高分信号】带来新的讨论路径；清楚说明新方案与现有方案的区别
【低分信号】把已有想法换个说法重复；自称推进实际只是复述

## D2：方案可执行性
提出的方向是否与会议目标相关，是否能落到可讨论、可验证、可执行的层面。
【高分信号】给出明确做法、条件、资源或下一步；抽象想法能转成具体方案
【低分信号】想法空泛、脱离议题；无法进入实际讨论或执行

## D3：论证与推动力
提出观点时是否给出理由、依据或比较；被质疑时是否能继续推进而不是原地重复。
【高分信号】能解释为什么值得做；面对质疑能补充依据并推动继续讨论
【低分信号】只强调"我觉得"；被追问后重复原话或靠强势语气推进

## D4：推进节奏控制
是否知道何时继续推进、何时适度收束，避免过度推销单一想法占满讨论空间。
【高分信号】在合适时机推动进入选择或行动；不过度挤占他人参与空间
【低分信号】反复售卖同一方案；推进方式打断团队节奏或压制他人""",

    "collaborator": """\
## C1：延伸构建能力
是否能在他人观点基础上进行有实质价值的补充，而非仅表示同意。
【高分信号】明确说"在你说的X基础上，还可以加入Y"且Y有实质内容
【低分信号】"对，就是这个意思"式的无实质延伸；重复他人原话后称之为补充

## C2：整合能力
在多种对立意见存在时，是否能识别共同点并提出融合方案。
【高分信号】提出一个包含多方核心诉求的整合方案；明确说明整合了哪些方的哪些想法
【低分信号】只是声称"大家说的都有道理"但不给出整合路径

## C3：降低摩擦能力
是否能缓和不必要的对立、减少误解和情绪摩擦，让不同意见继续留在同一讨论框架里。
【高分信号】帮助双方重新理解彼此；用更容易接受的方式重述冲突点
【低分信号】表面调和但实质回避；或因措辞不当让矛盾升级

## C4：支持性推进质量
对他人的支持是否具体且有推进作用，能否帮助被忽视的有效观点进入主讨论。
【高分信号】明确说出某观点为什么有价值，并帮助其进入下一步讨论
【低分信号】大量空泛附和；只追随强势方，不帮助讨论前进""",

    "evaluator": """\
## E1：质疑准确性
指出的问题是否真实存在；是否能精准定位方案的薄弱点，而非泛泛而谈。
【高分信号】质疑直接指向方案的具体漏洞，且漏洞确实在原方案中存在
【低分信号】质疑方向与方案实际内容不符；质疑过于抽象无法验证

## E2：批评建设性
提出问题后是否有配套的改进方向或替代方案；是否仅否定而不提供出路。
【高分信号】"这里有个问题是X，一个可能的解决方式是Y"
【低分信号】只说"这样不行"而不提改进方向；连续质疑但从不建议改进

## E3：论据充分性
质疑是否有支撑理由；是否能用举例、类比或逻辑推演支撑批评，而非凭感觉否定。
【高分信号】用具体例子或推演链条说明为什么会出问题
【低分信号】"我感觉这不太对"式无理据质疑；无法在追问时进一步说明

## E4：从风险到行动的转化
指出问题后，是否能帮助团队明确下一步验证、补救或决策条件，而非只把讨论卡住。
【高分信号】把风险转成待验证事项、修改条件或决策门槛
【低分信号】连续指出问题但不给任何推进路径；让讨论长期停在否定层面""",
}


# ══════════════════════════════════════════════════════════════════════════
# ❺  English Prompt Templates
# ══════════════════════════════════════════════════════════════════════════

_CALL1_SYS_EN = """\
You are a professional meeting behavior analyst specializing in identifying participant role types.

## Role Definitions

**Facilitator**
Core traits: opens or closes the meeting, actively transitions topics (3+ times), assigns speaking turns, announces or confirms decisions.
Typical language: "Let's move on to the next item", "Could you share your thoughts?", "Let's go with that", "That's all for today."

**Driver**
Core traits: high proportion of independently initiated new ideas, leads discussion direction, puts forward concrete proposals or action recommendations — initiates more than responds.
Typical language: "I suggest…", "We could do it this way…", "I think the direction should be…"

**Collaborator**
Core traits: builds on others' ideas, integrates different opinions, facilitates consensus — extending others' ideas more often than independently initiating.
Typical language: "Building on what you said…", "These two points can be combined…", "I think everyone is actually saying something similar."

**Evaluator**
Core traits: challenges existing proposals, identifies risks or constraints, raises counter-examples — questioning utterances exceed 35% of total contributions.
Typical language: "But there's a problem here…", "What if this leads to…", "There's a premise we haven't discussed."

## Role Priority Rules
Facilitator > Driver > Evaluator > Collaborator (default fallback)

When two roles are clearly satisfied, record both a primary and a secondary role.

## Output Format (strict JSON, no additional text)
{
  "primary_role": "Facilitator|Driver|Collaborator|Evaluator",
  "secondary_role": "Facilitator|Driver|Collaborator|Evaluator|null",
  "confidence": "high|medium|low",
  "turn_count": <integer — total speaking turns by the target speaker>,
  "key_evidence": [
    "Evidence 1 (direct quote + which trait category it supports)",
    "Evidence 2",
    "Evidence 3"
  ],
  "reasoning": "Core reasoning for the role assignment, within 100 words"
}"""

_CALL1_USER_EN = """\
Please analyze the following multi-party meeting transcript and identify the dominant role of [%%TARGET_SPEAKER%%].

Participants: %%ALL_SPEAKERS_LIST%%

Meeting Transcript:
%%MEETING_TRANSCRIPT%%

Output the role identification result for [%%TARGET_SPEAKER%%] in strict JSON format."""

# ── CALL 2a ─────────────────────────────────────────────────────────────

_CALL2A_SYS_EN = f"""\
You are a multi-party meeting capability assessment expert specializing in evaluating participants' specific abilities in multi-person meeting contexts.

{_RUBRIC_EN}

---

## Dimension M1: Multi-Speaker State Tracking

[Definition]
As the meeting progresses, can the participant consistently and accurately distinguish what each person has said and what position they hold, without confusion or misattribution over time?

[Evaluation Points]
- When quoting others, does the participant misattribute statements (attribute A's words to B)?
- When referencing earlier statements later in the meeting, are attributions consistent with the original?
- After multiple people speak consecutively, is the participant still accurate when describing someone's position?
- When there are 4+ participants, does tracking accuracy noticeably decline?

[High Score Signals] Clear attributions that match the original; can clearly distinguish multiple speakers' different positions
[Low Score Signals] Misattribution, position confusion, failure to recall key things someone said earlier

---

## Dimension M2: Differentiated Response Capability

[Definition]
When facing multiple participants with different positions, needs, or backgrounds, can the participant give distinguishably different responses rather than sending identical messages to everyone?

[Evaluation Points]
- Does the participant give different specific responses to participants with different positions?
- Are weaker voices or silent participants ignored?
- When facing disagreement, does the participant avoid real differentiation by offering generic statements like "everyone makes a good point"?

[High Score Signals] Explicitly says different things to different people; while acknowledging A, specifically addresses B's concerns
[Low Score Signals] Sends the same message to the whole group; only responds to the most vocal participant; silent participants are consistently ignored

---

%%JSON_BLOCK%%"""

_CALL2A_USER_EN = """\
Please evaluate [%%TARGET_SPEAKER%%]'s performance on M1 (Multi-Speaker State Tracking) and M2 (Differentiated Response Capability) based on the following multi-party meeting transcript.

Participants: %%ALL_SPEAKERS_LIST%%
Participant being assessed: %%TARGET_SPEAKER%%

Meeting Transcript:
%%MEETING_TRANSCRIPT%%"""

# ── CALL 2b ─────────────────────────────────────────────────────────────

_CALL2B_SYS_EN = f"""\
You are a multi-party meeting capability assessment expert.

{_RUBRIC_EN}

---

## Dimension M3: Cross-Speaker Information Integration

[Definition]
When key information is scattered across multiple speakers' contributions, can the participant piece together a complete picture and identify implicit consensus or contradictions that span across different speakers?

[Evaluation Points]
- Do summaries or decisions cover key information from different people, or rely only on the most recent statements?
- Does the participant identify logical contradictions scattered across different speakers and time points?
- Does the participant notice key information from less vocal participants?
- Can the participant recognize when multiple people are expressing the same idea in different words?

[High Score Signals] Summaries trace back to specific statements from different speakers; identifies cross-turn contradictions or latent consensus
[Low Score Signals] Summaries only reflect the later part of the discussion or the most vocal participants; ignores early or peripheral contributions

---

## Dimension M4: Multi-Party Disagreement Handling

[Definition]
When there are three or more conflicting positions (not just a simple two-party standoff), can the participant identify each party's core concerns and find a mutually acceptable integrative path, rather than simply siding with one party or deferring?

[Evaluation Points]
- When A, B, and C each have different positions, does the participant only handle the most prominent conflict and miss the third party?
- Can the participant identify a proposal that actually contains multiple parties' concerns and present it as an integration point?
- Is the conclusion a substantive convergence, or an empty deferral like "we need to discuss this further"?
- Does the participant show obvious bias toward the party with the most speaking time?

[High Score Signals] Explicitly identifies the core concerns of 3+ parties; finds an integrative direction accommodating multiple parties
[Low Score Signals] Only handles two-party conflict; gives empty deferrals; biases toward the most vocal party

---

%%JSON_BLOCK%%"""

_CALL2B_USER_EN = """\
Please evaluate [%%TARGET_SPEAKER%%]'s performance on M3 (Cross-Speaker Information Integration) and M4 (Multi-Party Disagreement Handling) based on the following multi-party meeting transcript.

Participants: %%ALL_SPEAKERS_LIST%%
Participant being assessed: %%TARGET_SPEAKER%%

Meeting Transcript:
%%MEETING_TRANSCRIPT%%"""

# ── CALL 2c ─────────────────────────────────────────────────────────────

_CALL2C_SYS_EN = f"""\
You are a multi-party meeting capability assessment expert.

{_RUBRIC_EN}

---

## Dimension M5: Group Awareness in Speaking

[Definition]
Is the participant aware that everything they say is visible to all present in a multi-person context, and do they accordingly consider the impact of their words on different attendees — rather than speaking as if in a one-on-one conversation?

[Evaluation Points]
- When responding to one person, does the participant inadvertently ignore the positions or feelings of others present?
- In front of people with differing positions, does the participant strongly endorse one side without any balancing acknowledgment?
- Does the participant use large amounts of assumed context that only specific people would understand?
- Does the participant leverage the multi-person setting to deliver valuable information to multiple parties in a single utterance?

[High Score Signals] When responding to A, proactively accounts for B and C's needs; uses public statements to advance group consensus
[Low Score Signals] Strongly endorses A's controversial proposal in front of B; uses large amounts of insider language only A would understand

---

## Dimension M6: Noise Resistance and Input Priority Judgment

[Definition]
When multiple participants simultaneously or consecutively send competing inputs, can the participant make reasonable priority judgments rather than simply executing the most recent or loudest instruction?

[Evaluation Points]
- Is the participant completely overridden by the last message, forgetting earlier important instructions or context?
- Does the participant appropriately differentiate priority between the facilitator's agenda instructions and ordinary participants' spontaneous interruptions?
- After one party applies sustained pressure, does the participant change position without substantive justification?
- When facing contradictory inputs, does the participant explicitly flag the conflict rather than quietly siding with one?

[High Score Signals] Explicitly explains priority reasoning; proactively clarifies contradictory inputs rather than unilaterally executing
[Low Score Signals] Gets completely overridden by the last message; changes position without justification after repeated pressure

---

%%JSON_BLOCK%%"""

_CALL2C_USER_EN = """\
Please evaluate [%%TARGET_SPEAKER%%]'s performance on M5 (Group Awareness in Speaking) and M6 (Noise Resistance and Input Priority Judgment) based on the following multi-party meeting transcript.

Participants: %%ALL_SPEAKERS_LIST%%
Participant being assessed: %%TARGET_SPEAKER%%

Meeting Transcript:
%%MEETING_TRANSCRIPT%%"""

# ── CALL 3a ─────────────────────────────────────────────────────────────

_CALL3A_SYS_EN = f"""\
You are a meeting performance assessment expert evaluating a participant across four universal dimensions. These apply to all roles — assess meeting behaviors only, not domain knowledge accuracy.

{_RUBRIC_EN}

---

## Dimension G1: Clarity and Actionability

[Definition] Can the participant's contributions be clearly understood, and do they contain enough concrete information for others to react, decide, or act? Assess both clarity and practical usefulness.

[Evaluation Points]
- Does the participant clearly express the point, proposal, concern, or conclusion?
- Is the wording too vague for others to know how to respond or act?
- Does the utterance provide concrete information that helps move discussion forward?

---

## Dimension G2: Relevance and Accuracy of Response

[Definition] When responding to others, does the participant address what was actually said, with a response that is relevant, accurate, and substantive rather than off-target, misunderstood, or empty?

[Evaluation Points]
- Does the response address the speaker's actual core point rather than an imagined version?
- Does it miss important qualifiers, distort the issue, or only answer the easiest part?
- Does it add substance that clarifies, advances, or resolves the issue?

---

## Dimension G3: Agenda Fit and Rhythm Awareness

[Definition] Are the participant's contributions aligned with the current topic and stage of the meeting? Assess whether they sense when to explore, align, decide, or wrap up.

[Evaluation Points]
- Do contributions stay on the active agenda item rather than jumping elsewhere?
- Does the participant match the current stage of discussion instead of restarting or derailing it?
- When input is needed, do they contribute at the right moment rather than remaining absent or out of sync?

---

## Dimension G4: Contribution to Meeting Progress

[Definition] Do the participant's contributions help the meeting move forward by clarifying issues, narrowing disagreement, surfacing decisions, or defining next steps, rather than causing stagnation, repetition, or circular talk?

[Evaluation Points]
- After the participant speaks, does the discussion become clearer, more focused, or more actionable?
- Do they help transform vague ideas into options, decisions, or next steps?
- Do they mainly repeat prior content or express stance without adding forward motion?

---

%%JSON_BLOCK%%"""

_CALL3A_USER_EN = """\
Please evaluate [%%TARGET_SPEAKER%%]'s performance on G1 (Clarity and Actionability), G2 (Relevance and Accuracy of Response), G3 (Agenda Fit and Rhythm Awareness), and G4 (Contribution to Meeting Progress) based on the following multi-party meeting transcript.

Participants: %%ALL_SPEAKERS_LIST%%
Participant being assessed: %%TARGET_SPEAKER%%

Meeting Transcript:
%%MEETING_TRANSCRIPT%%"""

# ── CALL 3b ─────────────────────────────────────────────────────────────

_CALL3B_SYS_EN_BASE = f"""\
You are a meeting performance assessment expert evaluating a participant on role-specific dimensions for their dominant role. Assess meeting behaviors only, not domain knowledge correctness.

{_RUBRIC_EN}

---

%%ROLE_BLOCK%%"""

_CALL3B_USER_EN = """\
Please evaluate [%%TARGET_SPEAKER%%]'s performance on the role-specific dimensions for their role as %%PRIMARY_ROLE%%, based on the following multi-party meeting transcript.

Participants: %%ALL_SPEAKERS_LIST%%
Participant being assessed: %%TARGET_SPEAKER%%
Identified role: %%PRIMARY_ROLE%%

Meeting Transcript:
%%MEETING_TRANSCRIPT%%

Output the evaluation results in strict JSON format."""

_ROLE_BLOCKS_EN: Dict[str, str] = {
    "facilitator": """\
## F1: Agenda Momentum
Does the participant effectively advance topics? Are transitions natural and logical? Does the meeting cover expected agenda items within available time?
[High] Proactively announces topic transitions with reasoning; ensures each item has a conclusion before moving on
[Low] Discussion circles on the same point too long; abrupt jumps without explanation; agenda items skipped

## F2: Participation Activation
Does the participant actively invite different members to speak? Do they prevent any one person from dominating? Do they bring silent members into the discussion?
[High] Explicitly invites silent participants by name; naturally redirects when someone has been speaking too long
[Low] Only interacts with already-active speakers; someone stays silent throughout without being noticed

## F3: Decision Convergence
When facing disagreements, does the participant effectively summarize consensus? Can they make clear decisions rather than deferring? Are conclusions accepted by the group?
[High] Reframes disagreements as actionable choices; clearly announces conclusions and confirms acceptance
[Low] Leaves disagreements to "be discussed later"; conclusions are too vague to act on

## F4: Summary and Closure Quality
Are periodic summaries accurate? Are action items assigned to specific people with deadlines? Does the meeting end with a clear next-steps list?
[High] Summary includes "who does what by when"; consistent with what was actually discussed
[Low] Summary content diverges from actual discussion; action items have no owner or deadline

## F5: Neutrality and Fairness
Does the participant treat different viewpoints equally? Is there obvious favoritism? Can they maintain an objective stance?
[High] Uses comparable language when handling different viewpoints; facilitates without presupposing outcomes
[Low] Noticeably more enthusiastic about one person's ideas; signals preference before conclusions are formed""",

    "driver": """\
## D1: New-Direction Initiation
Can the participant introduce genuinely new directions, proposals, or frames rather than merely rephrasing what others already said?
[High] Opens a direction that had not yet been explored and clearly distinguishes it from prior ideas
[Low] Repackages existing ideas as if they were new; adds motion without new substance

## D2: Proposal Actionability
Are the ideas relevant to the meeting goal and concrete enough to be discussed, tested, or executed?
[High] Gives a clear path, conditions, resources, or next steps; turns abstract thoughts into workable proposals
[Low] Produces vague or off-topic ideas that cannot practically move into execution

## D3: Argumentation and Push
Does the participant support ideas with reasons, comparisons, or evidence, and keep the discussion moving when challenged?
[High] Explains why the proposal is worth doing and responds to objections with substantive support
[Low] Relies on repetition, confidence, or pressure rather than reasoning

## D4: Pace Control
Can the participant push discussion forward without monopolizing the room or overselling one direction past its useful point?
[High] Pushes toward selection or action at the right time while preserving others' room to contribute
[Low] Repeatedly pushes the same idea in a way that distorts the meeting rhythm or suppresses others""",

    "collaborator": """\
## C1: Extension and Building Ability
Can the participant add substantive value on top of others' ideas, rather than merely expressing agreement?
[High] Explicitly says "building on your X, we could also add Y" where Y has real substance
[Low] "Yes, exactly" style with no substantive extension; repeats someone else's words as a contribution

## C2: Integration Ability
When opposing views coexist, can the participant identify common ground and propose a synthesis?
[High] Proposes an integrated solution capturing multiple parties' core concerns; names what was integrated
[Low] Only claims "everyone has a point" without offering an integration path

## C3: Friction Reduction
Can the participant reduce unnecessary tension or misunderstanding so that disagreement remains discussable?
[High] Helps parties better understand each other and rephrases conflict in a more workable frame
[Low] Offers superficial harmony or, through poor framing, makes conflict worse

## C4: Supportive Progress
When supporting others, does the participant make that support concrete and useful enough to help the idea advance?
[High] Explains why a point matters and helps bring overlooked useful ideas into the main discussion
[Low] Provides empty affirmation or only echoes the strongest voice without helping progress""",

    "evaluator": """\
## E1: Accuracy of Challenges
Do the problems identified actually exist? Can the participant pinpoint the weak spots with precision?
[High] Challenges directly target specific flaws in the proposal that genuinely exist
[Low] Challenges are misaligned with actual proposal content; challenges too abstract to verify

## E2: Constructiveness of Criticism
After raising a problem, does the participant offer an improvement direction or an alternative?
[High] "The issue here is X; one possible solution would be Y"
[Low] Only says "that won't work" without suggesting improvements; raises objections repeatedly without ever recommending changes

## E3: Sufficiency of Evidence
Are challenges supported by reasoning? Does the participant use examples, analogies, or logical chains?
[High] Uses specific examples or reasoning chains to explain why something would go wrong
[Low] "I just feel like this isn't right" style without evidence; unable to elaborate further when pressed

## E4: Risk-to-Action Conversion
After identifying risks, can the participant help the group define validation steps, conditions, or modifications rather than leaving the discussion stuck at rejection?
[High] Converts concerns into checks, decision gates, or concrete revisions
[Low] Repeatedly blocks momentum with criticism while offering no path to proceed""",
}


# ══════════════════════════════════════════════════════════════════════════
# ❻  Template Registry
# ══════════════════════════════════════════════════════════════════════════

_TEMPLATE_REGISTRY: Dict[str, Dict[str, tuple]] = {
    "zh": {
        "call1":  (_CALL1_SYS_ZH,       _CALL1_USER_ZH),
        "call2a": (_CALL2A_SYS_ZH,      _CALL2A_USER_ZH),
        "call2b": (_CALL2B_SYS_ZH,      _CALL2B_USER_ZH),
        "call2c": (_CALL2C_SYS_ZH,      _CALL2C_USER_ZH),
        "call3a": (_CALL3A_SYS_ZH,      _CALL3A_USER_ZH),
        "call3b": (_CALL3B_SYS_ZH_BASE, _CALL3B_USER_ZH),
    },
    "en": {
        "call1":  (_CALL1_SYS_EN,       _CALL1_USER_EN),
        "call2a": (_CALL2A_SYS_EN,      _CALL2A_USER_EN),
        "call2b": (_CALL2B_SYS_EN,      _CALL2B_USER_EN),
        "call2c": (_CALL2C_SYS_EN,      _CALL2C_USER_EN),
        "call3a": (_CALL3A_SYS_EN,      _CALL3A_USER_EN),
        "call3b": (_CALL3B_SYS_EN_BASE, _CALL3B_USER_EN),
    },
}

_ROLE_BLOCKS_ALL: Dict[str, Dict[str, str]] = {
    "zh": _ROLE_BLOCKS_ZH,
    "en": _ROLE_BLOCKS_EN,
}


# ══════════════════════════════════════════════════════════════════════════
# ❼  PromptBuilder
# ══════════════════════════════════════════════════════════════════════════

class PromptBuilder:
    """
    Assembles all prompt pairs for the 6-call MPIQ evaluation pipeline.

    Execution order::

        call1  (sequential)
          └──► call2a ─┐
               call2b  ├─ parallel group "mpiq"
               call2c ─┘
               call3a ─┐
               call3b  ├─ parallel group "dim"  (requires primary_role)
                       ┘
    """

    # ── internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _tmpl(inp: PipelineInput, call_id: str) -> tuple:
        return _TEMPLATE_REGISTRY[inp.language.value][call_id]

    @staticmethod
    def _base_vars(inp: PipelineInput) -> Dict[str, str]:
        return {
            "TARGET_SPEAKER":     inp.target_speaker,
            "ALL_SPEAKERS_LIST":  inp.speakers_str,
            "MEETING_TRANSCRIPT": inp.transcript,
        }

    def _call2x(self, inp: PipelineInput, call_id: str) -> PromptPair:
        dims       = _CALL2_DIMS[call_id]
        sys_t, usr_t = self._tmpl(inp, call_id)
        json_block = _build_json_block(dims, inp.language)
        system     = _render(sys_t, JSON_BLOCK=json_block)
        user       = _render(usr_t, **self._base_vars(inp))
        desc = {
            "call2a": "M1 Multi-Speaker Tracking  +  M2 Differentiated Response",
            "call2b": "M3 Cross-Speaker Integration  +  M4 Multi-Party Disagreement",
            "call2c": "M5 Group Awareness  +  M6 Noise Resistance",
        }[call_id]
        return PromptPair(call_id=call_id, system=system, user=user,
                          parallel_group="mpiq", description=desc)

    # ── public build methods ─────────────────────────────────────────────

    def build_call1(self, inp: PipelineInput) -> PromptPair:
        """CALL 1 — role identification. Must run first (sequential)."""
        inp.validate()
        sys_t, usr_t = self._tmpl(inp, "call1")
        return PromptPair(
            call_id="call1",
            system=sys_t,
            user=_render(usr_t, **self._base_vars(inp)),
            parallel_group=None,
            description="Role identification",
        )

    def build_call2a(self, inp: PipelineInput) -> PromptPair:
        """CALL 2a — dimensions M1 + M2."""
        return self._call2x(inp, "call2a")

    def build_call2b(self, inp: PipelineInput) -> PromptPair:
        """CALL 2b — dimensions M3 + M4."""
        return self._call2x(inp, "call2b")

    def build_call2c(self, inp: PipelineInput) -> PromptPair:
        """CALL 2c — dimensions M5 + M6."""
        return self._call2x(inp, "call2c")

    def build_call3a(self, inp: PipelineInput) -> PromptPair:
        """CALL 3a — universal dimensions G1 + G2 + G3 + G4."""
        sys_t, usr_t = self._tmpl(inp, "call3a")
        json_block   = _build_json_block(["G1", "G2", "G3", "G4"], inp.language)
        system       = _render(sys_t, JSON_BLOCK=json_block)
        user         = _render(usr_t, **self._base_vars(inp))
        return PromptPair(
            call_id="call3a", system=system, user=user,
            parallel_group="dim",
            description="G1 Clarity/Actionability  +  G2 Response Accuracy  +  G3 Agenda Fit  +  G4 Progress",
        )

    def build_call3b(self, inp: PipelineInput) -> PromptPair:
        """
        CALL 3b — role-specific dimensions.
        **Requires** ``inp.primary_role`` to be set (from CALL 1 result).
        """
        if inp.primary_role is None:
            raise ValueError(
                "inp.primary_role must be set before building CALL 3b. "
                "Parse the CALL 1 JSON result and assign inp.primary_role first."
            )
        role_key = inp.role_key()
        if role_key is None:
            raise ValueError(
                f"Unrecognized primary_role '{inp.primary_role}'. "
                f"Accepted values: {sorted(ROLE_CANONICAL.keys())}"
            )
        lang            = inp.language.value
        sys_t, usr_t    = self._tmpl(inp, "call3b")
        role_block      = _ROLE_BLOCKS_ALL[lang][role_key]
        role_label      = ROLE_LABELS[role_key][lang]
        dim_codes       = _ROLE_DIM_CODES[role_key]
        json_block      = _build_json_block(dim_codes, inp.language)
        full_role_block = role_block + "\n\n" + json_block
        system = _render(sys_t, ROLE_BLOCK=full_role_block)
        user   = _render(usr_t, PRIMARY_ROLE=role_label, **self._base_vars(inp))
        return PromptPair(
            call_id="call3b", system=system, user=user,
            parallel_group="dim",
            description=f"{role_label} role-specific dims ({', '.join(dim_codes)})",
        )

    # ── group convenience methods ────────────────────────────────────────

    def build_mpiq_calls(self, inp: PipelineInput) -> List[PromptPair]:
        """Return CALL 2a, 2b, 2c  (parallel group 'mpiq')."""
        return [self.build_call2a(inp),
                self.build_call2b(inp),
                self.build_call2c(inp)]

    def build_dim_calls(self, inp: PipelineInput) -> List[PromptPair]:
        """Return CALL 3a, 3b  (parallel group 'dim')."""
        return [self.build_call3a(inp),
                self.build_call3b(inp)]

    def build_remaining(self, inp: PipelineInput) -> List[PromptPair]:
        """Return all 5 calls that follow CALL 1."""
        return self.build_mpiq_calls(inp) + self.build_dim_calls(inp)

    def build_all(self, inp: PipelineInput) -> List[PromptPair]:
        """
        Return all 6 calls in pipeline order.
        Note: CALL 3b requires ``inp.primary_role`` to already be set.
        Parallel groups are annotated but not enforced — caller handles concurrency.
        """
        return [self.build_call1(inp)] + self.build_remaining(inp)


# ══════════════════════════════════════════════════════════════════════════
# ❽  ScoreCalculator
# ══════════════════════════════════════════════════════════════════════════

class ScoreCalculator:
    """
    Computes the final 0–100 score.

    Formula
    ───────
    For each LLM-judged group X ∈ {General, MPIQ, Role}:
        group_pct(X) = mean over X's dims of (score − 1) / 4 × 100   ∈ [0, 100]
        N/A is treated as score 1 (rubric minimum).

    The mechanical participation coefficient is computed *per sample* against
    the human ground-truth ratio for that same (meeting, target speaker)::

        coef = max(0, 1 − |model_ratio − gt_ratio|)             ∈ [0, 1]

    So a model that matches the human's natural participation level pays no
    penalty (coef = 1); a silent model loses ≈ 1 − gt; a monologue model on
    a sample where gt is already near 1.0 loses very little.

    The coefficient multiplies each group score::

        G = group_pct(G1..G4)            × coef
        M = group_pct(M1..M6)            × coef
        R = group_pct(role-specific 4/5) × coef
        final_score = (G + M + R) / 3   ∈ [0, 100]

    Equivalently ``final_score = avg(G_pct, M_pct, R_pct) × coef``.
    Aggregate metrics across samples are computed by averaging this per-sample
    final_score (not by averaging coefs and group means independently).

    The three LLM groups are equally weighted (1/3 each). P1 (= 1 + 4·ratio)
    remains in ``scores`` for human-readable reporting only.
    """

    GENERAL_DIMS = ("G1", "G2", "G3", "G4")
    MPIQ_DIMS    = ("M1", "M2", "M3", "M4", "M5", "M6")

    GRADE_THRESHOLDS = [
        (90, "A+"), (85, "A"), (80, "A-"),
        (75, "B+"), (70, "B"), (65, "B-"),
        (60, "C+"), (55, "C"), (50, "C-"),
        (0,  "F"),
    ]

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalised(score) -> Tuple[float, bool]:
        """Return ((score − 1) / 4 × 100, is_na). N/A is normalised to 0."""
        if score is None or str(score).strip().upper() == "N/A":
            return 0.0, True
        s = float(score)
        return (s - 1) / 4 * 100, False

    def _group_pct(
        self,
        scores: Dict[str, object],
        dims: Sequence[str],
        na_dims: List[str],
    ) -> Tuple[float, Dict[str, Dict]]:
        if not dims:
            return 0.0, {}
        per_dim: Dict[str, Dict] = {}
        total = 0.0
        for d in dims:
            pct, is_na = self._normalised(scores.get(d))
            if is_na:
                na_dims.append(d)
            per_dim[d] = {"score": scores.get(d), "pct": round(pct, 2), "na": is_na}
            total += pct
        return total / len(dims), per_dim

    # ── main entry point ─────────────────────────────────────────────────

    def calculate(
        self,
        scores: Dict[str, object],             # dim_code → int (1–5) or "N/A" / None
        role_key: str,
        participation_ratio: float = 0.0,      # ∈ [0, 1] — model's actual ratio
        gt_participation_ratio: Optional[float] = None,  # ∈ [0, 1] — human GT for THIS sample
    ) -> Dict:
        """
        Parameters
        ----------
        scores                 : flat dict of dimension scores
        role_key               : canonical role key, e.g. ``"facilitator"``
        participation_ratio    : the model's mechanical span ratio.
        gt_participation_ratio : the human ground-truth span ratio for this same
                                 (meeting, target speaker). When provided, the
                                 multiplicative coefficient is
                                 ``coef = max(0, 1 − |participation_ratio −
                                 gt_participation_ratio|)``. When ``None``, the
                                 coefficient falls back to ``participation_ratio``
                                 (legacy behaviour: monotone in presence).

        Returns
        -------
        dict with keys:
          final_score          float  (0–100)
          grade                str
          participation_ratio  float  (echo of input, clamped)
          gt_participation_ratio  float | None  (echo)
          coef                 float  multiplicative coefficient applied
          groups               {general, mpiq, role} → {pct_pre, pct_post, weight, dims}
          na_dims              list of dimensions scored N/A
        """
        ratio = max(0.0, min(1.0, float(participation_ratio or 0.0)))
        if gt_participation_ratio is None:
            gt = None
            coef = ratio
        else:
            gt = max(0.0, min(1.0, float(gt_participation_ratio)))
            coef = max(0.0, 1.0 - abs(ratio - gt))
        role_dims = list(_ROLE_DIM_CODES.get(role_key, []))

        na_dims: List[str] = []
        g_pct, g_breakdown = self._group_pct(scores, self.GENERAL_DIMS, na_dims)
        m_pct, m_breakdown = self._group_pct(scores, self.MPIQ_DIMS, na_dims)
        r_pct, r_breakdown = self._group_pct(scores, role_dims, na_dims)

        groups = {
            "general":  {"pct_pre": round(g_pct, 2), "pct_post": round(g_pct * coef, 2),
                         "weight":  1 / 3, "dims": g_breakdown},
            "mpiq":     {"pct_pre": round(m_pct, 2), "pct_post": round(m_pct * coef, 2),
                         "weight":  1 / 3, "dims": m_breakdown},
            "role":     {"pct_pre": round(r_pct, 2), "pct_post": round(r_pct * coef, 2),
                         "weight":  1 / 3, "dims": r_breakdown,
                         "role_key": role_key},
        }

        final = round(((g_pct + m_pct + r_pct) / 3.0) * coef, 1)
        return {
            "final_score":            final,
            "grade":                  self.grade(final),
            "participation_ratio":    round(ratio, 4),
            "gt_participation_ratio": (round(gt, 4) if gt is not None else None),
            "coef":                   round(coef, 4),
            "groups":                 groups,
            "na_dims":                na_dims,
        }

    @staticmethod
    def grade(score: float) -> str:
        for threshold, letter in ScoreCalculator.GRADE_THRESHOLDS:
            if score >= threshold:
                return letter
        return "F"


# ══════════════════════════════════════════════════════════════════════════
# ❾  Parse Helpers
# ══════════════════════════════════════════════════════════════════════════

_SCORE_FIELD_RE = re.compile(
    r'"(?P<dim>[A-Z]\d)"\s*:\s*(?:\{\s*"score"\s*:\s*(?P<nested>"N/?A"|[1-5])|(?P<flat>"N/?A"|[1-5]))',
    flags=re.IGNORECASE | re.DOTALL,
)


def score_dims_for_call(call_id: str, primary_role: Optional[str] = None) -> List[str]:
    """Return the expected score dimension keys for a score-producing call."""
    if call_id in _CALL2_DIMS:
        return list(_CALL2_DIMS[call_id])
    if call_id == "call3a":
        return ["G1", "G2", "G3", "G4"]
    if call_id == "call3b":
        if primary_role is None:
            raise ValueError("primary_role is required for call3b score dims")
        role_key = ROLE_CANONICAL.get(primary_role)
        if role_key is None:
            raise ValueError(
                f"Unknown primary_role '{primary_role}' for call3b dims"
            )
        return list(_ROLE_DIM_CODES[role_key])
    raise ValueError(f"Unsupported score call id: {call_id}")


def _clean_llm_json_output(llm_output: str) -> str:
    cleaned = llm_output.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _load_first_json_dict(llm_output: str) -> Dict:
    cleaned = _clean_llm_json_output(llm_output)
    decoder = json.JSONDecoder()
    first_error: Optional[json.JSONDecodeError] = None

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level JSON object, got {type(data).__name__}")
        return data
    except json.JSONDecodeError as exc:
        first_error = exc

    start = cleaned.find("{")
    while start >= 0:
        try:
            data, _end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            start = cleaned.find("{", start + 1)
            continue
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level JSON object, got {type(data).__name__}")
        return data

    if first_error is not None:
        raise first_error
    raise ValueError("No JSON object found in LLM output")


def _normalize_score_value(value: object) -> object:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        stripped = value.strip().strip('"')
        if stripped.upper() == "N/A":
            return "N/A"
        if stripped.isdigit():
            value = int(stripped)
        else:
            raise ValueError(f"Invalid score value: {value!r}")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        if 1 <= value <= 5:
            return value
        raise ValueError(f"Score must be within 1-5, got {value}")
    raise ValueError(f"Invalid score value type: {type(value).__name__}")


def _extract_scores_by_regex(
    llm_output: str,
    *,
    expected_dims: Optional[List[str]] = None,
) -> Dict[str, object]:
    cleaned = _clean_llm_json_output(llm_output)
    extracted: Dict[str, object] = {}

    for match in _SCORE_FIELD_RE.finditer(cleaned):
        dim_code = match.group("dim")
        raw_score = match.group("nested") or match.group("flat")
        if expected_dims is not None and dim_code not in expected_dims:
            continue
        extracted[dim_code] = _normalize_score_value(raw_score)

    if expected_dims is not None:
        missing = [dim for dim in expected_dims if dim not in extracted]
        if missing:
            raise ValueError(
                f"Missing score fields after regex fallback: {', '.join(missing)}"
            )
        return {dim: extracted[dim] for dim in expected_dims}

    if not extracted:
        raise ValueError("Could not extract any score fields from LLM output")
    return extracted

def parse_call1_result(llm_output: str) -> Dict:
    """
    Extract and validate the JSON payload from a CALL 1 LLM response.
    Strips markdown code fences if present. Returns the parsed dict.
    """
    data = _load_first_json_dict(llm_output)
    required = {"primary_role", "secondary_role", "confidence", "turn_count"}
    missing  = required - set(data.keys())
    if missing:
        raise ValueError(f"CALL 1 result is missing required keys: {missing}")
    if data["primary_role"] not in ROLE_CANONICAL:
        raise ValueError(
            f"Unknown primary_role '{data['primary_role']}'. "
            f"Expected one of: {sorted(ROLE_CANONICAL.keys())}"
        )
    return data


def parse_score_result(
    llm_output: str,
    expected_dims: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Extract dimension scores from any CALL 2x / CALL 3x LLM response.
    Returns a flat dict: ``{dim_code: score_value, ...}``
    where score_value is an int (1–5) or the string "N/A".
    """
    try:
        data = _load_first_json_dict(llm_output)
        flat: Dict[str, object] = {}
        for dim_code, content in data.items():
            flat[dim_code] = (
                content.get("score", "N/A")
                if isinstance(content, dict)
                else content
            )
        normalized = {
            dim_code: _normalize_score_value(score)
            for dim_code, score in flat.items()
        }
        if expected_dims is not None:
            missing = [dim for dim in expected_dims if dim not in normalized]
            if missing:
                raise ValueError(
                    f"CALL score result is missing required dims: {', '.join(missing)}"
                )
            return {dim: normalized[dim] for dim in expected_dims}
        return normalized
    except (json.JSONDecodeError, ValueError, TypeError):
        return _extract_scores_by_regex(llm_output, expected_dims=expected_dims)


def merge_scores(*score_dicts: Dict[str, object]) -> Dict[str, object]:
    """Merge multiple parse_score_result() outputs into one flat dict."""
    merged: Dict[str, object] = {}
    for d in score_dicts:
        merged.update(d)
    return merged


# ══════════════════════════════════════════════════════════════════════════
# ❿  CLI Demo
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import textwrap

    SAMPLE = textwrap.dedent("""\
        Alice: Let's kick off — today we need to decide on the Q3 marketing budget.
        Bob:   I think we should increase digital spend by at least 30%.
        Carol: I'm not sure. Last quarter's digital ROI was below target.
        Dave:  We also haven't talked about the event budget yet.
        Alice: Good point, Dave. Bob, can you elaborate on which channels?
        Bob:   Mainly paid social and search — both showed strong early signals.
        Carol: The early signals were only two weeks of data, though.
        Alice: Dave, what's your take on events vs. digital trade-offs?
        Dave:  Events drive enterprise deals. I'd keep that budget flat.
        Alice: OK — so Bob proposes +30% digital, Carol flags data risk,
               Dave wants to preserve events. Let me suggest a middle path...
    """)

    inp = PipelineInput(
        transcript=SAMPLE,
        target_speaker="Alice",
        all_speakers=["Alice", "Bob", "Carol", "Dave"],
        language=Language.EN,
    )
    builder = PromptBuilder()

    # ── CALL 1 ──────────────────────────────────────────────────────────
    print("=" * 68)
    print("CALL 1 — Role Identification")
    print("=" * 68)
    c1 = builder.build_call1(inp)
    print(f"[System — first 200 chars]\n{c1.system[:200]}...\n")
    print(f"[User]\n{c1.user}\n")

    # Simulate CALL 1 LLM result
    inp.primary_role = "Facilitator"
    print(f"→ Simulated CALL 1 result: primary_role = '{inp.primary_role}'\n")

    # ── Remaining calls ──────────────────────────────────────────────────
    print("=" * 68)
    print("Remaining calls (parallel groups annotated)")
    print("=" * 68)
    for p in builder.build_remaining(inp):
        print(f"  {p.call_id:<8}  group={str(p.parallel_group):<6}  {p.description}")

    # ── ScoreCalculator demo ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("ScoreCalculator demo  (Facilitator, no N/A dims)")
    print("=" * 68)
    sample_scores = {
        "G1": 4, "G2": 5, "G3": 4, "G4": 4,
        "M1": 5, "M2": 4, "M3": 3, "M4": 4, "M5": 4, "M6": 3,
        "F1": 5, "F2": 4, "F3": 4, "F4": 3, "F5": 5,
    }
    result = ScoreCalculator().calculate(
        sample_scores, role_key="facilitator", participation_ratio=0.85,
    )
    print(f"  Final score        : {result['final_score']}")
    print(f"  Grade              : {result['grade']}")
    print(f"  Participation x    : {result['participation_ratio']}")
    print(f"  N/A dims           : {result['na_dims'] or 'none'}")
    print("\n  Group breakdown:")
    for name, g in result["groups"].items():
        print(f"    {name:<8}  pct_pre={g['pct_pre']:>6.2f}  "
              f"pct_post={g['pct_post']:>6.2f}  "
              f"weight={g['weight']:.3f}")

    # ── Language switch demo ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Language switch demo  (ZH)")
    print("=" * 68)
    inp_zh = PipelineInput(
        transcript="Alice：我们来讨论一下第三季度预算…",
        target_speaker="Alice",
        all_speakers=["Alice", "Bob", "Carol"],
        language=Language.ZH,
        primary_role="主持人",
    )
    c3b_zh = builder.build_call3b(inp_zh)
    print(f"CALL 3b (ZH) description : {c3b_zh.description}")
    print(f"System prompt first line : {c3b_zh.system.splitlines()[0]}")
