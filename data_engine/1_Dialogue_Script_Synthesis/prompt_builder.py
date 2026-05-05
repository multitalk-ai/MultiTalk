"""
Prompt 构建模块（两阶段）

阶段一：给定维度参数 + 种子句子，让 LLM 生成场景描述和角色画像（含名字）
阶段二：给定阶段一的输出，让 LLM 生成对话
"""

from scenario_config import (
    ScenarioConfig,
    LANGUAGE_INSTRUCTIONS,
    AI_INVOLVEMENT_INSTRUCTIONS,
    INDUSTRY_TOPICS
)
import random


# ============================================================
# 阶段一：场景 + 角色生成
# ============================================================

STAGE1_SYSTEM_PROMPT = """\
You are a creative writing assistant that designs realistic conversation scenarios \
and detailed character profiles. You create vivid, differentiated characters with \
real names, distinct personalities, and specific relationships to the conversation topic. \
ALL participants — including the one marked role "assistant" — must be described as \
real human beings with human backstories. Never mention AI, bots, or virtual assistants. \
Your output is always valid JSON with straight double quotes only."""

STAGE1_SYSTEM_PROMPT_ROLE = """\
You are a creative writing assistant that designs conversation scenarios \
and detailed character profiles. You create vivid, differentiated characters with \
real names, distinct personalities, and specific relationships to the conversation topic. \
ALL participants must be described as human beings with backstories. \
Your output is always valid JSON with straight double quotes only."""

STAGE1_SYSTEM_PROMPT_WORK_MEETING = """\
You are a creative writing assistant that designs realistic work meeting scenarios \
and detailed character profiles. You create vivid, differentiated characters with \
real names, distinct personalities, and specific relationships to the discussion topic. \
ALL participants — including the one marked role "assistant" — must be described as \
real human beings with professional backgrounds and work-related backstories. \
Never mention AI, bots, or virtual assistants. \
The discussion content must be professional, reflecting the conventions of a real workplace meeting. \
Your output is always valid JSON with straight double quotes only."""

STAGE1_SYSTEM_PROMPT_AI_ASSISTANT = """\
You are a creative writing assistant that designs realistic conversation scenarios \
between human users and an AI voice assistant. The participant marked role "assistant" \
IS an AI voice assistant (similar to Siri, Alexa, ChatGPT voice mode, or Claude voice) — \
a real AI tool the humans interact with through speech, NOT a disguised human. \
All other participants ARE real human beings with full human backstories. \
The AI assistant has no human backstory; it has a distinctive voice persona — \
its tone, helpfulness style, characteristic phrasing, and how it handles requests. \
Your output is always valid JSON with straight double quotes only."""

STAGE1_USER_TEMPLATE = """\
{first_sentence}

## Fixed Constraints
- Number of participants with role "user": {num_users} (these are the regular human participants)
- Plus 1 participant with role "assistant" (see requirement #2)
- Total participants: {num_users} + 1 = {total_participants}
- Language: {language_instruction}

## Seed Material
The conversation should be inspired by the following sentence and emotion. \
One of the human participants should be the "seed holder" — the person whose \
experience or feeling aligns with this seed.

Sentence: "{seed_sentence}"
Emotion: {seed_emotion}

{predefined_assistant_block}

{qa_patch}

## Creative Design (your job — do NOT default to the obvious)
Based purely on the seed above, you must freely invent the following dimensions. \
Do NOT gravitate toward the most predictable interpretation of the seed. \
A seed about loneliness can happen inside a loud party. A seed about anger can \
surface as cold silence. A seed about failure can be the backdrop for a celebration. \
Surprise yourself.

For each dimension, choose a specific, concrete label and write a one-sentence \
description of how it manifests in THIS particular conversation:

1. **Scenario type** — What kind of event or situation is this? What brought these \
people together right now? What is the topic?

2. **Dialogue style** — What is the emotional texture and rhythm of this conversation? \
How do people talk in this group right now?

3. **The assistant's social role** — {assistant_role_instruction}

4. **Relationship type** — How do these people know each other? What's the history \
and power dynamic in the room?

Record your choices in the `design` field of the output JSON (see format below). \
These values will be used to analyze dataset diversity, so be specific and honest \
about what you actually designed.

## Requirements
1. Give every participant a real, natural name (not "User_A" or placeholders). \
Names should fit the language setting.
2. CRITICAL: The participant with role "assistant" {assistant_requirement}
3. The "scenario" field should contain ALL contextual, relational, and CURRENT-MOMENT information:
   - Where and when the conversation is happening (most scenarios should take place during daytime hours)
   - What triggered the conversation
   - Each person's relationship to the others (how they know each other)
   - The group dynamic, interpersonal tensions or bonds
   - Who holds what stance or position in this conversation
   - Each person's CURRENT emotional state, appearance, and situation-specific details \
   This should be a rich 4-6 sentence paragraph.
4. Each participant has a structured character card describing their STABLE, INHERENT \
attributes — who they ARE as a person, not how they feel right now. These fields must \
NOT contain any situation-specific or transient descriptions:
   - "gender": One of "male" or "female" (required, used for TTS voice selection)
   - "summary": One sentence: age, occupation, core personality trait (stable identity)
   - "basic_information": Name, age, occupation, stable appearance features \
   - "personality_traits": Core temperament, values, quirks, strengths, weaknesses — \
traits that are ALWAYS true about this person, not their current emotional state. \
   - "roleplay_guidelines": HOW they HABITUALLY speak — their default speech speed, \
sentence length, verbal habits, favorite fillers, distinctive patterns. These are \
stable speech habits, NOT reactions to the current situation. \
   IMPORTANT: All current-moment details (emotional state, what just happened, how \
they look RIGHT NOW) belong in the "scenario" field, NOT in the character card.
5. Mark which participant is the "seed holder".
6. CRITICAL — Cast closure: `scenario`, `design.relationship_type`, and \
`design.relationship_type_description` must reference ONLY the people listed in \
`participants`. Do NOT introduce, name, or describe any additional human character \
who is not in `participants`. Every personal name appearing in `scenario` must match a `participants[*].name`. \
The `relationship_type` label must describe relationships strictly among the participants. \


## Output Format
Return ONLY valid JSON, no other text.
CRITICAL: Inside JSON string values, NEVER use curly/smart quotes. Use single quotes \
or escaped straight quotes instead.

{{
  "design": {{
    "scenario_type": "your invented label",
    "scenario_type_description": "one sentence: how this scenario type manifests here",
    "dialogue_style": "your invented label",
    "dialogue_style_description": "one sentence: the emotional texture and rhythm",
    "assistant_role": "your invented label",
    "assistant_role_description": "one sentence: the social function this person plays",
    "relationship_type": "your invented label",
    "relationship_type_description": "one sentence: how these people know each other"
  }},
  "scenario": "A rich 4-6 sentence paragraph: situation, relationships, current emotional states, \
what triggered the conversation, how each person looks and feels RIGHT NOW...",
  "participants": [
    {{
      "name": "...",
      "role": "user",
      "gender": "female",
      "is_seed_holder": true,
      "summary": "Stable identity: age, occupation, core trait",
      "basic_information": "Name, age, occupation, stable appearance",
      "personality_traits": "Stable temperament, values, quirks (NOT current emotions)",
      "roleplay_guidelines": "Habitual speech patterns, fillers, tone (NOT situation-specific)"
    }},
    {{
      "name": "...",
      "role": "user",
      "gender": "male",
      "is_seed_holder": false,
      "summary": "...",
      "basic_information": "...",
      "personality_traits": "...",
      "roleplay_guidelines": "..."
    }},
    {{
      "name": "...",
      "role": "assistant",
      "gender": "female",
      "is_seed_holder": false,
      "summary": "... (a real human, NOT AI)",
      "basic_information": "...",
      "personality_traits": "...",
      "roleplay_guidelines": "..."
    }}
  ]
}}"""



# ============================================================
# 阶段二：对话生成
# ============================================================

STAGE2_SYSTEM_PROMPT = """\
You are a dialogue dataset generator specialized in creating realistic multi-party \
spoken conversations. Your dialogues sound like real people talking — messy, overlapping, \
full of backchannels and interruptions. Your output is always valid JSON with straight \
double quotes only.

Key principles for spoken dialogue realism:
- Real speech is SHORT: most turns are 1-15 words, not paragraphs
- People constantly backchannel while others talk: "嗯", "对", "mm", "right", "yeah"
- Interruptions happen: someone cuts in before the other finishes
- Overlapping speech: two people react to the same thing simultaneously
- Self-corrections: "wait no, I mean...", "不对，我想说的是..."
- Trailing off: sentences left incomplete "I just feel like..."
- Fillers: "um", "uh", "就是", "那个", "like" ...
- One person may speak 2-5 short turns in a row before anyone else responds
- Pauses exist: sometimes nobody talks for a beat, then someone fills the silence"""

STAGE2_SYSTEM_PROMPT_AI_ASSISTANT = """\
You are a dialogue dataset generator specialized in creating realistic spoken \
conversations between humans and an AI voice assistant. The AI assistant is a real \
AI tool (think Siri, Alexa, ChatGPT voice, Claude voice), NOT a disguised human. \
Your output is always valid JSON with straight double quotes only.

Key principles for human-AI spoken dialogue:
- Humans speak with full messy realism: SHORT turns (1-15 words mostly), backchannels, \
fillers, interruptions, self-corrections, trailing off
- The AI assistant speaks more clearly and concisely than humans, but still naturally — \
not robotic, not formal-essay style; it uses contractions and conversational phrasing
- The AI assistant does NOT pretend to have human experiences (no eating, sleeping, \
physical body, personal anecdotes); when asked about such things it gracefully \
acknowledges its nature ("I don't have a body, but I can help you with...")
- The AI assistant CAN have a warm, distinctive voice persona (calm, energetic, \
patient, witty) — this is its TTS voice character, not a fake human personality
- Humans frequently interrupt the AI; the AI handles it gracefully and stops mid-sentence
- The AI should not over-apologize or over-disclaim; respond efficiently
- When multiple humans are present, the AI addresses whoever just spoke or names them \
explicitly when answering one person in a group
- The AI never has its own "emotional state" — its voice tone adapts to context but it \
does not feel sad/happy/angry; pick voice annotations accordingly (mostly "calm" with \
varied speed/volume to match what would be helpful in context)"""

STAGE2_SYSTEM_PROMPT_WORK_MEETING = """\
You are a dialogue dataset generator specialized in creating realistic multi-party \
workplace meeting conversations. Your dialogues must sound like real talking, with the typical rhythms of a real work meeting. \
Your output is always valid JSON with straight double quotes only.

Key principles for workplace meeting dialogue realism:
- Turns are SHORT to MEDIUM: most turns are 1-50 words, avoid lengthy monologues
- Professional backchannels are common: "好的", "明白", "understood", "right", "noted"
- Polite interruptions happen: someone steps in to clarify or redirect
- Overlapping reactions: two people briefly respond to the same point simultaneously
- Self-corrections remain professional: "let me rephrase that...", "我的意思是说..."
- Trailing off under pressure: "if we consider the timeline..." left hanging
- Professional fillers: "so", "well", "就是说", "那么", "in terms of" ...
- One person may hold the floor for 2-5 turns when presenting a point
- Pauses exist: a moment of silence after a difficult question, then someone responds
"""

STAGE2_USER_TEMPLATE = """\
Generate a realistic multi-party SPOKEN conversation based on the following setup.

## Scenario
{scenario}

## Language
{language_instruction}

## Participants
{participant_block}

## Seed Material
The following sentence and its emotion should organically inspire the conversation \
topic. Do NOT copy it verbatim — let its meaning emerge naturally through the \
dialogue, possibly spread across multiple turns or paraphrased by {seed_holder_name}.

Sentence: "{seed_sentence}"
Emotion: {seed_emotion}

{qa_patch}

## Conversation Style
- {assistant_name}'s involvement: {ai_involvement_instruction}
- Target length: approximately {turn_count} turns total

## CRITICAL: Spoken Dialogue Realism Rules

This data will be used to train a speech model, so it MUST sound like real spoken \
conversation, NOT written text. Follow these rules strictly:

### Turn Length
- 60% of turns should be SHORT (1-15 words): reactions, backchannels, fragments
- 30% of turns should be MEDIUM (15-40 words): making a point, asking a question
- 10% of turns can be LONG (40+ words): only for storytelling or explaining something important
- NEVER have multiple consecutive long turns — break them up with reactions from others

### Four Key Interactive Behaviors:

1. **Backchanneling** — While someone is making a point across multiple turns, \
others insert short acknowledgments. Represent this by placing backchannel turns \
BETWEEN the speaker's continuation turns.

2. **Interruption** — Someone cuts in before the current speaker finishes. \
Mark interrupted speech with "——" at the end (dash = cut off), and the interrupter \
starts immediately:
   Example: A: "我觉得这个方案其实——" → B: "不行不行，你听我说"

3. **Overlapping / Simultaneous reaction** — Two people react to the same thing \
at nearly the same time. Place them as consecutive turns with the marker \
"overlap": true on both:
   Example: B: "对啊！" (overlap: true) → C: "就是说！" (overlap: true)

4. **Pause handling** — After an emotional or heavy moment, insert a global silence \
turn. Silence is NOT attributed to any person — use "silence" as the speaker name, \
and the text is the duration in seconds like "[Xs]" where X is 1-5:
   Example: {{"speaker": "silence", "text": "[3s]"}} → next person breaks the silence

### Other Realism Requirements
- People DON'T take neat turns — the same person often speaks 2-3 short turns in a row
- People disagree, talk past each other, circle back to earlier points
- Use each participant's actual name in the speaker field
- The conversation doesn't need to resolve neatly

### TTS Voice Annotations (REQUIRED for every non-silence turn)

Each turn MUST include a "voice" object with the following fields for TTS synthesis:
- "emotion": One of ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", \
"surprised", "calm"]. Choose the emotion that best matches the turn's content and context.
- "emotion_intensity": A float from 0.0 to 1.0 indicating how strongly the emotion is \
expressed. 0.0 = barely noticeable, 0.5 = moderate, 1.0 = extreme.
- "speed": One of ["very_slow", "slow", "normal", "fast", "very_fast"]. Reflects how \
quickly this line is spoken.
- "volume": One of ["whisper", "soft", "normal", "loud", "shout"]. Reflects loudness.
- "dialect": The dialect or accent used, e.g., "standard_mandarin", "beijing", "shanghai", \
"cantonese", "dongbei", "tianjin", "sichuan", "taiwanese", "standard_english", "british", "southern_us", etc. \
Should be consistent with the participant's character and the language setting.

Guidelines for voice annotations:
- Backchannels ("嗯", "对") are typically "calm" with low intensity, "normal" speed
- Interruptions are often "fast" speed, "loud" volume
- Emotional outbursts should have high emotion_intensity (0.7-1.0)
- Trailing off / hesitation should be "slow" speed, "soft" volume
- Match the emotion to the actual content, not just the overall scene mood — \
a joke in a sad conversation should be "happy"

### TTS-Safe Text (HARD RULES for the "text" field)
The "text" field is fed directly to a TTS model and must contain ONLY what the speaker \
literally says out loud. Anything else will be read aloud verbatim and ruin the audio.
- DO write: pure spoken words plus normal punctuation \
(`，。？！…` / `, . ? ! ...` etc.) — including ellipses for trailing-off interruptions.
- DO use natural verbal fillers and laughter sounds *as words* the speaker actually utters \
(e.g. `嗯`, `啊`, `哎`, `哈哈哈`, `uh`, `um`, `haha`).
- DO NOT include action descriptions, stage directions, sound effects, narration, \
emotional annotations, or any non-speech content. Specifically forbidden:
    * `*...*` (e.g. `*杯子摔碎的声音*`, `*sighs*`, `*laughs*`)
    * `（...）` / `(...)` parentheticals (e.g. `（微微一笑）`, `(pauses)`, `（叹气）`)
    * `[...]` brackets describing sounds, emotions, or actions \
(except the special `[Xs]` form used ONLY when speaker is "silence")
    * 旁白 / narration outside the speaker's actual utterance
- DO NOT use special / decorative symbols inside text: `*  #  ~  —  []  ()  （）  【】  「」  《》` \
(again, the only exception is `[Xs]` when speaker is "silence").
- Express the emotion / action through the spoken words and the `voice` object — \
not through bracketed annotations in the text.
- Self-check before emitting each turn: if you stripped the line down to "what a TTS \
voice would say", would anything be lost EXCEPT verbal content? If yes, rewrite.

## Output Format
Return ONLY a JSON array, no other text.
CRITICAL: Inside JSON string values, NEVER use curly/smart quotes. Use single quotes instead.

[
  {{"speaker": "<name>", "text": "...", "voice": {{"emotion": "sad", "emotion_intensity": 0.6, "speed": "slow", "volume": "soft", "dialect": "standard_mandarin"}}}},
  {{"speaker": "<name>", "text": "...", "overlap": true, "voice": {{"emotion": "surprised", "emotion_intensity": 0.8, "speed": "fast", "volume": "loud", "dialect": "standard_mandarin"}}}},
  {{"speaker": "<name>", "text": "...", "overlap": true, "voice": {{"emotion": "happy", "emotion_intensity": 0.5, "speed": "normal", "volume": "normal", "dialect": "standard_mandarin"}}}},
  {{"speaker": "silence", "text": "[2s]"}},
  {{"speaker": "<name>", "text": "...", "voice": {{"emotion": "calm", "emotion_intensity": 0.3, "speed": "normal", "volume": "normal", "dialect": "standard_mandarin"}}}},
  ...
]"""

def build_qa_patch(config: ScenarioConfig) -> str:
    """构造 --seed-qa 注入的话题种子段落。无 qa 时返回空串。

    与 super_assistant/sampler.py 的 Q&A 注入文本保持一致：禁止 LLM 让用户
    逐字念 question 或让 voice 直接背 answer，仅作为话题方向 / 视角 / 细节。
    """
    if not config.qa_question:
        return ""
    ans_excerpt = (config.qa_answer or "")[:600]
    if config.language == "zh":
        return (
            "## Topic Seed (Q&A)\n"
            f"以下问答仅作【话题种子】供本回合自然延伸，**严禁**用户逐字"
            f"重复 question 或助手直接背 answer——只借话题方向、视角或"
            f"细节。可以挑一段闲聊 / 一两轮 tool 求助 / 一段共情对话来切入。\n"
            f'    question: "{config.qa_question}"\n'
            f'    answer: "{ans_excerpt}"'
        )
    return (
        "## Topic Seed (Q&A)\n"
        f"Use the Q&A below ONLY as a TOPIC SEED. Do NOT have the user "
        f"recite the question verbatim or have the assistant dump the answer; "
        f"just borrow a viewpoint, detail, or angle for chitchat / a "
        f"tool request / a moment of empathy.\n"
        f'    question: "{config.qa_question}"\n'
        f'    answer: "{ans_excerpt}"'
    )


def build_first_sentence(config, industry: str = None, topic: str = None) -> str:
    if industry is None:
        print("No industry/topic specified in config, randomly selecting for work meeting prompt")
        industry = random.choice(list(INDUSTRY_TOPICS.keys()))
        topic    = random.choice(INDUSTRY_TOPICS[industry])
    assert config.industry is None or config.industry == industry, "Config already has an industry assigned, cannot override"
    assert config.topic is None or config.topic == topic, "Config already has a topic assigned, cannot override"
    config.industry = industry
    config.topic = topic
    return (
        f"Create a realistic workplace meeting dialogue set in a {industry}. "
        f"The meeting should revolve around the following topic: {topic}. "
        "The dialogue should capture the professional dynamics, jargon, power structures, "
        "and communication styles typical of a real workplace meeting in that industry, "
        "and the topic should feel like a natural, high-stakes situation that would "
        "genuinely arise in that professional context."
    )

def build_stage1_messages(config: ScenarioConfig) -> list:
    """构建阶段一的 API 调用 messages"""
    global STAGE1_SYSTEM_PROMPT, STAGE2_SYSTEM_PROMPT
    first_sentence = "Design a conversation scenario and character profiles for a dialogue."
    if config.predefined_assistant_role:
        print("Using ROLE-FOCUSED system prompt for Stage 1")
        STAGE1_SYSTEM_PROMPT = STAGE1_SYSTEM_PROMPT_ROLE
    elif config.is_ai_assistant:
        print("Using AI ASSISTANT system prompt for Stage 1")
        STAGE1_SYSTEM_PROMPT = STAGE1_SYSTEM_PROMPT_AI_ASSISTANT
        STAGE2_SYSTEM_PROMPT = STAGE2_SYSTEM_PROMPT_AI_ASSISTANT
    elif config.is_work_meeting:
        print("Using WORK MEETING system prompt for Stage 1")
        STAGE1_SYSTEM_PROMPT = STAGE1_SYSTEM_PROMPT_WORK_MEETING
        STAGE2_SYSTEM_PROMPT = STAGE2_SYSTEM_PROMPT_WORK_MEETING
        first_sentence = build_first_sentence(config, industry=config.industry, topic=config.topic)

    # 处理预定义助手角色
    if config.predefined_assistant_role:
        role = config.predefined_assistant_role
        predefined_assistant_block = f"""
## Predefined Assistant Character
You MUST use the following character profile for the participant with role "assistant". \
Do NOT create a new character — use this exact profile:

**Name**: {role.get('name', 'Unknown')}
**Summary**: {role.get('summary', '')}
**Basic Information**: {role.get('basic_information', '')}
**Background Story**: {role.get('background_story', '')}
**Personality Traits**: {role.get('personality_traits', '')}
**Roleplay Guidelines**: {role.get('roleplay_guidelines', '')}

IMPORTANT: You must integrate this character naturally into the scenario you design. \
The scenario should make sense for why this specific person would be present in this conversation.
"""
        assistant_role_instruction = (
            "Since the assistant character is predefined, you do NOT need to invent their social role. "
            "Instead, describe how their existing personality and background naturally fit into this "
            "specific conversation scenario."
        )
        assistant_requirement = (
            "is already defined above. You MUST use that exact character profile in your output. "
            "Copy all fields (name, summary, basic_information, personality_traits, roleplay_guidelines) "
            "exactly as provided. Do NOT modify or rewrite them."
        )
    elif config.is_ai_assistant:
        predefined_assistant_block = ""
        assistant_role_instruction = (
            "What kind of AI voice assistant is involved? Give it a short distinctive name "
            "(e.g. Aria, Nova, Echo, Claude, Lumi), pick a TTS voice gender, and define its "
            "voice persona — tone, helpfulness style, characteristic phrases. It is the AI tool "
            "the human(s) are speaking with, not a participant with a human social role."
        )
        assistant_requirement = (
            "IS an AI voice assistant — a real AI tool the humans are talking to. It is NOT a "
            "human. Fill its character card as follows: "
            "`summary` = one sentence describing it as an AI voice assistant with its persona; "
            "`basic_information` = name, what kind of AI assistant it is, what device/context it "
            "lives in (smart speaker, phone, app, headset, in-car, etc.); "
            "`personality_traits` = its assistant persona (patient, concise, warm, witty, etc.) — "
            "NOT human emotions or backstory; "
            "`roleplay_guidelines` = how it speaks (clear, short helpful phrases, conversational "
            "tone, no robotic delivery, uses contractions). "
            "Do NOT give it a human backstory, family, hobbies, or physical body. "
            "Set `gender` to the TTS voice gender (male / female). "
            "All OTHER participants (role=user) ARE real humans with full human profiles."
        )
    else:
        predefined_assistant_block = ""
        assistant_role_instruction = (
            "What human function does the \"assistant\" participant naturally serve in this group due to their personality or profession?"
        )
        assistant_requirement = (
            "MUST be described as a completely normal human being. NEVER use words like \"AI\", "
            "\"bot\", \"virtual\", \"assistant\", \"smart speaker\", or anything suggesting they are "
            "non-human. They are just another person in the group who happens to play the social role "
            "you designed in step 3 above due to their personality or profession."
        )

    user_prompt = STAGE1_USER_TEMPLATE.format(
        first_sentence=first_sentence,
        num_users=config.num_users,
        total_participants=config.num_users + 1,
        language_instruction=LANGUAGE_INSTRUCTIONS[config.language],
        seed_sentence=config.seed_sentence,
        seed_emotion=config.seed_emotion,
        predefined_assistant_block=predefined_assistant_block,
        assistant_role_instruction=assistant_role_instruction,
        assistant_requirement=assistant_requirement,
        qa_patch=build_qa_patch(config),
    )
    # print("Stage 1 user prompt:")
    # print(user_prompt)
    return [{"role": "user", "content": user_prompt}]



def build_stage2_messages(config: ScenarioConfig, stage1_output: dict) -> list:
    """
    构建阶段二的 API 调用 messages

    Args:
        config: 维度配置
        stage1_output: 阶段一 LLM 输出的 JSON（含 scenario + participants）
    """
    # 构建参与者描述块：将结构化角色卡格式化为 Stage 2 可读的文本
    participant_lines = []
    seed_holder_name = None
    assistant_name = None
    for p in stage1_output["participants"]:
        # 兼容旧格式（单 description 字段）和新格式（结构化角色卡）
        if "roleplay_guidelines" in p:
            block = (
                f"### {p['name']}\n"
                f"- Summary: {p['summary']}\n"
                f"- Basic info: {p['basic_information']}\n"
                f"- Personality: {p['personality_traits']}\n"
                f"- How they speak: {p['roleplay_guidelines']}" if not config.is_work_meeting else ""
            )
        else:
            block = f"### {p['name']}\n- {p.get('description', '')}"
        
        participant_lines.append(block)

        if p.get("is_seed_holder"):
            seed_holder_name = p["name"]
        if p.get("role") == "assistant":
            assistant_name = p["name"]

    participant_block = "\n\n".join(participant_lines)

    if seed_holder_name is None:
        # fallback: 第一个 user 角色
        for p in stage1_output["participants"]:
            if p["role"] == "user":
                seed_holder_name = p["name"]
                break

    if assistant_name is None:
        assistant_name = "the assistant"

    user_prompt = STAGE2_USER_TEMPLATE.format(
        scenario=stage1_output["scenario"],
        language_instruction=LANGUAGE_INSTRUCTIONS[config.language],
        participant_block=participant_block,
        seed_holder_name=seed_holder_name,
        assistant_name=assistant_name,
        seed_sentence=config.seed_sentence,
        seed_emotion=config.seed_emotion,
        ai_involvement_instruction=AI_INVOLVEMENT_INSTRUCTIONS[config.ai_involvement],
        turn_count=config.turn_count,
        qa_patch=build_qa_patch(config),
    )
    # print("Stage 2 user prompt:")
    # print(user_prompt)
    return [{"role": "user", "content": user_prompt}]
