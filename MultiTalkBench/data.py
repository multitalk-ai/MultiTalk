"""HuggingFace dataset reader for MultiTalkBench.

The dataset (https://huggingface.co/datasets/MultiTalk/MultiTalkBench) ships a
``metadata.jsonl`` with one row per ``(meeting, target_speaker)`` evaluation
sample. This module turns those rows into ``Sample`` objects with absolute
paths and provides cross-sample lookups (peer speakers in the same meeting).

The eval pipeline never touches the legacy ``test_dict.json`` /
``processed/<lang>/<split>/<meeting>/`` layout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from prompt_builder import Language


@dataclass(frozen=True)
class Sample:
    sample_id: str
    language: Language
    meeting_id: str
    target_speaker: str
    audio_path: Path
    alignment_path: Path
    persona_path: Path
    meeting_profile_path: Path
    transcript_context_path: Path
    duration: float
    minimal_system_prompt: str

    def system_prompt(self, *, detailed: bool) -> str:
        if detailed:
            return self.persona_path.read_text(encoding="utf-8").strip()
        return self.minimal_system_prompt


def _row_to_sample(row: dict, root: Path) -> Sample:
    return Sample(
        sample_id=row["sample_id"],
        language=Language(row["language"]),
        meeting_id=row["meeting_id"],
        target_speaker=row["target_speaker"],
        audio_path=root / row["audio"],
        alignment_path=root / row["alignment"],
        persona_path=root / row["persona"],
        meeting_profile_path=root / row["meeting_profile"],
        transcript_context_path=root / row["transcript_context"],
        duration=float(row.get("duration") or 0.0),
        minimal_system_prompt=row.get("minimal_system_prompt", ""),
    )


def load_samples(data_root: Path) -> List[Sample]:
    """Read every row of metadata.jsonl into a Sample list (canonical 104 samples)."""
    metadata_path = data_root / "metadata.jsonl"
    samples: List[Sample] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(_row_to_sample(json.loads(line), data_root))
    return samples


def load_samples_dict(data_root: Path) -> Dict[str, Sample]:
    """Map sample_id → Sample. ``sample_id`` is unique."""
    return {s.sample_id: s for s in load_samples(data_root)}


def group_by_meeting(samples: List[Sample]) -> Dict[tuple, List[Sample]]:
    """Group samples by (language, meeting_id)."""
    out: Dict[tuple, List[Sample]] = {}
    for s in samples:
        out.setdefault((s.language, s.meeting_id), []).append(s)
    return out


def find_meeting_speakers(
    samples: List[Sample],
    language: Language,
    meeting_id: str,
) -> List[Sample]:
    """All samples that share the same (language, meeting_id) — i.e., every
    real speaker in the meeting (the target speaker is one of them).
    """
    return [s for s in samples if s.language == language and s.meeting_id == meeting_id]


def filter_meetings(samples: List[Sample], meetings: Optional[str]) -> List[Sample]:
    """Restrict to a comma-separated list of ``meeting_id`` values (no-op if empty)."""
    if not meetings:
        return samples
    wanted = {m.strip() for m in meetings.split(",") if m.strip()}
    return [s for s in samples if s.meeting_id in wanted]
