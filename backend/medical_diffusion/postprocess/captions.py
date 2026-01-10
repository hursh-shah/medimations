from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


@dataclass(frozen=True)
class CaptionSegment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class CaptionResult:
    summary: str
    segments: List[CaptionSegment]
    narration: str
    medical_uncertainties: List[str]

    def to_srt(self) -> str:
        lines: List[str] = []
        for i, seg in enumerate(self.segments, start=1):
            lines.append(str(i))
            lines.append(f"{_fmt_srt_time(seg.start_s)} --> {_fmt_srt_time(seg.end_s)}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def parse_caption_result(raw_text: str) -> Optional[CaptionResult]:
    """
    Best-effort parse for TwelveLabs analyze() output when you ask for JSON.
    Returns None if JSON parsing fails.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return None

    try:
        data = json.loads(raw_text)
    except Exception:
        return None

    try:
        summary = str(data.get("summary", "")).strip()
        narration = str(data.get("narration", "")).strip()
        uncertainties = data.get("medical_uncertainties") or []
        medical_uncertainties = [str(x).strip() for x in uncertainties if str(x).strip()]
        segments = []
        for item in data.get("captions") or data.get("segments") or []:
            start_s = float(item.get("start_s"))
            end_s = float(item.get("end_s"))
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            segments.append(CaptionSegment(start_s=start_s, end_s=end_s, text=text))
        if not segments:
            return None
        return CaptionResult(
            summary=summary,
            segments=segments,
            narration=narration or " ".join([s.text for s in segments]).strip(),
            medical_uncertainties=medical_uncertainties,
        )
    except Exception:
        return None


def _fmt_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000.0))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def normalize_segments(segments: Iterable[CaptionSegment]) -> List[CaptionSegment]:
    out: List[CaptionSegment] = []
    for seg in segments:
        start = float(seg.start_s)
        end = float(seg.end_s)
        if end < start:
            start, end = end, start
        out.append(CaptionSegment(start_s=max(0.0, start), end_s=max(0.0, end), text=str(seg.text).strip()))
    out.sort(key=lambda s: (s.start_s, s.end_s))
    return out

