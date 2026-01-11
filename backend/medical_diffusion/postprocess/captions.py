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


def _extract_json_from_text(raw_text: str) -> Optional[dict]:
    """
    Extract a JSON object from text that may contain markdown fences or other content.
    """
    text = raw_text.strip()
    if not text:
        return None
    
    # Try direct parsing first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        # Find the end of the opening fence line
        first_newline = text.find("\n")
        if first_newline > 0:
            text = text[first_newline + 1:]
        else:
            text = text[3:]  # Just strip ```
        
        # Strip trailing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
        
        # Try parsing again
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    
    # Try to find a JSON object within the text (between first { and last })
    start_idx = raw_text.find("{")
    end_idx = raw_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = raw_text[start_idx:end_idx + 1]
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    
    return None


def parse_caption_result(raw_text: str) -> Optional[CaptionResult]:
    """
    Best-effort parse for TwelveLabs analyze() output when you ask for JSON.
    
    Handles:
    - Direct JSON
    - Markdown-wrapped JSON (```json ... ```)
    - JSON embedded in larger text response
    
    Returns None if JSON parsing fails.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return None

    data = _extract_json_from_text(raw_text)
    if data is None:
        return None

    try:
        summary = str(data.get("summary", "")).strip()
        narration = str(data.get("narration", "")).strip()
        uncertainties = data.get("medical_uncertainties") or []
        medical_uncertainties = [str(x).strip() for x in uncertainties if str(x).strip()]
        segments = []
        
        # Try multiple keys for captions
        caption_data = data.get("captions") or data.get("segments") or data.get("caption") or []
        if not isinstance(caption_data, list):
            caption_data = []
            
        for item in caption_data:
            if not isinstance(item, dict):
                continue
            # Handle multiple time key formats
            start_s = item.get("start_s") or item.get("start") or item.get("startTime") or 0.0
            end_s = item.get("end_s") or item.get("end") or item.get("endTime") or 0.0
            try:
                start_s = float(start_s)
                end_s = float(end_s)
            except (TypeError, ValueError):
                continue
            text = str(item.get("text", "") or item.get("caption", "")).strip()
            if not text:
                continue
            segments.append(CaptionSegment(start_s=start_s, end_s=end_s, text=text))
        
        # If we have a narration but no segments, that's still useful
        if not segments and not narration:
            return None
        
        # If no explicit narration, build from segments
        final_narration = narration or " ".join([s.text for s in segments]).strip()
        
        return CaptionResult(
            summary=summary,
            segments=segments,
            narration=final_narration,
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

