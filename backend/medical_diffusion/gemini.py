from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


class GeminiError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str = "gemini-3.0-flash"
    temperature: float = 0.2
    max_output_tokens: int = 1024


def load_gemini_config(*, model: str = "gemini-3.0-flash") -> GeminiConfig:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise GeminiError("GOOGLE_API_KEY is not set (required for Gemini prompt rewriting)")
    return GeminiConfig(api_key=api_key, model=model)


def generate_text(*, system: str, user: str, config: GeminiConfig) -> str:
    """
    Thin wrapper around google-genai that returns response.text.
    """
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise GeminiError("Missing dependency: pip install google-genai") from e

    client = genai.Client(api_key=config.api_key)
    res = client.models.generate_content(
        model=config.model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=float(config.temperature),
            max_output_tokens=int(config.max_output_tokens),
        ),
    )
    text = (res.text or "").strip()
    if not text:
        raise GeminiError("Gemini returned an empty response")
    return text


def generate_json(*, system: str, user: str, config: GeminiConfig) -> Dict[str, Any]:
    """
    Requests JSON output and parses it; raises GeminiError on failure.
    """
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise GeminiError("Missing dependency: pip install google-genai") from e

    client = genai.Client(api_key=config.api_key)
    res = client.models.generate_content(
        model=config.model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=float(config.temperature),
            max_output_tokens=int(config.max_output_tokens),
            response_mime_type="application/json",
        ),
    )
    raw = (res.text or "").strip()
    if not raw:
        raise GeminiError("Gemini returned an empty response")

    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        raise GeminiError("Gemini JSON response was not an object")
    return data


def _extract_json_object(text: str) -> Any:
    # Strip common markdown fences.
    if text.startswith("```"):
        text = text.strip().lstrip("`")
        # Drop a leading language tag line if present.
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"json", "javascript"}:
            text = "\n".join(lines[1:])
        text = text.strip()
        if text.endswith("```"):
            text = text[: -len("```")].strip()

    # Best-effort: parse the largest {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise GeminiError("Could not find a JSON object in Gemini response")

    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception as e:
        raise GeminiError(f"Failed to parse Gemini JSON: {e}") from e


def get_optional_str(d: Dict[str, Any], key: str) -> Optional[str]:
    v = d.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None
