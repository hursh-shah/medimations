from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DeepgramTTSConfig:
    api_key: str
    model: str = "aura-asteria-en"
    encoding: str = "mp3"


def synthesize_narration_with_deepgram(
    *,
    text: str,
    output_path: Path,
    config: DeepgramTTSConfig,
) -> Path:
    """
    Template Deepgram TTS hook (add later).

    Requires:
      pip install deepgram-sdk

    This intentionally keeps the interface stable so you can wire it into a
    web app later without changing call sites.
    """
    try:
        from deepgram import DeepgramClient
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install deepgram-sdk") from e

    client = DeepgramClient(api_key=config.api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = client.speak.v1.audio.generate(text=text, model=config.model, encoding=config.encoding)
    except TypeError:
        # Older/newer SDK variants may not expose model/encoding kwargs.
        response = client.speak.v1.audio.generate(text=text)
    audio = getattr(response, "stream", None)
    if audio is None:
        raise RuntimeError("Deepgram TTS returned no audio stream")

    data = audio.getvalue()
    if not data:
        raise RuntimeError("Deepgram TTS returned empty audio")

    output_path.write_bytes(data)
    return output_path
