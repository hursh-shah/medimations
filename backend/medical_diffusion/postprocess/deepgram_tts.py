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
        from deepgram import DeepgramClient, SpeakOptions
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install deepgram-sdk") from e

    client = DeepgramClient(config.api_key)

    options = SpeakOptions(model=config.model, encoding=config.encoding)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = client.speak.v("1").save(output_path, {"text": text}, options)
    if response is None:
        raise RuntimeError("Deepgram TTS returned no response")
    return output_path

