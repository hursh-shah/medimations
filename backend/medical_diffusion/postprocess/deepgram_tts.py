from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DeepgramTTSConfig:
    api_key: str
    model: str = "aura-2-odysseus-en"
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

    try:
        from deepgram.core.api_error import ApiError  # type: ignore
    except Exception:  # pragma: no cover
        ApiError = Exception  # type: ignore

    client = DeepgramClient(api_key=config.api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = client.speak.v1.audio.generate(text=text, model=config.model, encoding=config.encoding)
    except TypeError:
        # Older/newer SDK variants may not expose model/encoding kwargs.
        response = client.speak.v1.audio.generate(text=text)
    except ApiError as e:
        raise RuntimeError(f"Deepgram TTS request failed ({getattr(e, 'status_code', '?')}): {getattr(e, 'body', e)}") from e

    # deepgram-sdk v5 returns an Iterator[bytes]; older variants may return an object with .stream.
    data: bytes = b""
    stream = getattr(response, "stream", None)
    if stream is not None:
        try:
            data = bytes(stream.getvalue())
        except Exception:
            data = b""
    elif isinstance(response, (bytes, bytearray, memoryview)):
        data = bytes(response)
    else:
        try:
            chunks = []
            for chunk in response:
                if isinstance(chunk, (bytes, bytearray, memoryview)):
                    chunks.append(bytes(chunk))
            data = b"".join(chunks)
        except TypeError:
            data = b""

    if not data:
        raise RuntimeError("Deepgram TTS returned empty audio")

    output_path.write_bytes(data)
    return output_path
