"""
Template: text-to-speech narration with Deepgram.

Usage:
  export DEEPGRAM_API_KEY=...
  python3 backend/examples/deepgram_tts.py "Hello world" runs/narration.mp3
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_diffusion.postprocess.deepgram_tts import DeepgramTTSConfig, synthesize_narration_with_deepgram


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print('Usage: deepgram_tts.py "text to speak" out.mp3')
        return 2

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPGRAM_API_KEY in your environment")

    text = argv[1]
    out = Path(argv[2]).expanduser()
    synthesize_narration_with_deepgram(
        text=text,
        output_path=out,
        config=DeepgramTTSConfig(api_key=api_key),
    )
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
