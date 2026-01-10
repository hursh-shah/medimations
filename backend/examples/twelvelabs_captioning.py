"""
Template: caption + narration generation with TwelveLabs.

Usage:
  export TWELVELABS_API_KEY=...
  python3 backend/examples/twelvelabs_captioning.py path/to/video.mp4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_diffusion.postprocess.twelvelabs import (
    DEFAULT_TWELVELABS_CAPTION_PROMPT,
    TwelveLabsConfig,
    generate_captions_with_twelvelabs,
)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: twelvelabs_captioning.py path/to/video.mp4")
        return 2

    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise RuntimeError("Set TWELVELABS_API_KEY in your environment")

    video_path = Path(argv[1]).expanduser().resolve()
    res = generate_captions_with_twelvelabs(
        video_path=video_path,
        config=TwelveLabsConfig(api_key=api_key),
        prompt=DEFAULT_TWELVELABS_CAPTION_PROMPT,
    )

    print("SUMMARY:")
    print(res.summary)
    print("\nNARRATION:")
    print(res.narration)
    if res.segments:
        print("\nSRT:")
        print(res.to_srt())
    if res.medical_uncertainties:
        print("\nMEDICAL_UNCERTAINTIES:")
        for item in res.medical_uncertainties:
            print("-", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
