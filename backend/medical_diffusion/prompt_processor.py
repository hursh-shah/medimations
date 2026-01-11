from __future__ import annotations

import re
from dataclasses import replace

from .types import AnimationSpec
from .veo_guidelines import get_default_negative_prompt


class PromptProcessor:
    """
    Minimal, rules-based prompt parsing.
    """

    _DURATION_RE = re.compile(r"(?P<seconds>\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b", re.I)
    _FPS_RE = re.compile(r"(?P<fps>\d+)\s*(?:fps)\b", re.I)

    def parse(self, raw_input: str) -> AnimationSpec:
        raw_input = raw_input.strip()
        spec = AnimationSpec(prompt=raw_input)

        duration_match = self._DURATION_RE.search(raw_input)
        if duration_match:
            spec = replace(spec, duration_s=float(duration_match.group("seconds")))

        fps_match = self._FPS_RE.search(raw_input)
        if fps_match:
            spec = replace(spec, fps=int(fps_match.group("fps")))

        # Remove control tokens from the actual prompt text.
        cleaned = self._DURATION_RE.sub("", raw_input)
        cleaned = self._FPS_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            spec = replace(spec, prompt=cleaned)

        return spec

    def rewrite_for_veo(self, spec: AnimationSpec) -> AnimationSpec:
        """
        Turn a short user request into a higher-signal Veo prompt.

        This is intentionally rule-based (no LLM required) so the repo stays
        runnable offline. Replace this with an LLM-to-JSON prompt agent later.
        """
        base = spec.prompt.strip()
        if not base:
            return spec

        metadata = dict(spec.metadata or {})
        metadata.setdefault("user_prompt", base)

        # Keep it readable and explicit. Veo tends to respond well to:
        # - subject + action
        # - style + constraints
        # - explicit "no text / no watermark" constraints
        prompt = (
            f"Create a short {spec.duration_s:g}s 3D medical animation. "
            f"Scene: {base}. "
            "Requirements: medically accurate, anatomically correct, temporally consistent motion. "
            "Style: clinical textbook CGI, clean lighting, neutral background, high detail. "
            "Constraints: no on-screen text, no labels, no watermark."
        )
        if spec.input_image_path is not None:
            prompt = (
                prompt
                + " Use the provided reference medical image as the first frame; preserve its anatomy, viewpoint, and style, and only animate realistic motion consistent with the reference."
            )

        negative = spec.negative_prompt
        if not negative:
            negative = get_default_negative_prompt()

        return replace(spec, prompt=prompt, negative_prompt=negative, metadata=metadata)

    def rewrite_for_veo_gemini(self, spec: AnimationSpec, *, model: str = "gemini-3.0-flash") -> AnimationSpec:
        """
        Rewrite using Gemini (requires GOOGLE_API_KEY + google-genai).

        Falls back to rewrite_for_veo() if Gemini is unavailable.
        """
        base = spec.prompt.strip()
        if not base:
            return spec

        metadata = dict(spec.metadata or {})
        metadata.setdefault("user_prompt", base)

        try:
            from .prompt_rewriter import rewrite_user_prompt_for_veo

            rewritten = rewrite_user_prompt_for_veo(
                user_prompt=base,
                duration_s=spec.duration_s,
                fps=spec.fps,
                reference_image_provided=spec.input_image_path is not None,
                model=model,
            )
        except Exception as e:
            metadata["prompt_rewrite_mode"] = "rule_fallback"
            metadata["prompt_rewrite_error"] = str(e)
            return self.rewrite_for_veo(replace(spec, metadata=metadata))

        metadata["prompt_rewrite_mode"] = "gemini"
        metadata["prompt_rewrite_model"] = model
        if rewritten.target_label:
            metadata["target_label"] = rewritten.target_label
        if rewritten.notes:
            metadata["prompt_rewrite_notes"] = rewritten.notes

        negative = spec.negative_prompt or rewritten.negative_prompt
        if not negative:
            negative = get_default_negative_prompt()

        return replace(spec, prompt=rewritten.veo_prompt, negative_prompt=negative, metadata=metadata)
