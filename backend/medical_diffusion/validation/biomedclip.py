from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import GenerationResult, ValidationScore


@dataclass(frozen=True)
class BiomedCLIPThresholds:
    min_target_top1_ratio: float = 0.55
    min_target_topk_ratio: float = 0.80
    min_mean_margin: float = 0.015
    min_mean_confidence: float = 0.20
    top_k: int = 3


def infer_target_label(prompt: str, *, candidate_labels: Sequence[str]) -> Optional[str]:
    prompt_l = prompt.lower()
    # Prefer longer labels so "small intestine" matches before "intestine", etc.
    for label in sorted(candidate_labels, key=lambda s: (-len(s), s)):
        l = label.lower()
        if l and l in prompt_l:
            return label
    return None


def _uniform_sample(items: Sequence[Path], n: int) -> List[Path]:
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    out: List[Path] = []
    for i in range(n):
        t = i / float(n - 1)
        idx = int(round(t * (len(items) - 1)))
        out.append(items[idx])
    return out


def _linux_mem_total_mb() -> Optional[int]:
    """
    Best-effort total RAM check (helps avoid OOM-killing small Railway instances).
    """
    try:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return None
        for line in meminfo.read_text().splitlines():
            if not line.startswith("MemTotal:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                return None
            kb = int(parts[1])
            return max(0, kb // 1024)
    except Exception:
        return None


class BiomedCLIPMedicalValidator:
    """
    Frame-level, zero-shot scoring using BiomedCLIP (OpenCLIP).

    This is a "guardrail" signal: it can help detect if the generated frames
    drift away from the intended anatomy/modality, but it is not a medical
    correctness oracle.
    """

    name = "biomedclip"

    def __init__(
        self,
        *,
        target_label: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        hf_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        prompt_template: str = "a biomedical image of {label}",
        n_frames: int = 12,
        context_length: int = 256,
        device: Optional[str] = None,
        thresholds: Optional[BiomedCLIPThresholds] = None,
    ) -> None:
        self._hf_id = hf_id
        self._prompt_template = prompt_template
        self._n_frames = int(n_frames)
        self._context_length = int(context_length)
        self._device_override = device
        self._thresholds = thresholds or BiomedCLIPThresholds()

        self._labels = list(labels or _default_labels())
        self._target_label = target_label

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None
        self._F = None
        self._pil_image = None
        self._text_feature_cache: Dict[Tuple[str, ...], Any] = {}

    def score(self, generation: GenerationResult) -> ValidationScore:
        if not generation.frames:
            return ValidationScore(name=self.name, score=0.0, feedback="No frames produced")

        try:
            self._ensure_loaded()
        except Exception as e:
            return ValidationScore(
                name=self.name,
                score=1.0,
                skipped=True,
                feedback=str(e),
                details={"skipped_reason": "biomedclip_unavailable"},
            )

        labels = list(self._labels)
        target = self._target_label or infer_target_label(generation.spec.prompt, candidate_labels=labels)
        if not target:
            return ValidationScore(
                name=self.name,
                score=1.0,
                skipped=True,
                feedback="No target label provided and none could be inferred from prompt",
                details={"skipped_reason": "missing_target_label", "labels": labels},
            )
        if target not in labels:
            labels = [target] + [l for l in labels if l != target]

        frame_paths = _uniform_sample(generation.frames, self._n_frames)
        pil_images: List[Any] = []
        sampled_paths: List[str] = []
        for p in frame_paths:
            try:
                with self._pil_image.open(p) as im:
                    pil_images.append(im.convert("RGB"))
                sampled_paths.append(str(p))
            except Exception:
                continue
        if not pil_images:
            return ValidationScore(name=self.name, score=0.0, feedback="Could not load any sampled frames")

        organ_prompts = [self._prompt_template.format(label=l) for l in labels]

        image_feats = self._encode_images(pil_images)
        text_feats = self._encode_texts(organ_prompts)
        logits = self._similarity_logits(image_feats, text_feats)

        probs = self._F.softmax(logits, dim=-1).detach().cpu()
        logits_cpu = logits.detach().cpu()

        target_idx = labels.index(target)
        top_k = max(1, int(self._thresholds.top_k))

        frame_predictions: List[Dict[str, Any]] = []
        label_hist: Dict[str, int] = {l: 0 for l in labels}
        top1_hits = 0
        topk_hits = 0
        margins: List[float] = []
        target_confs: List[float] = []

        for i in range(probs.shape[0]):
            row_p = probs[i]
            row_l = logits_cpu[i]
            sorted_idx = row_p.argsort(descending=True)
            top1 = int(sorted_idx[0].item())
            label_hist[labels[top1]] = int(label_hist.get(labels[top1], 0)) + 1

            rank = (
                int((sorted_idx == target_idx).nonzero(as_tuple=False)[0].item())
                if (sorted_idx == target_idx).any().item()
                else len(labels)
            )
            if top1 == target_idx:
                top1_hits += 1
            if rank < top_k:
                topk_hits += 1

            # Margin: target vs best non-target.
            best_non_target = float("-inf")
            for j in range(len(labels)):
                if j == target_idx:
                    continue
                best_non_target = max(best_non_target, float(row_l[j].item()))
            margin = float(row_l[target_idx].item()) - float(best_non_target)
            margins.append(margin)

            target_conf = float(row_p[target_idx].item())
            target_confs.append(target_conf)

            topk_labels = [labels[int(j.item())] for j in sorted_idx[:top_k]]
            frame_predictions.append(
                {
                    "frame_index": i,
                    "path": sampled_paths[i] if i < len(sampled_paths) else None,
                    "top_label": labels[top1],
                    "top_prob": float(row_p[top1].item()),
                    "target_prob": target_conf,
                    "target_rank": rank,
                    "topk_labels": topk_labels,
                }
            )

        n = float(len(frame_predictions) or 1)
        target_top1_ratio = top1_hits / n
        target_topk_ratio = topk_hits / n
        mean_margin = sum(margins) / max(1, len(margins))
        mean_target_conf = sum(target_confs) / max(1, len(target_confs))

        score = _aggregate_threshold_score(
            target_top1_ratio=target_top1_ratio,
            target_topk_ratio=target_topk_ratio,
            mean_margin=mean_margin,
            mean_target_confidence=mean_target_conf,
            thresholds=self._thresholds,
        )

        suggested_keywords = _suggest_keywords(
            target_label=target,
            label_histogram=label_hist,
            prompt=generation.spec.prompt,
        )
        feedback = ""
        if score < 0.85:
            confusion = sorted(label_hist.items(), key=lambda kv: (-kv[1], kv[0]))
            top_guess = confusion[0][0] if confusion else None
            if top_guess and top_guess != target:
                feedback = f"BiomedCLIP frames resemble '{top_guess}' more than target '{target}'; emphasize the target anatomy/view"
            else:
                feedback = "Increase medical/anatomical specificity; reduce stylization; enforce consistent anatomy"

        return ValidationScore(
            name=self.name,
            score=float(score),
            details={
                "target_label": target,
                "labels": labels,
                "n_frames_sampled": len(frame_predictions),
                "thresholds": {
                    "min_target_top1_ratio": self._thresholds.min_target_top1_ratio,
                    "min_target_topk_ratio": self._thresholds.min_target_topk_ratio,
                    "min_mean_margin": self._thresholds.min_mean_margin,
                    "min_mean_confidence": self._thresholds.min_mean_confidence,
                    "top_k": self._thresholds.top_k,
                },
                "metrics": {
                    "target_top1_ratio": target_top1_ratio,
                    "target_topk_ratio": target_topk_ratio,
                    "mean_margin": mean_margin,
                    "mean_target_confidence": mean_target_conf,
                },
                "label_histogram": label_hist,
                "frame_predictions": frame_predictions[: min(24, len(frame_predictions))],
                "suggested_keywords": suggested_keywords,
            },
            feedback=feedback,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        min_mem_mb_raw = os.environ.get("MEDICAL_DIFFUSION_BIOMEDCLIP_MIN_MEM_MB", "2048").strip()
        try:
            min_mem_mb = int(min_mem_mb_raw) if min_mem_mb_raw else 0
        except Exception:
            min_mem_mb = 2048
        if min_mem_mb > 0:
            total_mb = _linux_mem_total_mb()
            if total_mb is not None and total_mb < min_mem_mb:
                raise RuntimeError(
                    f"BiomedCLIP disabled: RAM {total_mb}MB < {min_mem_mb}MB "
                    "(increase Railway memory or set MEDICAL_DIFFUSION_BIOMEDCLIP_MIN_MEM_MB=0 to force)"
                )

        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            from open_clip import create_model_from_pretrained, get_tokenizer
        except Exception as e:
            raise RuntimeError(
                "BiomedCLIP validator requires: pip install open_clip_torch==2.23.0 transformers==4.35.2 torch pillow"
            ) from e

        device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = create_model_from_pretrained(self._hf_id)
        tokenizer = get_tokenizer(self._hf_id)

        self._torch = torch
        self._F = F
        self._pil_image = Image
        self._model = model.to(device)
        self._model.eval()
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._device = device

    def _encode_images(self, images: Sequence[Any]) -> Any:
        torch = self._torch
        assert torch is not None
        imgs = torch.stack([self._preprocess(im) for im in images]).to(self._device)
        inference_mode = getattr(torch, "inference_mode", torch.no_grad)
        with inference_mode():
            feats = self._model.encode_image(imgs)
            feats = self._F.normalize(feats, dim=-1)
        return feats

    def _encode_texts(self, prompts: Sequence[str]) -> Any:
        torch = self._torch
        assert torch is not None
        key = tuple(prompts)
        if key in self._text_feature_cache:
            return self._text_feature_cache[key]

        try:
            tokens = self._tokenizer(list(prompts), context_length=self._context_length)
        except TypeError:
            tokens = self._tokenizer(list(prompts))
        tokens = tokens.to(self._device)
        inference_mode = getattr(torch, "inference_mode", torch.no_grad)
        with inference_mode():
            feats = self._model.encode_text(tokens)
            feats = self._F.normalize(feats, dim=-1)
        self._text_feature_cache[key] = feats
        return feats

    def _similarity_logits(self, image_feats: Any, text_feats: Any) -> Any:
        torch = self._torch
        assert torch is not None
        logits = image_feats @ text_feats.T
        if hasattr(self._model, "logit_scale"):
            logits = logits * self._model.logit_scale.exp()
        return logits


def _aggregate_threshold_score(
    *,
    target_top1_ratio: float,
    target_topk_ratio: float,
    mean_margin: float,
    mean_target_confidence: float,
    thresholds: BiomedCLIPThresholds,
) -> float:
    def ratio(v: float, t: float) -> float:
        if t <= 0:
            return 1.0
        return max(0.0, min(1.0, v / t))

    parts = [
        ratio(target_top1_ratio, thresholds.min_target_top1_ratio),
        ratio(target_topk_ratio, thresholds.min_target_topk_ratio),
        ratio(mean_margin, thresholds.min_mean_margin),
        ratio(mean_target_confidence, thresholds.min_mean_confidence),
    ]
    return float(sum(parts) / len(parts))


def _suggest_keywords(*, target_label: str, label_histogram: Dict[str, int], prompt: str) -> List[str]:
    suggestions = [
        target_label,
        "medically accurate",
        "anatomically correct",
        "textbook style",
        "clinical illustration",
        "high detail anatomy",
        "no cartoon",
        "no fantasy",
    ]
    # If the model is consistently predicting a different label, include a disambiguation cue.
    ranked = sorted(label_histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    if ranked and ranked[0][0] != target_label:
        suggestions.append(f"focus on {target_label}, not {ranked[0][0]}")
    # If the prompt already contains "cartoon"/"stylized", nudge toward realism.
    if any(w in prompt.lower() for w in ["cartoon", "anime", "stylized", "pixar", "comic"]):
        suggestions.append("photorealistic medical rendering")
    # De-dupe while preserving order.
    seen = set()
    out: List[str] = []
    for s in suggestions:
        s = str(s).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out[:12]


def _default_labels() -> List[str]:
    return [
        "heart",
        "lung",
        "liver",
        "kidney",
        "brain",
        "stomach",
        "pancreas",
        "spleen",
        "colon",
        "small intestine",
        "blood vessel",
        "capillary",
        "red blood cells",
        "white blood cells",
        "artery",
        "vein",
    ]
