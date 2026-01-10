from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from medmnist import INFO, OrganMNIST3D
except Exception:  # pragma: no cover
    INFO = None
    OrganMNIST3D = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


@dataclass(frozen=True)
class OrganMNIST3DCacheConfig:
    root: Path
    size: int = 28  # 28 or 64
    split: str = "train"  # train/val/test
    embed_size: int = 32
    projection_axis: int = 0
    max_per_label: int = 64


@dataclass(frozen=True)
class OrganMNIST3DPrototypes:
    label_names: List[str]  # index-aligned (0..10)
    prototypes: np.ndarray  # shape (n_labels, embed_dim), normalized
    embed_size: int

    def similarity(self, vec: np.ndarray) -> np.ndarray:
        return self.prototypes @ vec


def label_names() -> List[str]:
    info = _require_info()
    labels = info["label"]
    out = []
    for i in range(len(labels)):
        out.append(str(labels[str(i)]))
    return out


def infer_target_label(prompt: str) -> Optional[int]:
    """
    Returns OrganMNIST3D label index if the prompt mentions one of the organs.
    """
    prompt_norm = " " + _normalize_text(prompt) + " "
    best: Optional[Tuple[int, int]] = None  # (idx, match_len)
    for idx, name in enumerate(label_names()):
        cand = _label_patterns(name)
        for pat in cand:
            if f" {pat} " in prompt_norm:
                m = (idx, len(pat))
                if best is None or m[1] > best[1]:
                    best = m
    return best[0] if best else None


def load_or_build_prototypes(
    *,
    cfg: OrganMNIST3DCacheConfig,
    download: bool,
    force_rebuild: bool = False,
    export_gifs_dir: Optional[Path] = None,
    gifs_per_label: int = 0,
    gif_scale: int = 4,
) -> OrganMNIST3DPrototypes:
    cache_dir = cfg.root / "organmnist3d"
    cache_dir.mkdir(parents=True, exist_ok=True)
    proto_path = cache_dir / f"prototypes_size{cfg.size}_{cfg.split}_e{cfg.embed_size}.npz"

    if proto_path.exists() and not force_rebuild:
        data = np.load(proto_path)
        names = [str(x) for x in data["label_names"].tolist()]
        protos = data["prototypes"].astype(np.float32)
        embed_size = int(data["embed_size"])
        return OrganMNIST3DPrototypes(label_names=names, prototypes=protos, embed_size=embed_size)

    ds = _load_dataset(cfg=cfg, download=download)
    names = label_names()
    n_labels = len(names)
    dim = cfg.embed_size * cfg.embed_size

    sums = np.zeros((n_labels, dim), dtype=np.float32)
    counts = np.zeros((n_labels,), dtype=np.int32)
    exported = np.zeros((n_labels,), dtype=np.int32)

    if export_gifs_dir is not None and gifs_per_label > 0:
        export_gifs_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(ds)):
        vol, y = ds[i]
        label = _to_int_label(y)
        if label < 0 or label >= n_labels:
            continue
        if counts[label] >= cfg.max_per_label:
            continue

        vol3 = _to_volume3d(vol)
        proj = _project(vol3, axis=cfg.projection_axis)
        vec = embed_2d(proj, embed_size=cfg.embed_size)

        sums[label] += vec
        counts[label] += 1

        if export_gifs_dir is not None and gifs_per_label > 0 and exported[label] < gifs_per_label:
            out_dir = export_gifs_dir / names[label]
            out_path = out_dir / f"{cfg.split}_{i:04d}.gif"
            try:
                save_volume_gif(vol3, out_path, axis=cfg.projection_axis, scale=gif_scale)
                exported[label] += 1
            except Exception:
                # GIF export is optional; don't block prototype build.
                pass

        if int(counts.sum()) >= cfg.max_per_label * n_labels:
            break

    protos = sums / np.maximum(counts[:, None].astype(np.float32), 1.0)
    protos = _l2_normalize_rows(protos)

    np.savez_compressed(
        proto_path,
        label_names=np.array(names, dtype=str),
        prototypes=protos.astype(np.float32),
        embed_size=np.array(cfg.embed_size, dtype=np.int32),
        size=np.array(cfg.size, dtype=np.int32),
        split=np.array(cfg.split, dtype=str),
        projection_axis=np.array(cfg.projection_axis, dtype=np.int32),
        max_per_label=np.array(cfg.max_per_label, dtype=np.int32),
        counts=counts,
    )
    return OrganMNIST3DPrototypes(label_names=names, prototypes=protos, embed_size=cfg.embed_size)


def save_volume_gif(volume: np.ndarray, out_path: Path, *, axis: int = 0, scale: int = 4, duration_ms: int = 50) -> None:
    if Image is None:
        raise RuntimeError("Pillow not installed; cannot export GIFs")

    vol3 = _to_volume3d(volume)
    vol3 = np.moveaxis(vol3, axis, 0)  # depth first

    frames = []
    for sl in vol3:
        sl_u8 = _to_uint8(sl)
        img = Image.fromarray(sl_u8, mode="L")
        if scale and scale != 1:
            img = img.resize((img.size[0] * scale, img.size[1] * scale), resample=Image.NEAREST)
        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def embed_2d(img: np.ndarray, *, embed_size: int) -> np.ndarray:
    """
    Returns a normalized (embed_size*embed_size,) vector in [-1, 1] space.
    """
    img_f = img.astype(np.float32)
    img_f = resize_nearest(img_f, embed_size, embed_size)
    img_f = _standardize(img_f)
    vec = img_f.reshape(-1)
    denom = float(np.linalg.norm(vec) + 1e-8)
    return (vec / denom).astype(np.float32)


def resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == new_h and w == new_w:
        return img
    ys = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, new_w)).astype(np.int32)
    return img[np.ix_(ys, xs)]


def _load_dataset(*, cfg: OrganMNIST3DCacheConfig, download: bool):
    if OrganMNIST3D is None:
        raise RuntimeError("medmnist is not installed")
    return OrganMNIST3D(
        split=cfg.split,
        download=download,
        size=cfg.size,
        root=str(cfg.root),
    )


def _require_info() -> Dict:
    if INFO is None:
        raise RuntimeError("medmnist is not installed")
    if "organmnist3d" not in INFO:
        raise RuntimeError("medmnist INFO missing organmnist3d")
    return INFO["organmnist3d"]


def _project(vol3: np.ndarray, *, axis: int) -> np.ndarray:
    # Mean projection yields a stable “gist” for prototypes.
    return vol3.astype(np.float32).mean(axis=axis)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    lo = float(arr_f.min())
    hi = float(arr_f.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr_f - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def _standardize(arr: np.ndarray) -> np.ndarray:
    mu = float(arr.mean())
    sigma = float(arr.std())
    if sigma < 1e-6:
        return arr - mu
    return (arr - mu) / sigma


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)


def _to_volume3d(x) -> np.ndarray:
    # medmnist can return numpy arrays or torch tensors.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    return arr


def _to_int_label(y) -> int:
    if hasattr(y, "item"):
        try:
            return int(y.item())
        except Exception:
            pass
    arr = np.asarray(y)
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    raise ValueError(f"Unexpected label shape: {arr.shape}")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_patterns(label_name: str) -> List[str]:
    """
    "kidney-right" -> ["kidney right", "right kidney"]
    """
    base = label_name.replace("-", " ").strip()
    parts = base.split()
    if len(parts) >= 2 and parts[-1] in {"right", "left"}:
        side = parts[-1]
        organ = " ".join(parts[:-1])
        return [f"{organ} {side}", f"{side} {organ}"]
    return [base]

