"""
Mask generation utilities for video editing.

Provides utilities to create mask images for Veo's INSERT/REMOVE operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class MaskRegion:
    """Defines a rectangular region for mask generation."""
    x: float  # 0-1, left edge
    y: float  # 0-1, top edge
    width: float  # 0-1
    height: float  # 0-1

    def to_pixels(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2)."""
        x1 = int(self.x * img_width)
        y1 = int(self.y * img_height)
        x2 = int((self.x + self.width) * img_width)
        y2 = int((self.y + self.height) * img_height)
        return (x1, y1, x2, y2)


# Predefined regions for common anatomical positions
REGION_PRESETS = {
    "center": MaskRegion(0.25, 0.25, 0.5, 0.5),
    "center_small": MaskRegion(0.35, 0.35, 0.3, 0.3),
    "center_large": MaskRegion(0.15, 0.15, 0.7, 0.7),
    "top": MaskRegion(0.2, 0.05, 0.6, 0.35),
    "bottom": MaskRegion(0.2, 0.6, 0.6, 0.35),
    "left": MaskRegion(0.05, 0.2, 0.35, 0.6),
    "right": MaskRegion(0.6, 0.2, 0.35, 0.6),
    "top_left": MaskRegion(0.05, 0.05, 0.4, 0.4),
    "top_right": MaskRegion(0.55, 0.05, 0.4, 0.4),
    "bottom_left": MaskRegion(0.05, 0.55, 0.4, 0.4),
    "bottom_right": MaskRegion(0.55, 0.55, 0.4, 0.4),
    "upper_quadrant": MaskRegion(0.15, 0.05, 0.7, 0.45),
    "lower_quadrant": MaskRegion(0.15, 0.5, 0.7, 0.45),
    "full": MaskRegion(0.0, 0.0, 1.0, 1.0),
}


def parse_region_description(description: str) -> Optional[MaskRegion]:
    """
    Parse a natural language region description into a MaskRegion.
    
    Args:
        description: Text like "upper left", "center", "bottom right quadrant"
        
    Returns:
        MaskRegion or None if parsing fails
    """
    desc = description.lower().strip()
    
    # Direct preset matches
    for preset_name, region in REGION_PRESETS.items():
        if preset_name.replace("_", " ") in desc or preset_name in desc:
            return region
    
    # Parse compound descriptions
    has_top = any(w in desc for w in ["top", "upper", "above"])
    has_bottom = any(w in desc for w in ["bottom", "lower", "below"])
    has_left = "left" in desc
    has_right = "right" in desc
    has_center = "center" in desc or "middle" in desc
    
    if has_center and not (has_top or has_bottom or has_left or has_right):
        return REGION_PRESETS["center"]
    
    if has_top and has_left:
        return REGION_PRESETS["top_left"]
    if has_top and has_right:
        return REGION_PRESETS["top_right"]
    if has_bottom and has_left:
        return REGION_PRESETS["bottom_left"]
    if has_bottom and has_right:
        return REGION_PRESETS["bottom_right"]
    if has_top:
        return REGION_PRESETS["top"]
    if has_bottom:
        return REGION_PRESETS["bottom"]
    if has_left:
        return REGION_PRESETS["left"]
    if has_right:
        return REGION_PRESETS["right"]
    
    # Default to center
    return REGION_PRESETS["center"]


def generate_rectangular_mask(
    *,
    region: MaskRegion,
    output_path: Path,
    width: int = 720,
    height: int = 1280,
    feather: int = 0,
) -> Path:
    """
    Generate a rectangular mask image.
    
    The mask is black with a white rectangle in the specified region.
    White pixels indicate the area to edit.
    
    Args:
        region: The MaskRegion defining the edit area
        output_path: Where to save the mask PNG
        width: Image width in pixels
        height: Image height in pixels
        feather: Optional feathering/blur radius (0 = hard edges)
        
    Returns:
        Path to the generated mask file
    """
    if not HAS_PIL:
        raise RuntimeError("PIL/Pillow is required for mask generation. Install with: pip install Pillow")
    
    # Create black image
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)
    
    # Get pixel coordinates
    x1, y1, x2, y2 = region.to_pixels(width, height)
    
    # Draw white rectangle
    draw.rectangle([x1, y1, x2, y2], fill=255)
    
    # Apply feathering if requested
    if feather > 0:
        try:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=feather))
        except Exception:
            pass  # Skip feathering if it fails
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    
    return output_path


def generate_elliptical_mask(
    *,
    region: MaskRegion,
    output_path: Path,
    width: int = 720,
    height: int = 1280,
    feather: int = 10,
) -> Path:
    """
    Generate an elliptical mask image.
    
    The mask is black with a white ellipse in the specified region.
    Often better for organic shapes like organs.
    
    Args:
        region: The MaskRegion defining the bounding box for the ellipse
        output_path: Where to save the mask PNG
        width: Image width in pixels
        height: Image height in pixels
        feather: Optional feathering/blur radius
        
    Returns:
        Path to the generated mask file
    """
    if not HAS_PIL:
        raise RuntimeError("PIL/Pillow is required for mask generation. Install with: pip install Pillow")
    
    # Create black image
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)
    
    # Get pixel coordinates
    x1, y1, x2, y2 = region.to_pixels(width, height)
    
    # Draw white ellipse
    draw.ellipse([x1, y1, x2, y2], fill=255)
    
    # Apply feathering
    if feather > 0:
        try:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=feather))
        except Exception:
            pass
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    
    return output_path


def generate_mask_from_description(
    *,
    description: str,
    output_path: Path,
    width: int = 720,
    height: int = 1280,
    shape: Literal["rectangle", "ellipse"] = "ellipse",
    feather: int = 10,
) -> Path:
    """
    Generate a mask image from a natural language region description.
    
    This is a convenience function that parses the description and
    generates an appropriate mask.
    
    Args:
        description: Natural language description like "upper left quadrant"
        output_path: Where to save the mask PNG
        width: Image width in pixels
        height: Image height in pixels
        shape: "rectangle" or "ellipse"
        feather: Feathering radius for soft edges
        
    Returns:
        Path to the generated mask file
    """
    region = parse_region_description(description)
    if region is None:
        region = REGION_PRESETS["center"]
    
    if shape == "ellipse":
        return generate_elliptical_mask(
            region=region,
            output_path=output_path,
            width=width,
            height=height,
            feather=feather,
        )
    else:
        return generate_rectangular_mask(
            region=region,
            output_path=output_path,
            width=width,
            height=height,
            feather=feather,
        )


def extract_first_frame(video_path: Path, output_path: Path) -> Path:
    """
    Extract the first frame from a video for mask reference.
    
    Uses ffmpeg to extract frame 0.
    
    Args:
        video_path: Path to the video file
        output_path: Where to save the frame PNG
        
    Returns:
        Path to the extracted frame
    """
    import subprocess
    
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vframes", "1",
        "-f", "image2",
        str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract frame: {result.stderr}")
    
    if not output_path.exists():
        raise RuntimeError("ffmpeg did not produce output frame")
    
    return output_path


def get_video_dimensions(video_path: Path) -> Tuple[int, int]:
    """
    Get the width and height of a video.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        (width, height) tuple
    """
    import subprocess
    import json as json_module
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    data = json_module.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video streams found")
    
    width = streams[0].get("width", 720)
    height = streams[0].get("height", 1280)
    
    return (width, height)
