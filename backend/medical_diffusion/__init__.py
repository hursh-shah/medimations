"""Medical diffusion animation pipeline (hackathon skeleton)."""

from .agent import (
    AgentConfig,
    GeminiPromptAdjuster,
    PromptAdjuster,
    RuleBasedPromptAdjuster,
    SmartAgentConfig,
    SmartValidatorAgent,
    ValidatorAgent,
)
from .edit_analyzer import (
    CorrectionAction,
    CorrectionPlan,
    EditInstruction,
    analyze_for_corrections,
)
from .generation.veo_genai import (
    ExtendVideoResult,
    VeoGenaiBackend,
    VideoEditMode,
    VideoEditResult,
)
from .mask_generator import (
    MaskRegion,
    generate_elliptical_mask,
    generate_mask_from_description,
    generate_rectangular_mask,
    parse_region_description,
)
from .types import (
    AgentResult,
    AgentRound,
    AnimationSpec,
    GenerationResult,
    ValidationReport,
    ValidationScore,
)

__all__ = [
    # Agent
    "AgentConfig",
    "AgentResult",
    "AgentRound",
    "GeminiPromptAdjuster",
    "PromptAdjuster",
    "RuleBasedPromptAdjuster",
    "SmartAgentConfig",
    "SmartValidatorAgent",
    "ValidatorAgent",
    # Edit analyzer
    "CorrectionAction",
    "CorrectionPlan",
    "EditInstruction",
    "analyze_for_corrections",
    # Generation
    "AnimationSpec",
    "ExtendVideoResult",
    "GenerationResult",
    "VeoGenaiBackend",
    "VideoEditMode",
    "VideoEditResult",
    # Mask generation
    "MaskRegion",
    "generate_elliptical_mask",
    "generate_mask_from_description",
    "generate_rectangular_mask",
    "parse_region_description",
    # Validation
    "ValidationReport",
    "ValidationScore",
]
