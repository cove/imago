"""Deprecated: use _caption_matching instead."""
from ._caption_matching import (  # noqa: F401
    CAPTION_MATCHING_PROMPT as GEMMA4_CAPTION_PROMPT,
    DEFAULT_TIMEOUT as DEFAULT_GEMMA4_TIMEOUT,
    assign_captions_from_lmstudio as assign_captions_from_gemma4,
    call_lmstudio_caption_matching as call_gemma4_caption_matching,
    sort_regions_reading_order,
)
