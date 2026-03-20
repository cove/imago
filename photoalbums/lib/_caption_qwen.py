"""Backward-compatibility shim. Use _caption_local_hf instead."""
from ._caption_local_hf import (  # noqa: F401
    DEFAULT_LOCAL_AUTO_MAX_PIXELS as DEFAULT_QWEN_AUTO_MAX_PIXELS,
    DEFAULT_LOCAL_CAPTION_MODEL as DEFAULT_QWEN_CAPTION_MODEL,
    LOCAL_ATTN_IMPLEMENTATIONS as QWEN_ATTN_IMPLEMENTATIONS,
    LocalHFCaptioner as QwenLocalCaptioner,
    _load_hf_transformers as _load_qwen_transformers,
    _parse_local_combined_json_output as _parse_qwen_combined_json_output,
    _parse_local_json_output as _parse_qwen_json_output,
    normalize_local_attn_implementation as normalize_qwen_attn_implementation,
)
from .ai_ocr import _resolve_local_hf_snapshot  # noqa: F401
