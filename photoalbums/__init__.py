"""Photo albums pipeline package."""

from importlib import import_module


def __getattr__(name: str):
    if name == "stitch_oversized_pages":
        return import_module(f"{__name__}.stitch_oversized_pages")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["stitch_oversized_pages"]
