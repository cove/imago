from __future__ import annotations

try:
    from .scanwatch import IncomingScanHandler, ScanWatchService, main
except ImportError:
    from scanwatch import IncomingScanHandler, ScanWatchService, main

__all__ = ["IncomingScanHandler", "ScanWatchService", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
