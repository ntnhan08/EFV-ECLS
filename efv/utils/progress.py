"""EFV · Progress Bar – lightweight terminal spinner, no dependencies."""
from __future__ import annotations

import sys
import time


class ProgressBar:
    """
    A minimal single-line progress bar.

    Example
    -------
    >>> bar = ProgressBar(total=100, label="Processing")
    >>> for i in range(100):
    ...     bar.update(i + 1)
    >>> bar.finish()
    """

    BAR_WIDTH = 30

    def __init__(self, total: int, label: str = "EFV"):
        self.total    = max(1, total)
        self.label    = label
        self.start    = time.monotonic()
        self._last_pct = -1
        self._tty     = sys.stderr.isatty()

    def update(self, done: int) -> None:
        pct = int(done / self.total * 100)
        if pct == self._last_pct:
            return
        self._last_pct = pct

        filled  = int(self.BAR_WIDTH * done / self.total)
        bar     = "█" * filled + "░" * (self.BAR_WIDTH - filled)
        elapsed = time.monotonic() - self.start
        fps     = done / elapsed if elapsed > 0 else 0

        if self.total > done > 0:
            eta = (self.total - done) / (done / elapsed) if elapsed > 0 else 0
            eta_str = f"ETA {eta:.0f}s"
        else:
            eta_str = "      "

        line = (
            f"\r[EFV] {self.label}: [{bar}] {pct:3d}% "
            f"{done}/{self.total} frames  {fps:.1f} fps  {eta_str}"
        )

        if self._tty:
            sys.stderr.write(line)
            sys.stderr.flush()
        else:
            if pct in (0, 25, 50, 75, 100):
                sys.stderr.write(line + "\n")
                sys.stderr.flush()

    def finish(self) -> None:
        elapsed = time.monotonic() - self.start
        fps     = self.total / elapsed if elapsed > 0 else 0
        filled  = self.BAR_WIDTH
        bar     = "█" * filled
        line    = (
            f"\r[EFV] {self.label}: [{bar}] 100%  "
            f"{self.total}/{self.total} frames  {fps:.1f} fps  "
            f"Done in {elapsed:.2f}s\n"
        )
        sys.stderr.write(line)
        sys.stderr.flush()
