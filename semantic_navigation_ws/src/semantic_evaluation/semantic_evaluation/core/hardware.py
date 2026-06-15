"""Hardware utilisation sampling — pure Python, no rclpy.

Wraps ``psutil`` behind a tiny interface that degrades gracefully when the
dependency is missing (samples come back as NaN instead of raising). CPU percent
is measured as a delta since the previous call, so ``refresh()`` is meant to be
driven from a node timer to keep the delta window warm, while ``sample()`` reads
the instantaneous value at the moment a test case resolves.
"""
from __future__ import annotations

from semantic_evaluation.core.metrics import HardwareSample

try:  # psutil is an optional (rosdep: python3-psutil) runtime dependency.
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

_BYTES_PER_MB = 1024.0 * 1024.0


class HardwareSampler:
    """Samples CPU% and used RAM, with an optional GPU/VRAM extension point."""

    def __init__(self) -> None:
        self._available = _HAS_PSUTIL
        if self._available:
            # Prime the cpu_percent delta so the first real sample is meaningful.
            psutil.cpu_percent(interval=None)

    @property
    def available(self) -> bool:
        """True when psutil is importable and sampling will yield real values."""
        return self._available

    def refresh(self) -> None:
        """Advance the CPU-percent delta window (call from a periodic timer)."""
        if self._available:
            psutil.cpu_percent(interval=None)

    def sample(self) -> HardwareSample:
        """Read an instantaneous utilisation sample (NaN fields if unavailable)."""
        if not self._available:
            return HardwareSample()  # NaN, NaN

        cpu_percent = float(psutil.cpu_percent(interval=None))
        ram_used_mb = float(psutil.virtual_memory().used) / _BYTES_PER_MB

        # ── GPU / VRAM extension point (Jetson) ───────────────────────────── #
        # On NVIDIA Jetson, augment HardwareSample with GPU load and VRAM here,
        # e.g. via `jtop` (jetson-stats) or by parsing `tegrastats`:
        #     from jtop import jtop
        #     with jtop() as jt:
        #         gpu_percent = jt.gpu['GPU']['val']
        #         vram_used_mb = jt.ram['used'] / 1024.0
        # Keep the import guarded so non-Jetson hosts are unaffected.

        return HardwareSample(cpu_percent=cpu_percent, ram_used_mb=ram_used_mb)
