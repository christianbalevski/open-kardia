"""kardia6l — research-grade macOS BLE interface for the KardiaMobile 6L.

NOT A MEDICAL DEVICE. Output is a research signal, not clinically validated,
and must not be used for diagnosis.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "kardia6l is an independent research tool. It is NOT a medical device, is NOT "
    "clinically validated, and must NOT be used for diagnosis or treatment decisions. "
    "Use only with hardware you own."
)

_printed = False


def print_disclaimer(file=None) -> None:
    """Print the non-diagnostic disclaimer once per process."""
    global _printed
    if _printed:
        return
    _printed = True
    import sys

    print(f"[kardia6l] {DISCLAIMER}", file=file or sys.stderr)
