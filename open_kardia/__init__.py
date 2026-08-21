"""open-kardia — research-grade macOS BLE interface for the KardiaMobile 6L.

NOT A MEDICAL DEVICE. Output is a research signal, not clinically validated,
and must not be used for diagnosis.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "open-kardia is an independent research tool. It is NOT a medical device, is NOT "
    "clinically validated, and must NOT be used for diagnosis or treatment decisions. "
    "Use only with hardware you own."
)

def data_dir():
    """~/.open-kardia, migrating a pre-rename ~/.kardia6l directory the first time.

    The rename must not orphan existing readings / the cached device UUID / the live event
    log: if the old directory exists and the new one does not, move it. If the move fails
    (permissions, cross-device), fall back to using the old directory in place."""
    from pathlib import Path
    new = Path.home() / ".open-kardia"
    for old_name in (".kardia6l", ".openkardia"):     # pre-rename directory names
        old = Path.home() / old_name
        if not new.exists() and old.exists():
            try:
                import os
                os.rename(old, new)
            except OSError:
                return old
    return new


_printed = False


def print_disclaimer(file=None) -> None:
    """Print the non-diagnostic disclaimer once per process."""
    global _printed
    if _printed:
        return
    _printed = True
    import sys

    print(f"[open-kardia] {DISCLAIMER}", file=file or sys.stderr)
