"""Derive the augmented / limb leads from Lead I and Lead II.

Standard identities (Einthoven / Goldberger):
    III = II - I
    aVR = -(I + II) / 2
    aVL = I - II / 2      (== (I - III) / 2)
    aVF = II - I / 2      (== (II + III) / 2)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEAD_NAMES_6 = ("I", "II", "III", "aVR", "aVL", "aVF")


@dataclass(frozen=True)
class SixLeads:
    I: np.ndarray
    II: np.ndarray
    III: np.ndarray
    aVR: np.ndarray
    aVL: np.ndarray
    aVF: np.ndarray

    def as_matrix(self) -> np.ndarray:
        """(n_samples, 6) array ordered as LEAD_NAMES_6."""
        return np.column_stack([getattr(self, n) for n in LEAD_NAMES_6])

    def __getitem__(self, name: str) -> np.ndarray:
        return getattr(self, name)


def derive_leads(lead_i, lead_ii) -> SixLeads:
    """Compute III, aVR, aVL, aVF from I and II (same length, same units)."""
    I = np.asarray(lead_i, dtype=np.float64)
    II = np.asarray(lead_ii, dtype=np.float64)
    if I.shape != II.shape:
        raise ValueError(f"Lead I and II must have the same shape, got {I.shape} vs {II.shape}")
    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - II / 2.0
    aVF = II - I / 2.0
    return SixLeads(I=I, II=II, III=III, aVR=aVR, aVL=aVL, aVF=aVF)
