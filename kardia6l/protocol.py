"""Wire-format decoding for KardiaMobile 6L notifications.

STATUS: THE 6L FORMAT IS NOT YET KNOWN. Nothing in this module is a verified
constant. This file provides:

  * `FrameFormat`   — a fully parameterised description of one candidate byte layout
                      (header bytes, sample width, endianness, signedness, channels,
                      interleave, sequence-counter position, gain/offset).
  * `Decoder`       — turns raw notification payloads into (n, channels) sample arrays,
                      tracking a sequence counter for gap detection when configured.
  * `CANDIDATES`    — the enumerated hypothesis space from spec §4.3.
  * `score_format`  — how ECG-like is a decoded stream? Used by tools/analyze_capture.py
                      to pick the winning hypothesis from a Phase-1 capture.
  * `entropy_report`— §5 diagnostic: is the payload plaintext-ish or ciphertext-like?

Once Phase 1 identifies the real layout, set `KNOWN_FORMAT` and pin it with a golden
vector in tests/test_protocol.py.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- format
@dataclass(frozen=True)
class FrameFormat:
    name: str
    header_bytes: int = 0            # bytes to skip at start of each notification
    trailer_bytes: int = 0           # bytes to ignore at the end
    sample_bytes: int = 2            # 1, 2, 3, 4
    little_endian: bool = True
    signed: bool = True
    channels: int = 1                # 1 (Lead I) or 2 (Lead I, Lead II)
    interleaved: bool = True         # True: s0c0 s0c1 s1c0 s1c1 ...; False: block per channel
    seq_offset: int | None = None    # byte offset of a sequence counter within the header
    seq_bytes: int = 1
    seq_little_endian: bool = True
    gain_mv: float = 1.0             # mV per LSB (unknown until fitted; 1.0 = raw counts)
    offset_counts: float = 0.0       # raw value that corresponds to 0 mV
    sample_rate_hz: float | None = None  # measured, not assumed

    @property
    def frame_bytes(self) -> int:
        return self.sample_bytes * self.channels

    def payload_slice(self, n: int) -> tuple[int, int]:
        lo = self.header_bytes
        hi = n - self.trailer_bytes
        return lo, hi


# The unknown-format hypothesis space (spec §4.3). Header sizes 0..4, widths 2/3,
# both endians, both signs, 1/2 channels, interleave both ways.
def enumerate_candidates(header_range: Iterable[int] = (0, 1, 2, 3, 4),
                         widths: Iterable[int] = (2, 3),
                         channels: Iterable[int] = (1, 2)) -> list[FrameFormat]:
    out: list[FrameFormat] = []
    for h in header_range:
        for w in widths:
            for le in (True, False):
                for sg in (True, False):
                    for ch in channels:
                        for il in ((True, False) if ch > 1 else (True,)):
                            name = (f"h{h}_{'s' if sg else 'u'}{8 * w}{'le' if le else 'be'}"
                                    f"_c{ch}{'i' if il else 'b'}")
                            out.append(FrameFormat(name, header_bytes=h, sample_bytes=w,
                                                   little_endian=le, signed=sg, channels=ch,
                                                   interleaved=il,
                                                   seq_offset=0 if h else None))
    return out


CANDIDATES: list[FrameFormat] = enumerate_candidates()

# --------------------------------------------------------------------------- KardiaMobile 6L
# Established 2026-08-16 from a PacketLogger capture of the iOS Kardia app talking to a
# retail 6L (AC-019, fw 3.0.1). See tools/capture_decode_notes.md.
#   * ECG stream: vendor characteristic ac060003-328c-a28f-9846-5a8aa212661b (notify)
#   * 36-byte notifications, ~33.4/s -> 9 samples/channel each -> 300 Hz
#   * signed int16 little-endian, 2 channels interleaved: [I0, II0, I1, II1, ...]
#   * plaintext (no app-layer encryption); no per-notification header or sequence counter
#   * gain: PROVISIONAL. Fit from an app export via validate/compare.py to finalise.
KARDIA_6L_DATA_CHAR = "ac060003-328c-a28f-9846-5a8aa212661b"
KARDIA_6L_CMD_CHAR = "ac060002-328c-a28f-9846-5a8aa212661b"
# mV per LSB. ESTIMATE (~2x uncertain). The Kardia app PDF's 1 mV calibration pulse is
# exact (measured 28.35 pt = 1.000 mV at 10 mm/mV), but the only PDF+counts pairs available
# were from *different* sessions and the app applies an amplitude-altering "Enhanced Filter",
# so Lead I and Lead II amplitude-matching bracketed the gain at 8.9e-5..1.8e-4 mV/LSB.
# This is their geometric mean. For an exact value, capture ONE recording with PacketLogger
# (gives raw counts) AND export THAT SAME recording's PDF (gives mV), then run
# tools/pdf_calibrate.py fit-pklg — see tools/capture_decode_notes.md.
KARDIA_6L_GAIN_MV = 1.27e-4   # ESTIMATE, mV/LSB (geo-mean of per-lead cross-session fits)

KNOWN_FORMAT: FrameFormat = FrameFormat(
    name="kardia6l_6l", header_bytes=0, sample_bytes=2, little_endian=True, signed=True,
    channels=2, interleaved=True, seq_offset=None, gain_mv=KARDIA_6L_GAIN_MV,
    offset_counts=0.0, sample_rate_hz=300.0,
)


# --------------------------------------------------------------------------- decoder
def _unpack(buf: bytes, fmt: FrameFormat) -> np.ndarray:
    """Return int64 array of raw counts from a payload slice (already header-stripped)."""
    w = fmt.sample_bytes
    n = len(buf) // w
    buf = buf[: n * w]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if w in (1, 2, 4):
        dt = np.dtype(f"{'<' if fmt.little_endian else '>'}{'i' if fmt.signed else 'u'}{w}")
        return np.frombuffer(buf, dtype=dt).astype(np.int64)
    # 3-byte
    a = np.frombuffer(buf, dtype=np.uint8).reshape(n, 3).astype(np.int64)
    if fmt.little_endian:
        v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
    else:
        v = a[:, 2] | (a[:, 1] << 8) | (a[:, 0] << 16)
    if fmt.signed:
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
    return v


def _read_seq(payload: bytes, fmt: FrameFormat) -> int | None:
    if fmt.seq_offset is None:
        return None
    b = payload[fmt.seq_offset: fmt.seq_offset + fmt.seq_bytes]
    if len(b) < fmt.seq_bytes:
        return None
    return int.from_bytes(b, "little" if fmt.seq_little_endian else "big")


@dataclass
class DecodeStats:
    notifications: int = 0
    samples: int = 0
    gaps: int = 0
    lost_frames_est: int = 0
    last_seq: int | None = None
    sizes: dict[int, int] = field(default_factory=dict)


class Decoder:
    """Stateful decoder: feed payloads, get (n, channels) arrays in raw counts or mV."""

    def __init__(self, fmt: FrameFormat, to_mv: bool = True):
        self.fmt = fmt
        self.to_mv = to_mv
        self.stats = DecodeStats()

    def decode(self, payload: bytes | bytearray) -> np.ndarray:
        f = self.fmt
        payload = bytes(payload)
        st = self.stats
        st.notifications += 1
        st.sizes[len(payload)] = st.sizes.get(len(payload), 0) + 1

        seq = _read_seq(payload, f)
        if seq is not None:
            if st.last_seq is not None:
                mod = 1 << (8 * f.seq_bytes)
                expected = (st.last_seq + 1) % mod
                if seq != expected:
                    st.gaps += 1
                    st.lost_frames_est += (seq - expected) % mod
            st.last_seq = seq

        lo, hi = f.payload_slice(len(payload))
        raw = _unpack(payload[lo:hi], f)
        c = f.channels
        n = len(raw) // c
        raw = raw[: n * c]
        if c == 1:
            m = raw.reshape(n, 1)
        elif f.interleaved:
            m = raw.reshape(n, c)
        else:
            m = raw.reshape(c, n).T
        st.samples += n
        if self.to_mv:
            return (m.astype(np.float64) - f.offset_counts) * f.gain_mv
        return m

    def decode_many(self, payloads: Sequence[bytes]) -> np.ndarray:
        parts = [self.decode(p) for p in payloads]
        parts = [p for p in parts if len(p)]
        if not parts:
            return np.zeros((0, self.fmt.channels))
        return np.vstack(parts)


def encode(samples: np.ndarray, fmt: FrameFormat, per_notification: int,
           start_seq: int = 0) -> list[bytes]:
    """Inverse of Decoder — used by tests to build golden vectors and by the
    synthetic-device fixture. `samples` is (n, channels) raw integer counts."""
    s = np.asarray(samples)
    if s.ndim == 1:
        s = s[:, None]
    n, c = s.shape
    assert c == fmt.channels
    w = fmt.sample_bytes
    order = "little" if fmt.little_endian else "big"
    out = []
    seq = start_seq
    for i in range(0, n, per_notification):
        blk = s[i: i + per_notification]
        flat = blk.reshape(-1) if fmt.interleaved else blk.T.reshape(-1)
        body = b"".join(int(v).to_bytes(w, order, signed=fmt.signed) for v in flat)
        hdr = bytearray(fmt.header_bytes)
        if fmt.seq_offset is not None and fmt.header_bytes:
            hdr[fmt.seq_offset: fmt.seq_offset + fmt.seq_bytes] = (
                seq % (1 << (8 * fmt.seq_bytes))).to_bytes(
                fmt.seq_bytes, "little" if fmt.seq_little_endian else "big")
        out.append(bytes(hdr) + body + bytes(fmt.trailer_bytes))
        seq += 1
    return out


# --------------------------------------------------------------------------- §5 entropy
@dataclass
class EntropyReport:
    n_bytes: int
    shannon_bits: float        # 8.0 == uniformly random bytes
    compress_ratio: float      # zlib compressed / raw; ~1.0 == incompressible
    byte_lag1_autocorr: float  # ECG-ish little-endian streams show structure; ciphertext ~0
    verdict: str               # 'plaintext-like' | 'ciphertext-like' | 'ambiguous'


def entropy_report(payloads: Sequence[bytes], strip_header: int = 0) -> EntropyReport:
    blob = b"".join(bytes(p)[strip_header:] for p in payloads)
    n = len(blob)
    if n < 64:
        return EntropyReport(n, float("nan"), float("nan"), float("nan"), "ambiguous")
    counts = np.bincount(np.frombuffer(blob, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / n
    H = float(-(p * np.log2(p)).sum())
    cr = len(zlib.compress(blob, 9)) / n
    a = np.frombuffer(blob, dtype=np.uint8).astype(np.float64)
    a = a - a.mean()
    ac = float(np.dot(a[:-1], a[1:]) / (np.dot(a, a) + 1e-12))
    # Heuristics: real ADC streams have H well below 8, compress noticeably, and
    # (for multi-byte little-endian samples) show non-trivial byte autocorrelation.
    if H > 7.9 and cr > 0.98:
        verdict = "ciphertext-like"
    elif H < 7.5 or cr < 0.90:
        verdict = "plaintext-like"
    else:
        verdict = "ambiguous"
    return EntropyReport(n, H, cr, ac, verdict)


# --------------------------------------------------------------------------- scoring
@dataclass
class FormatScore:
    fmt: FrameFormat
    score: float
    sample_rate_est: float
    hr_bpm: float
    n_peaks: int
    band_ratio: float      # power in 0.5–40 Hz / total
    qrs_snr: float
    notes: str = ""


def _channel_metrics(x: np.ndarray, fs: float, duration_s: float) -> dict:
    """ECG-likeness metrics for one normalised channel."""
    from . import dsp
    from scipy import signal as ss

    x = x - np.median(x)
    sd = np.std(x)
    if sd == 0:
        return {"flat": True}
    x = x / sd
    dx = np.diff(x)
    roughness = float(np.std(dx) / (np.std(x) + 1e-9))   # ~0.1-0.4 for real ECG at 250-600 Hz
    f, p = ss.welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    total = p.sum() + 1e-12
    band = float(p[(f >= 0.5) & (f <= 40)].sum() / total)
    try:
        peaks = dsp.detect_r_peaks(x, fs)
        hr = dsp.heart_rate_from_peaks(peaks, fs)
    except Exception:  # noqa: BLE001
        peaks, hr = np.array([]), float("nan")
    n_peaks = int(len(peaks))
    rr = np.diff(peaks) / fs if n_peaks > 2 else np.array([])
    rr_cv = float(np.std(rr) / np.mean(rr)) if len(rr) > 1 else 1.0
    qrs_snr = 0.0
    if n_peaks >= 3:
        clean_x = dsp.bandpass(x, fs, 0.5, min(40, 0.45 * fs))
        mask = np.ones(len(x), bool)
        w = int(0.05 * fs)
        for pk in peaks:
            mask[max(0, pk - w): pk + w] = False
        # robust residual scale (MAD) so spiky aliases don't look "clean"
        resid = 1.4826 * np.median(np.abs(clean_x[mask] - np.median(clean_x[mask]))) + 1e-9
        qrs_snr = float(np.median(np.abs(clean_x[peaks])) / resid)
    expected = duration_s * (hr / 60.0) if np.isfinite(hr) else 0
    return {"flat": False, "roughness": roughness, "band": band, "hr": hr, "n_peaks": n_peaks,
            "rr_cv": rr_cv, "qrs_snr": qrs_snr, "expected_peaks": expected}


def _channel_score(m: dict) -> float:
    if m.get("flat"):
        return -1.0
    sc = 2.0 * m["band"]
    sc += 1.5 * max(0.0, 1.0 - m["roughness"])
    hr, n_peaks = m["hr"], m["n_peaks"]
    if 35 <= hr <= 200 and n_peaks >= 3:
        sc += 1.5
        sc += 1.0 * max(0.0, 1.0 - 2.0 * m["rr_cv"])
        sc += min(1.0, m["qrs_snr"] / 8.0)
        exp = m["expected_peaks"]
        if exp and abs(n_peaks - exp) / exp < 0.2:
            sc += 0.5
    return sc


def _header_plausibility(payloads: Sequence[bytes], fmt: FrameFormat) -> float:
    """+bonus if hypothesised header bytes behave like a counter or constant,
    -penalty if they vary like sample data."""
    if fmt.header_bytes == 0:
        return 0.0
    hdr = np.array([list(bytes(p)[: fmt.header_bytes]) for p in payloads if len(p) >= fmt.header_bytes])
    if len(hdr) < 4:
        return 0.0
    bonus = 0.0
    for col in range(hdr.shape[1]):
        d = np.diff(hdr[:, col].astype(int)) % 256
        frac_counter = np.mean(d == 1)
        frac_const = np.mean(d == 0)
        if frac_counter > 0.8 or frac_const > 0.8 or (frac_counter + frac_const) > 0.9:
            bonus += 0.4
        else:
            bonus -= 0.6
    return bonus - 0.05 * fmt.header_bytes


def score_format(fmt: FrameFormat, payloads: Sequence[bytes], duration_s: float,
                 sample_rate_hint: float | None = None) -> FormatScore:
    """Decode `payloads` (spanning `duration_s` seconds) with `fmt` and score how
    ECG-like the result looks. Higher is better. Robust to unknown gain (uses
    normalised signals). All channels are scored; ch0 metrics are reported."""
    dec = Decoder(fmt, to_mv=False)
    try:
        m = dec.decode_many(payloads)
    except Exception as e:  # noqa: BLE001
        return FormatScore(fmt, -1.0, 0.0, float("nan"), 0, 0.0, 0.0, f"decode error: {e}")
    if m.shape[0] < 32:
        return FormatScore(fmt, -1.0, 0.0, float("nan"), 0, 0.0, 0.0, "too few samples")
    fs = sample_rate_hint or (m.shape[0] / max(duration_s, 1e-6))
    if fs < 50 or fs > 5000:
        return FormatScore(fmt, -1.0, fs, float("nan"), 0, 0.0, 0.0, f"implausible fs {fs:.0f}")

    mets = [_channel_metrics(m[:, c].astype(np.float64), fs, duration_s) for c in range(m.shape[1])]
    scores = [_channel_score(x) for x in mets]
    score = float(np.mean(scores))
    notes = []
    if m.shape[1] >= 2 and not mets[0].get("flat") and not mets[1].get("flat"):
        r = float(np.corrcoef(m[:, 0].astype(float), m[:, 1].astype(float))[0, 1])
        notes.append(f"ch01_r={r:.3f}")
        if r > 0.995:          # channels identical -> almost certainly a split 1-ch stream
            score -= 1.0
    hp = _header_plausibility(payloads, fmt)
    score += hp
    m0 = mets[0]
    if m0.get("flat"):
        return FormatScore(fmt, -1.0, fs, float("nan"), 0, 0.0, 0.0, "flat")
    notes.append(f"rough={m0['roughness']:.2f} rr_cv={m0['rr_cv']:.2f} hdr={hp:+.2f}")
    return FormatScore(fmt, score, fs, m0["hr"], m0["n_peaks"], m0["band"], m0["qrs_snr"],
                       " ".join(notes))


def rank_formats(payloads: Sequence[bytes], duration_s: float,
                 candidates: Sequence[FrameFormat] | None = None,
                 sample_rate_hint: float | None = None) -> list[FormatScore]:
    cands = list(candidates or CANDIDATES)
    scores = [score_format(c, payloads, duration_s, sample_rate_hint) for c in cands]
    scores.sort(key=lambda s: -s.score)
    return scores


def fit_gain_offset(raw: np.ndarray, ref_mv: np.ndarray) -> tuple[float, float]:
    """Least-squares fit mV = (raw - offset) * gain against an aligned reference trace
    (e.g. the Kardia app export). Returns (gain_mv_per_count, offset_counts)."""
    raw = np.asarray(raw, dtype=np.float64)
    ref = np.asarray(ref_mv, dtype=np.float64)
    n = min(len(raw), len(ref))
    A = np.column_stack([raw[:n], np.ones(n)])
    (g, b), *_ = np.linalg.lstsq(A, ref[:n], rcond=None)
    if g == 0:
        return 0.0, 0.0
    return float(g), float(-b / g)


def with_calibration(fmt: FrameFormat, gain_mv: float, offset_counts: float,
                     sample_rate_hz: float) -> FrameFormat:
    return replace(fmt, gain_mv=gain_mv, offset_counts=offset_counts, sample_rate_hz=sample_rate_hz)
