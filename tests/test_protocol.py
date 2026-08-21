"""Protocol tests.

* Round-trip: encode -> Decoder for every candidate layout (proves the decoder
  handles header/endian/sign/width/channels/interleave correctly).
* Sequence-gap accounting.
* Scorer: given a synthetic ECG packed with one layout, rank_formats must rank the
  true layout (or an equivalent one) first, and the entropy report must say
  'plaintext-like'; random bytes must read 'ciphertext-like'.
* GOLDEN VECTOR against the real device: xfail/skip until Phase 1 lands
  (see tools/capture_decode_notes.md).
"""
import numpy as np
import pytest

from open_kardia import dsp, protocol
from open_kardia.protocol import CANDIDATES, Decoder, FrameFormat, encode


@pytest.mark.parametrize("fmt", CANDIDATES, ids=lambda f: f.name)
def test_roundtrip_all_candidates(fmt):
    rng = np.random.default_rng(0)
    bits = 8 * fmt.sample_bytes
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if fmt.signed else (0, (1 << bits) - 1)
    s = rng.integers(lo, hi + 1, size=(97, fmt.channels))
    pkts = encode(s, fmt, per_notification=10)
    dec = Decoder(fmt, to_mv=False)
    out = dec.decode_many(pkts)
    np.testing.assert_array_equal(out, s)
    assert dec.stats.samples == 97
    assert dec.stats.gaps == 0


def test_sequence_gap_detection():
    fmt = FrameFormat("h2", header_bytes=2, seq_offset=0, seq_bytes=1)
    s = np.arange(50)[:, None]
    pkts = encode(s, fmt, per_notification=5)
    del pkts[3]  # drop one notification
    dec = Decoder(fmt, to_mv=False)
    dec.decode_many(pkts)
    assert dec.stats.gaps == 1
    assert dec.stats.lost_frames_est == 1
    assert dec.stats.samples == 45


def test_seq_wraparound_no_false_gap():
    fmt = FrameFormat("h1", header_bytes=1, seq_offset=0, seq_bytes=1)
    s = np.zeros((300 * 2, 1), dtype=int)
    pkts = encode(s, fmt, per_notification=2, start_seq=250)  # wraps 255->0
    dec = Decoder(fmt, to_mv=False)
    dec.decode_many(pkts)
    assert dec.stats.gaps == 0


def test_mv_calibration():
    fmt = FrameFormat("cal", gain_mv=0.001, offset_counts=100)
    d = Decoder(fmt)
    out = d.decode(encode(np.array([[100], [1100]]), fmt, 2)[0])
    np.testing.assert_allclose(out[:, 0], [0.0, 1.0])


def _synthetic_stream(fmt: FrameFormat, fs=300.0, secs=20.0, hr=72.0, per_notif=25):
    ecg = dsp.synthetic_ecg(secs, fs, hr_bpm=hr, noise_uv=25.0)
    counts = np.round(ecg / 0.0025).astype(int)         # 2.5 µV/LSB, ~±400 counts
    if not fmt.signed:
        counts = counts + (1 << (8 * fmt.sample_bytes - 1))
    if fmt.channels == 2:
        lead2 = dsp.synthetic_ecg(secs, fs, hr_bpm=hr, noise_uv=25.0, seed=1, amplitude_mv=1.3)
        c2 = np.round(lead2 / 0.0025).astype(int)
        if not fmt.signed:
            c2 = c2 + (1 << (8 * fmt.sample_bytes - 1))
        counts = np.column_stack([counts, c2])
    return encode(counts, fmt, per_notif), secs


@pytest.mark.parametrize("truth", [
    FrameFormat("h0_s16le_c1i"),
    FrameFormat("h2_s16le_c1i", header_bytes=2, seq_offset=0),
    FrameFormat("h0_s16be_c1i", little_endian=False),
    FrameFormat("h1_u16le_c1i", header_bytes=1, signed=False),
    FrameFormat("h0_s24le_c1i", sample_bytes=3),
    FrameFormat("h0_s16le_c2i", channels=2, interleaved=True),
    FrameFormat("h0_s16le_c2b", channels=2, interleaved=False),
], ids=lambda f: f.name)
def test_scorer_recovers_true_layout(truth):
    pkts, secs = _synthetic_stream(truth)
    ranked = protocol.rank_formats(pkts, secs)
    best = ranked[0]
    # The winner must decode channel 0 to a signal essentially identical to the truth's
    # channel 0 (some layouts are aliases of each other for 1-channel data).
    truth_x = Decoder(truth, to_mv=False).decode_many(pkts)[:, 0].astype(float)
    best_x = Decoder(best.fmt, to_mv=False).decode_many(pkts)[:, 0].astype(float)
    n = min(len(truth_x), len(best_x))
    r = np.corrcoef(truth_x[:n], best_x[:n])[0, 1]
    assert r > 0.98, f"best={best.fmt.name} score={best.score:.2f} r={r:.3f} truth={truth.name}"
    assert 70 <= best.hr_bpm <= 74
    assert abs(best.sample_rate_est - 300) < 5


def test_entropy_verdicts():
    pkts, _ = _synthetic_stream(FrameFormat("h0_s16le_c1i"))
    rep = protocol.entropy_report(pkts)
    assert rep.verdict == "plaintext-like", rep
    rng = np.random.default_rng(3)
    rnd = [rng.integers(0, 256, 40, dtype=np.uint8).tobytes() for _ in range(300)]
    rep2 = protocol.entropy_report(rnd)
    assert rep2.verdict == "ciphertext-like", rep2


def test_fit_gain_offset():
    raw = np.arange(1000, dtype=float) + 500
    ref = (raw - 500) * 0.002
    g, o = protocol.fit_gain_offset(raw, ref)
    assert g == pytest.approx(0.002)
    assert o == pytest.approx(500)


@pytest.mark.skipif(protocol.KNOWN_FORMAT is None,
                    reason="Phase 1 not complete: real 6L wire format not yet established")
def test_golden_vector_real_device():
    """Decode a frozen slice of the authoritative iOS-app capture with KNOWN_FORMAT and
    assert the first samples of Lead I and Lead II (raw counts). Locks the wire format."""
    import json, pathlib
    g = json.loads((pathlib.Path(__file__).parent / "golden_6l_notifications.json").read_text())
    pk = [bytes.fromhex(h) for h in g["payloads_hex"]]
    fmt = protocol.KNOWN_FORMAT
    assert fmt.name == "open_kardia_6l" and fmt.channels == 2 and fmt.sample_bytes == 2
    assert fmt.little_endian and fmt.signed and fmt.interleaved and fmt.sample_rate_hz == 300.0
    raw = protocol.Decoder(fmt, to_mv=False).decode_many(pk)
    assert raw.shape == (54, 2)                     # 6 notifications x 9 samples x 2 ch
    # First 4 interleaved samples decoded from payload[0] = c0fe2802 9cfe0002 3cfefc01 b4fe5c02
    np.testing.assert_array_equal(raw[0], [-320, 552])
    np.testing.assert_array_equal(raw[1], [-356, 512])
    np.testing.assert_array_equal(raw[2], [-452, 508])
    np.testing.assert_array_equal(raw[3], [-332, 604])
    # Decoded as calibrated mV, both leads carry a physiologic-amplitude ECG.
    mv = protocol.Decoder(fmt, to_mv=True).decode_many(pk)
    assert mv.shape == (54, 2)
    assert abs(mv[0, 0] - (-320 * fmt.gain_mv)) < 1e-9
