import asyncio

import numpy as np

from kardia6l import dsp
from kardia6l.protocol import FrameFormat, encode
from kardia6l.stream import Pipeline, RingBuffer


class MemSink:
    def __init__(self):
        self.chunks = []
        self.names = None
    def open(self, fs, channels):
        self.names = channels
    def write(self, t, x):
        self.chunks.append((t, x))
    def close(self):
        self.closed = True
    def all(self):
        return np.concatenate([t for t, _ in self.chunks]), np.vstack([x for _, x in self.chunks])


def test_ringbuffer_wrap():
    rb = RingBuffer(10, 1)
    for i in range(0, 25, 5):
        rb.append(np.arange(i, i + 5, dtype=float), np.arange(i, i + 5, dtype=float)[:, None])
    t, x = rb.latest(7)
    np.testing.assert_array_equal(x[:, 0], np.arange(18, 25))
    np.testing.assert_array_equal(t, np.arange(18, 25))
    assert rb.n_written == 25


def test_pipeline_end_to_end_two_channel_derive():
    fmt = FrameFormat("h1_s16le_c2i", header_bytes=1, seq_offset=0, channels=2, gain_mv=0.0025)
    fs = 300.0
    e1 = dsp.synthetic_ecg(10, fs, 66)
    e2 = dsp.synthetic_ecg(10, fs, 66, seed=2, amplitude_mv=1.4)
    counts = np.round(np.column_stack([e1, e2]) / 0.0025).astype(int)
    pk = encode(counts, fmt, 30)
    del pk[10]  # simulate one lost notification
    sink = MemSink()

    async def go():
        pipe = Pipeline(fmt, fs, [sink], ["I", "II"], derive_leads=True)
        pipe.start()
        for i, p in enumerate(pk):
            pipe.on_notify("x", i * 0.1, p)
        await asyncio.sleep(0.2)
        await pipe.stop()
        return pipe

    pipe = asyncio.run(go())
    assert sink.names == ["I", "II", "III", "aVR", "aVL", "aVF"]
    t, x = sink.all()
    assert x.shape[1] == 6
    assert pipe.stats.seq_gaps == 1
    assert pipe.stats.samples == (len(pk)) * 30
    # timestamps skipped over the lost block
    assert np.isclose(t[-1], (len(pk) * 30 + 30 - 1) / fs)
    # III == II - I on the derived output
    np.testing.assert_allclose(x[:, 2], x[:, 1] - x[:, 0], atol=1e-9)
    assert abs(dsp.heart_rate(x[:, 0], fs) - 66) < 1.5
    assert sink.closed


def test_csv_wfdb_edf_roundtrip(tmp_path):
    from kardia6l.sinks.csv_sink import CSVSink, read_csv
    from kardia6l.sinks.wfdb_sink import EDFSink, WFDBSink
    import wfdb, pyedflib
    fs = 300.0
    x = np.column_stack([dsp.synthetic_ecg(5, fs, 60), dsp.synthetic_ecg(5, fs, 60, amplitude_mv=0.5)])
    t = np.arange(len(x)) / fs
    for sink in (CSVSink(tmp_path / "r.csv"), WFDBSink(tmp_path / "r"), EDFSink(tmp_path / "r.edf")):
        sink.open(fs, ["I", "II"])
        sink.write(t[:700], x[:700]); sink.write(t[700:], x[700:])
        sink.close()
    fs2, names, t2, x2 = read_csv(tmp_path / "r.csv")
    assert fs2 == fs and names == ["I", "II"]
    np.testing.assert_allclose(x2, x, atol=1e-4)
    r = wfdb.rdrecord(str(tmp_path / "r"))
    assert r.fs == 300 and r.sig_name == ["I", "II"]
    np.testing.assert_allclose(r.p_signal, x, atol=2e-3)
    f = pyedflib.EdfReader(str(tmp_path / "r.edf"))
    assert f.getSignalLabels() == ["I", "II"]
    e0 = f.readSignal(0)
    f.close()
    np.testing.assert_allclose(e0[: len(x)], x[:, 0], atol=2e-3)


def test_validate_compare_pass_and_fail():
    from validate.compare import compare
    fs = 300.0
    ref = dsp.synthetic_ecg(30, fs, 71, noise_uv=15.0, seed=1)
    tool = np.roll(ref, 45) * 1.8 + 0.1  # shifted 150 ms, different gain/offset
    r = compare(tool, fs, ref, fs)
    assert r["pass_corr"] and r["pass_hr"], r
    assert abs(r["lag_s"] - 0.15) < 0.01
    bad = dsp.synthetic_ecg(30, fs, 95, noise_uv=15.0)
    r2 = compare(bad, fs, ref, fs)
    assert not (r2["pass_corr"] and r2["pass_hr"])
