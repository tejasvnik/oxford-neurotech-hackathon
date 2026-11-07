"""Input adapters: Unicorn CSV recordings, synthetic EEG, and a live LSL stream."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig


@dataclass
class Recording:
    data: np.ndarray        # (n_samples, n_channels), microvolts
    fs: float
    channel_names: list
    trigger: np.ndarray     # (n_samples,)
    source: str = ""

    @property
    def duration_s(self) -> float:
        return self.data.shape[0] / self.fs

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.data.shape[0]) / self.fs


def load_unicorn_csv(path, cfg: PipelineConfig | None = None) -> Recording:
    """Load a Unicorn Recorder CSV export (columns: EEG 1..8, TRIG).

    Values are assumed to be microvolts (Unicorn Recorder default). Note: the
    .bdf exports in this repo have a broken physical-range header, so prefer CSV.
    """
    cfg = cfg or PipelineConfig()
    path = Path(path)
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    eeg_cols = [c for c in df.columns if c.upper().startswith("EEG")]
    if len(eeg_cols) != cfg.n_channels:
        raise ValueError(f"{path.name}: expected {cfg.n_channels} EEG columns, found {len(eeg_cols)}")
    if "TRIG" in [c.upper() for c in df.columns]:
        trig = df[[c for c in df.columns if c.upper() == "TRIG"][0]].to_numpy(float)
    elif "marker" in df.columns:  # stream_unicorn.py format: text markers -> 1 at marker samples
        trig = df["marker"].notna().to_numpy(float)
    else:
        trig = np.zeros(len(df))
    return Recording(df[eeg_cols].to_numpy(float), cfg.fs, list(cfg.channel_names), trig, str(path))


def synthetic_eeg(duration_s=60.0, cfg: PipelineConfig | None = None, seed=0,
                  alpha_blocks=((20, 40),), alpha_uv=15.0, noise_uv=8.0,
                  line_uv=20.0, blink_times=()) -> Recording:
    """Realistic-ish fake EEG: 1/f background + alpha bursts (eyes closed) in posterior
    channels + 50 Hz mains + optional frontal blinks. Ground truth for tests/demos."""
    cfg = cfg or PipelineConfig()
    rng = np.random.default_rng(seed)
    n = int(duration_s * cfg.fs)
    t = np.arange(n) / cfg.fs
    # pink-ish noise via 1/f shaping
    white = rng.standard_normal((cfg.n_channels, n))
    spec = np.fft.rfft(white, axis=1)
    f = np.fft.rfftfreq(n, 1 / cfg.fs)
    spec /= np.sqrt(np.maximum(f, 1.0))
    bg = np.fft.irfft(spec, n=n, axis=1)
    bg = bg / bg.std(axis=1, keepdims=True) * noise_uv
    x = bg.T.copy()
    posterior = [i for i, c in enumerate(cfg.channel_names) if c.upper().startswith(("O", "PO", "P"))]
    for (a, b) in alpha_blocks:
        m = (t >= a) & (t < b)
        for ch in posterior:
            x[m, ch] += alpha_uv * np.sin(2 * np.pi * 10.0 * t[m] + rng.uniform(0, 2 * np.pi))
    x += line_uv * np.sin(2 * np.pi * 50.0 * t)[:, None]
    for bt in blink_times:
        x[:, 0] += 150 * np.exp(-0.5 * ((t - bt) / 0.08) ** 2)
    x += rng.normal(0, 200, cfg.n_channels)  # DC offsets like a real amp
    return Recording(x, cfg.fs, list(cfg.channel_names), np.zeros(n), "synthetic")


class LSLInlet:
    """Live adapter: pulls chunks from a Lab Streaming Layer stream.

    On the acquisition laptop, start the g.tec Unicorn LSL app (or UnicornLSL),
    then on any machine on the same network:  `for chunk in LSLInlet().chunks(): ...`
    """

    def __init__(self, stream_type="EEG", name=None, timeout=10.0):
        from pylsl import StreamInlet, resolve_byprop  # optional dependency
        prop, val = ("name", name) if name else ("type", stream_type)
        streams = resolve_byprop(prop, val, timeout=timeout)
        if not streams:
            raise RuntimeError(f"No LSL stream with {prop}={val!r} found")
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.fs = info.nominal_srate()
        self.n_channels = info.channel_count()

    def chunks(self, max_samples=64):
        while True:
            data, ts = self.inlet.pull_chunk(timeout=1.0, max_samples=max_samples)
            if data:
                yield np.asarray(data, float), np.asarray(ts)


def replay(rec: Recording, chunk=25):
    """Simulate a live stream from a recording (same interface as LSLInlet.chunks)."""
    for i in range(0, rec.data.shape[0], chunk):
        yield rec.data[i:i + chunk], rec.times[i:i + chunk]
