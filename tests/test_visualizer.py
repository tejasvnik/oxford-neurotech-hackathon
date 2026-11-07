import numpy as np
import pytest

from eeg_pipeline import PipelineConfig, synthetic_eeg
from scripts.visualize import Monitor


def test_monitor_uses_pipeline_filtered_data_and_alpha():
    cfg = PipelineConfig()
    rec = synthetic_eeg(5, cfg)
    monitor = Monitor(cfg)
    monitor.push(rec.data)
    assert monitor.frame["t"] == 5
    assert monitor.history[-1][1] is not None
    assert monitor.pipeline.buf.shape == (cfg.window_n, cfg.n_channels)


def test_monitor_suppresses_posterior_fallback_and_rejects_nonfinite():
    cfg = PipelineConfig()
    rec = synthetic_eeg(5, cfg)
    rec.data[:, 4:] = 0
    monitor = Monitor(cfg)
    monitor.push(rec.data)
    assert monitor.frame["summary"]["posterior_alpha_rel"] is not None
    assert monitor.history[-1][1] is None
    with pytest.raises(ValueError, match="finite samples"):
        monitor.push(np.full((1, 8), np.nan))
