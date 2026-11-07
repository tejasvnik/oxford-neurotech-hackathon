"""Show the existing EEG pipeline in a local Matplotlib window.

    python scripts/visualize.py --synthetic
    python scripts/visualize.py --replay UnicornRecorder_07_11_2026_17_30_360.csv
    python scripts/visualize.py --lsl
"""
import argparse
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
import sys
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_pipeline import EEGPipeline, PipelineConfig  # noqa: E402
from scripts.live import source  # noqa: E402


class Monitor:
    """Keep the existing pipeline's filtered window and recent alpha values."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = EEGPipeline(cfg)
        self.frame = None
        self.history = deque(maxlen=120)

    def push(self, chunk):
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim != 2 or chunk.shape[1] != self.cfg.n_channels or not np.isfinite(chunk).all():
            raise ValueError(f"Expected finite samples with {self.cfg.n_channels} EEG channels")
        for frame in self.pipeline.push(chunk):
            posterior_good = any(
                name.upper().startswith(("P", "O")) and quality == "good"
                for name, quality in zip(frame["channels"], frame["quality"])
            )
            # The pipeline can fall back to other channels; don't label that posterior alpha.
            alpha = frame["summary"]["posterior_alpha_rel"] if posterior_good else None
            self.history.append((frame["t"], alpha))
            self.frame = frame


def read_source(args, cfg, events, stop):
    def send(event):
        while not stop.is_set():
            try:
                events.put(event, timeout=0.2)
                return
            except Full:
                continue
    try:
        for chunk in source(args, cfg):
            if stop.is_set():
                return
            send(("chunk", chunk))
        send(("finished", None))
    except Exception as exc:
        send(("error", str(exc)))


def run(args):
    cfg = PipelineConfig()
    monitor = Monitor(cfg)
    events, stop = Queue(maxsize=256), threading.Event()
    threading.Thread(target=read_source, args=(args, cfg, events, stop), daemon=True).start()
    input_name = "Synthetic demo" if args.synthetic else ("Live LSL" if args.lsl else f"Recording: {Path(args.replay).name}")

    fig, (traces, alpha_ax) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1.3]})
    fig.canvas.manager.set_window_title("EEG pipeline visualiser")
    fig.suptitle("EEG pipeline visualiser", fontsize=17, fontweight="bold", x=0.08, ha="left")
    status_text = fig.text(0.08, 0.91, f"{input_name}  |  Waiting for samples", fontsize=10)
    detail_text = fig.text(0.08, 0.08, "Quality labels are basic amplitude checks, not proof of clean EEG.", fontsize=9, color="#536a60")
    fig.text(0.08, 0.04, "Synthetic alpha bursts: 20–40 s and 80–100 s. Close this window to stop.", fontsize=9, color="#536a60")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.15, hspace=0.55)

    lines = [traces.plot([], [], lw=0.9, color="#127d6b")[0] for _ in cfg.channel_names]
    traces.set_title("Filtered EEG · latest 2 seconds", loc="left", fontsize=12, fontweight="bold")
    traces.set_xlim(-cfg.window_s, 0)
    traces.set_ylim(cfg.n_channels - 0.5, -0.5)
    traces.set_yticks(range(cfg.n_channels), cfg.channel_names)
    traces.set_xlabel("Seconds from now")
    traces.grid(axis="y", color="#e4ebe6")
    alpha_line, = alpha_ax.plot([], [], color="#127d6b", lw=2)
    alpha_ax.set_title("Relative posterior alpha · good posterior channels only", loc="left", fontsize=12, fontweight="bold")
    alpha_ax.set_ylim(0, 100)
    alpha_ax.set_xlim(0, 60)
    alpha_ax.set_ylabel("Power (%)")
    alpha_ax.set_xlabel("Pipeline time (seconds)")
    alpha_ax.grid(color="#e4ebe6")
    stream_status = "Waiting for samples"
    last_received = None

    def update(_):
        nonlocal stream_status, last_received
        changed = False
        for _ in range(100):
            try:
                kind, value = events.get_nowait()
            except Empty:
                break
            changed = True
            if kind == "chunk":
                try:
                    monitor.push(value)
                    stream_status = "Receiving samples"
                    last_received = time.monotonic()
                except ValueError as exc:
                    stream_status = f"Input error: {exc}"
                    stop.set()
                    break
            elif kind == "finished":
                stream_status = "Input finished"
            else:
                stream_status = f"Input error: {value}"
        if stream_status == "Receiving samples" and last_received is not None and time.monotonic() - last_received > 3:
            stream_status = "Stream stalled: no recent samples"
            changed = True
        if not changed:
            return
        n = monitor.pipeline.filled
        if n:
            window = monitor.pipeline.buf[-n:]
            scale = max(1.0, float(np.percentile(np.abs(window), 98)))
            x = (np.arange(n) - (n - 1)) / cfg.fs
            for i, line in enumerate(lines):
                line.set_data(x, i - np.clip(window[:, i] / scale, -1, 1) * 0.4)
            traces.set_title(f"Filtered EEG · latest 2 seconds · shared scale ±{scale:.1f} µV (clipped)", loc="left", fontsize=11, fontweight="bold")
        frame = monitor.frame
        elapsed = monitor.pipeline.n_seen / cfg.fs
        status_text.set_text(f"{input_name}  |  {stream_status}  |  {elapsed:.1f} s")
        if frame:
            quality = frame["quality"]
            traces.set_yticklabels([f"{name}  ·  {q}" for name, q in zip(cfg.channel_names, quality)])
            for label, q in zip(traces.get_yticklabels(), quality):
                label.set_color("#157451" if q == "good" else "#b06724")
            detail_text.set_text(f"{frame['summary']['n_good']}/{cfg.n_channels} channels pass basic checks. 'Good' can still include blinks and other artefacts.")
        if monitor.history:
            t = [item[0] for item in monitor.history]
            a = [np.nan if item[1] is None else item[1] * 100 for item in monitor.history]
            alpha_line.set_data(t, a)
            end = max(60, t[-1])
            alpha_ax.set_xlim(max(0, end - 120), end)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("close_event", lambda event: stop.set())
    animation = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.show()
    stop.set()
    return animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--replay", metavar="CSV")
    group.add_argument("--lsl", action="store_true")
    parser.add_argument("--lsl-name", default=None)
    parser.add_argument("--speed", type=float, default=1.0, help="synthetic/replay speed multiplier")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be a positive finite number")
    run(args)


if __name__ == "__main__":
    main()
