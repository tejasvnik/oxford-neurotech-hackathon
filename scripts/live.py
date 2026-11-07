"""Live feature stream: EEG -> pipeline -> JSON lines (stdout) and/or WebSocket.

On the headset laptop (stream from Unicorn via LSL):
    python scripts/live.py --lsl
Anywhere else, without hardware (replays a recording at real-time speed):
    python scripts/live.py --replay data/raw/2026-11-07/UnicornRecorder_07_11_2026_17_24_280.csv
    python scripts/live.py --synthetic
Add --ws 8765 to broadcast each feature frame to ws://<this-ip>:8765 for the app.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_pipeline import EEGPipeline, PipelineConfig, load_unicorn_csv, synthetic_eeg
from eeg_pipeline.io import LSLInlet, replay


def source(a, cfg):
    if a.lsl:
        inlet = LSLInlet(name=a.lsl_name)
        for chunk, ts in inlet.chunks():
            yield chunk[:, :cfg.n_channels]  # Unicorn LSL also sends accel/gyro etc.
        return
    rec = synthetic_eeg(300, cfg, alpha_blocks=((20, 40), (80, 100))) if a.synthetic \
        else load_unicorn_csv(a.replay, cfg)
    chunk = 25
    for x, _ in replay(rec, chunk):
        yield x
        time.sleep(chunk / cfg.fs / a.speed)


async def main_async(a):
    cfg = PipelineConfig()
    pipe = EEGPipeline(cfg)
    clients = set()
    if a.ws:
        import websockets

        async def handler(ws):
            clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                clients.discard(ws)
        await websockets.serve(handler, "0.0.0.0", a.ws)
        print(f"WebSocket on ws://0.0.0.0:{a.ws}", file=sys.stderr)
    loop = asyncio.get_running_loop()
    it = source(a, cfg)
    while True:
        chunk = await loop.run_in_executor(None, next, it, None)
        if chunk is None:
            break
        for feat in pipe.push(chunk):
            msg = json.dumps(feat)
            if not a.quiet:
                s = feat["summary"]
                print(f"t={feat['t']:6.1f}s  good={s['n_good']}/8  "
                      f"postα={s['posterior_alpha_rel']}  q={feat['quality']}")
            for c in list(clients):
                await c.send(msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lsl", action="store_true")
    g.add_argument("--replay")
    g.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lsl-name", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--ws", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    asyncio.run(main_async(ap.parse_args()))
