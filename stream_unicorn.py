#!/usr/bin/env python3
"""stream_unicorn.py: ONE file to stream + record the g.tec Unicorn.

  * finds the Unicorn: Bluetooth serial (macOS/Linux, no g.tec software), the "Unicorn LSL" app, or UnicornPy (Windows)
  * records EEG + any experiment markers into ONE CSV:  recordings/unicorn_<time>.csv
        columns: lsl_time, EEG 1..8, marker
  * prints live signal quality every second
  * serves band-power features on ws://<this-laptop>:8765 for dashboard/index.html and mindcart.html
  * serves raw EEG on ws://<this-laptop>:8766 for dashboard/pipeline.html (live step-by-step pipeline board)

Install once:   pip install numpy scipy pylsl websockets pyserial
Mac:            pair "UN-xxxx" in System Settings > Bluetooth, then just run the commands below
Test device:    python stream_unicorn.py --check         (~60 s guided test -> PASS/FAIL)
Run:            python stream_unicorn.py                 (auto: Bluetooth serial -> LSL -> UnicornPy)
                python stream_unicorn.py --source unicornpy
                python stream_unicorn.py --fake          (no headset, for testing)
Stop:           Ctrl+C  (file is saved continuously)
"""
import argparse
import asyncio
import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos, welch

FS = 250.0
N_EEG = 8
CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
WIN, STEP = int(2 * FS), int(1 * FS)


# ----------------------------------------------------------------- sources
class LSLSource:
    def __init__(self, timeout=8):
        from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_streams
        streams = resolve_streams(wait_time=timeout)
        eeg = [s for s in streams if s.channel_count() >= N_EEG and s.nominal_srate() > 0 and
               ("unicorn" in s.name().lower() or s.type().lower() in ("eeg", "data"))]
        if not eeg:
            names = [f"{s.name()} ({s.type()}, {s.channel_count()} ch)" for s in streams]
            raise RuntimeError("No Unicorn LSL stream. Open the 'Unicorn LSL' app, pick the headset, "
                               f"click Open then Start. Streams seen: {names or 'none'}")
        s = eeg[0]
        print(f"[source] LSL: {s.name()}  type={s.type()}  {s.channel_count()} ch @ {s.nominal_srate():.0f} Hz")
        self.inlet = StreamInlet(s, max_buflen=30, processing_flags=proc_clocksync | proc_dejitter)
        self.markers = None
        mk = [m for m in streams if m.type().lower() == "markers"]
        if mk:
            print(f"[source] markers: {mk[0].name()}")
            self.markers = StreamInlet(mk[0], processing_flags=proc_clocksync | proc_dejitter)
        self._find_markers_later = not mk

    def read(self):
        data, ts = self.inlet.pull_chunk(timeout=0.1, max_samples=256)
        mks = []
        if self._find_markers_later and int(time.time()) % 5 == 0:   # experiment may start later
            from pylsl import StreamInlet, resolve_byprop
            m = resolve_byprop("type", "Markers", timeout=0.05)
            if m:
                print(f"[source] markers stream appeared: {m[0].name()}")
                self.markers = StreamInlet(m[0]); self._find_markers_later = False
        if self.markers:
            md, mt = self.markers.pull_chunk(timeout=0.0)
            mks = [(t, str(d[0])) for d, t in zip(md, mt)]
        return (np.asarray(data, float)[:, :N_EEG] if data else np.empty((0, N_EEG))), np.asarray(ts), mks


class UnicornPySource:
    """Direct USB-dongle access via g.tec's UnicornPy (Windows, ships with Unicorn Suite)."""
    def __init__(self):
        import UnicornPy
        devs = UnicornPy.GetAvailableDevices(True)
        if not devs:
            raise RuntimeError("No Unicorn paired. Pair it in Unicorn Suite first.")
        print(f"[source] UnicornPy: {devs[0]}")
        self.dev = UnicornPy.Unicorn(devs[0])
        self.n_ch = self.dev.GetNumberOfAcquiredChannels()
        self.n = 25
        self.buf = bytearray(self.n * self.n_ch * 4)
        self.dev.StartAcquisition(False)
        self.t0 = time.time()
        self.count = 0

    def read(self):
        self.dev.GetData(self.n, self.buf, len(self.buf))
        x = np.frombuffer(self.buf, dtype=np.float32).reshape(self.n, self.n_ch)[:, :N_EEG].astype(float)
        ts = self.t0 + (self.count + np.arange(self.n)) / FS
        self.count += self.n
        return x, ts, []

    def close(self):
        self.dev.StopAcquisition()


class SerialSource:
    """macOS/Linux: read the Unicorn directly over its Bluetooth serial port (no g.tec software needed).

    Pair "UN-xxxx.xx.xx" in System Settings > Bluetooth first. It shows up as /dev/cu.UN-*.
    Frame = 45 bytes: C0 00 | battery | 8x EEG int24 BE | 3x acc int16 LE | 3x gyro int16 LE | counter u32 LE | 0D 0A
    Also republishes everything as an LSL stream "Unicorn" (type EEG) so other tools can use it.
    """
    START, STOP = bytes([0x61, 0x7C, 0x87]), bytes([0x63, 0x5C, 0xC5])
    FRAME = 45
    UV = 4500000.0 / 50331642.0

    def __init__(self, port=None, lsl_out=True):
        import glob
        import serial  # pip install pyserial
        if port is None:
            ports = sorted(glob.glob("/dev/cu.UN-*") + glob.glob("/dev/tty.UN-*") + glob.glob("/dev/rfcomm*"))
            if not ports:
                raise RuntimeError("No Unicorn serial port (/dev/cu.UN-*). Pair the headset in Bluetooth settings first.")
            port = ports[0]
        print(f"[source] Bluetooth serial: {port}  (connecting… the LED should stop blinking)")
        self.ser = serial.Serial(port, 115200, timeout=1)
        self.ser.reset_input_buffer()
        self.ser.write(self.START)
        self.buf = bytearray()
        self.last_counter = None
        self.dropped = 0
        self.outlet = None
        if lsl_out:
            try:
                from pylsl import StreamInfo, StreamOutlet
                info = StreamInfo("Unicorn", "EEG", N_EEG, FS, "float32", "unicorn_serial")
                chs = info.desc().append_child("channels")
                for c in CH:
                    chs.append_child("channel").append_child_value("label", c).append_child_value("unit", "microvolts")
                self.outlet = StreamOutlet(info)
                print("[source] republishing as LSL stream 'Unicorn' (type EEG)")
            except Exception as e:
                print(f"[source] (LSL republish off: {e})")

    @classmethod
    def parse(cls, fr):
        eeg = []
        for i in range(8):
            b = fr[3 + 3 * i: 6 + 3 * i]
            v = (b[0] << 16) | (b[1] << 8) | b[2]
            if v & 0x800000:
                v -= 1 << 24
            eeg.append(v * cls.UV)
        counter = int.from_bytes(fr[39:43], "little")
        battery = (fr[2] & 0x0F) * 100.0 / 15.0
        return eeg, counter, battery

    def read(self):
        from pylsl import local_clock
        self.buf += self.ser.read(max(self.FRAME, self.ser.in_waiting or 0))
        rows = []
        while len(self.buf) >= self.FRAME:
            if self.buf[0] != 0xC0 or self.buf[1] != 0x00 or self.buf[43] != 0x0D or self.buf[44] != 0x0A:
                j = self.buf.find(b"\xC0\x00", 1)          # resync
                self.buf = self.buf[j:] if j > 0 else bytearray()
                continue
            eeg, counter, self.battery = self.parse(bytes(self.buf[:self.FRAME]))
            del self.buf[:self.FRAME]
            if self.last_counter is not None and counter != self.last_counter + 1:
                self.dropped += max(0, counter - self.last_counter - 1)
            self.last_counter = counter
            rows.append(eeg)
        if not rows:
            return np.empty((0, N_EEG)), np.empty(0), []
        now = local_clock()
        ts = now - (len(rows) - 1 - np.arange(len(rows))) / FS
        x = np.asarray(rows)
        if self.outlet:
            self.outlet.push_chunk(x.tolist(), now)
        return x, ts, []

    def close(self):
        try:
            self.ser.write(self.STOP); self.ser.close()
        except Exception:
            pass
        print(f"\n[source] stopped. dropped samples: {self.dropped}")


class FakeSource:
    def __init__(self, check=False):
        self.check = check
        print("[source] FAKE synthetic EEG (eyes closed every other 20 s)")
        self.k = 0; self.t0 = time.time(); self.rng = np.random.default_rng(0)

    def read(self):
        time.sleep(0.1)
        n = int((time.time() - self.t0) * FS) - self.k
        t = (self.k + np.arange(n)) / FS; self.k += n
        x = self.rng.normal(0, 8, (n, N_EEG))
        closed = ((t // 20) % 2 == 1) if not self.check else ((t > 25) & (t < 45))
        x[:, 4:] += (6 + 10 * closed[:, None]) * np.sin(2 * np.pi * 10 * t)[:, None]
        if self.check:  # blinks every 2 s in the last phase
            x[:, 0] += 150 * np.exp(-0.5 * (((t - 46) % 2 - 1) / 0.08) ** 2) * (t > 46)
        return x + 3000, self.t0 + t, []


# ----------------------------------------------------------------- processing
class Processor:
    def __init__(self):
        sos = butter(4, [1, 40], btype="band", fs=FS, output="sos")
        b, a = iirnotch(50, 30, fs=FS)
        self.sos = np.vstack([sos, tf2sos(b, a)])
        self.zi = None
        self.buf = np.zeros((WIN, N_EEG)); self.seen = 0; self.since = 0

    def push(self, x):
        if not len(x):
            return []
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        out = []
        for row in y:
            self.buf = np.roll(self.buf, -1, 0); self.buf[-1] = row
            self.seen += 1; self.since += 1
            if self.seen >= 2 * WIN and self.since >= STEP:
                self.since = 0; out.append(self.features())
        return out

    def features(self):
        w = self.buf
        f, p = welch(w.T, fs=FS, nperseg=int(FS), noverlap=int(FS) // 2)
        df = f[1] - f[0]
        tot = p[:, (f >= 1) & (f < 30)].sum(1) * df
        feat = {"t": round(self.seen / FS, 2), "channels": CH}
        for b, (lo, hi) in BANDS.items():
            bp = p[:, (f >= lo) & (f < hi)].sum(1) * df
            feat[b] = [round(float(v), 4) for v in bp]
            feat[b + "_rel"] = [round(float(v), 4) for v in bp / np.maximum(tot, 1e-12)]
        sd, ptp = w.std(0), np.ptp(w, 0)
        q = ["flat" if s < 0.5 else "noisy" if s > 100 else "artefact" if pp > 500 else "good" for s, pp in zip(sd, ptp)]
        feat["quality"] = q
        feat["std_uv"] = [round(float(s), 1) for s in sd]
        good = np.array([v == "good" for v in q]); post = np.array([c[0] in "PO" for c in CH])
        sel = good & post if (good & post).any() else good
        a, b_ = np.array(feat["alpha"]), np.array(feat["beta"])
        feat["summary"] = {"n_good": int(good.sum()),
                           "posterior_alpha_rel": float(np.mean(np.array(feat["alpha_rel"])[sel])) if sel.any() else None,
                           "beta_alpha_ratio": float(np.mean(b_[good] / np.maximum(a[good], 1e-12))) if good.any() else None}
        return feat


# ----------------------------------------------------------------- device check
def device_check(src, out: Path):
    """Guided ~60 s test: signal amplitude, eyes-closed alpha, blink detection. Prints PASS/FAIL."""
    from scipy.signal import sosfiltfilt

    def collect(seconds, label):
        print(f"\n>>> {label}  ({seconds} s)", flush=True)
        xs, t_end = [], time.time() + seconds
        while time.time() < t_end:
            x, _, _ = src.read()
            if len(x):
                xs.append(x)
            left = int(t_end - time.time())
            print(f"\r    {left:2d} s left", end="", flush=True)
        print()
        return np.vstack(xs) if xs else np.empty((0, N_EEG))

    print("\n=== UNICORN DEVICE CHECK ===  sit still, relaxed, look at the screen")
    collect(5, "Settling, sit still")
    opn = collect(20, "EYES OPEN, look at a point on the screen, don't blink much")
    cls = collect(20, "EYES CLOSED, relax and keep them closed until you hear the beep")
    print("\a", end="")
    blk = collect(10, "EYES OPEN, now BLINK FIRMLY 5 times, about 2 s apart")

    sos = butter(4, [1, 40], btype="band", fs=FS, output="sos")
    b, a = iirnotch(50, 30, fs=FS); sos = np.vstack([sos, tf2sos(b, a)])
    filt = lambda x: sosfiltfilt(sos, x - np.median(x, 0), axis=0)[int(FS):-int(FS)]
    yo, yc, yb = filt(opn), filt(cls), filt(blk)

    def alpha(y):
        f, p = welch(y.T, fs=FS, nperseg=int(2 * FS))
        return p[:, (f >= 8) & (f < 13)].mean(1)

    sd = np.median(yo[:len(yo) // int(FS) * int(FS)].reshape(-1, int(FS), N_EEG).std(1), 0)
    ratio = alpha(yc) / np.maximum(alpha(yo), 1e-12)
    post = [i for i, c in enumerate(CH) if c[0] in "PO"]
    fz = yb[:, 0]
    thr = max(80.0, 5 * np.median(np.abs(fz)))
    above = np.flatnonzero(fz > thr)
    blinks = int((np.diff(above) > FS * 0.4).sum() + (1 if len(above) else 0))

    ok_ch = [(2 <= s <= 60) for s in sd]
    alpha_ok = np.median(ratio[post]) >= 1.3
    blink_ok = 3 <= blinks <= 8
    print("\n=== RESULTS ===")
    print(f"{'ch':4s} {'σ µV (eyes open)':>17s} {'α closed/open':>14s}  status")
    for i, c in enumerate(CH):
        print(f"{c:4s} {sd[i]:17.1f} {ratio[i]:14.2f}  {'OK' if ok_ch[i] else ('FLAT' if sd[i] < 2 else 'NOISY: re-seat/wet electrode')}")
    print(f"\nSignal amplitude : {sum(ok_ch)}/8 channels in 2-60 µV          {'PASS' if sum(ok_ch) >= 7 else 'FAIL'}")
    print(f"Eyes-closed alpha: posterior ×{np.median(ratio[post]):.2f} (need ≥1.3)        {'PASS' if alpha_ok else 'FAIL'}")
    print(f"Blink detection  : {blinks} blinks seen on Fz (asked for 5)     {'PASS' if blink_ok else 'CHECK'}")
    verdict = sum(ok_ch) >= 7 and alpha_ok
    print(f"\nOVERALL: {'✅ READY TO RECORD' if verdict else '❌ NOT READY: fix the channels marked NOISY/FLAT, then rerun'}")
    path = out / f"device_check_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["phase"] + [f"EEG {i + 1}" for i in range(N_EEG)])
        for ph, x in (("open", opn), ("closed", cls), ("blink", blk)):
            for row in x:
                w.writerow([ph] + [f"{v:.3f}" for v in row])
    print(f"Raw check data saved: {path.resolve()}")
    return verdict


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "lsl", "serial", "unicornpy"], default="auto")
    ap.add_argument("--serial", dest="serial_port", default=None, help="serial port for --source serial (default: /dev/cu.UN-*)")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--name", default=None, help="file name (default unicorn_<date_time>.csv)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-ws", action="store_true")
    ap.add_argument("--check", action="store_true", help="run the ~60 s guided device test and exit")
    a = ap.parse_args()

    if a.fake:
        src = FakeSource(check=a.check)
    elif a.source == "unicornpy":
        src = UnicornPySource()
    elif a.source == "serial":
        src = SerialSource(a.serial_port)
    elif a.source == "lsl":
        try:
            src = LSLSource()
        except Exception as e:
            raise SystemExit(str(e))
    else:  # auto: serial (Mac/Linux) -> LSL -> UnicornPy (Windows)
        errs, src = [], None
        for name, make in (("Bluetooth serial", lambda: SerialSource(a.serial_port)),
                           ("LSL", lambda: LSLSource(timeout=4)), ("UnicornPy", UnicornPySource)):
            try:
                src = make(); break
            except Exception as e:
                errs.append(f"  {name}: {e}"); print(f"[source] {name} not available, trying next…")
        if src is None:
            raise SystemExit("Could not connect to the Unicorn.\n" + "\n".join(errs))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.check:
        try:
            ok = device_check(src, out)
        finally:
            if hasattr(src, "close"):
                src.close()
        raise SystemExit(0 if ok else 1)
    path = out / (a.name or f"unicorn_{datetime.now():%Y%m%d_%H%M%S}.csv")
    fh = open(path, "w", newline=""); w = csv.writer(fh)
    w.writerow(["lsl_time"] + [f"EEG {i + 1}" for i in range(N_EEG)] + ["marker"])
    print(f"[record] writing ONE file: {path.resolve()}")

    proc, latest, clients, raw_clients = Processor(), {"f": None}, set(), set()
    import collections
    raw_q = collections.deque(maxlen=5000)   # raw samples for the live pipeline board
    stop = threading.Event()

    def acquire():
        n, last_print, pending = 0, time.time(), []
        while not stop.is_set():
            try:
                x, ts, mks = src.read()
            except Exception as e:
                print(f"\n[source] headset disconnected / read error: {e}\n[source] stopping. Data so far is saved; rerun to continue.")
                stop.set(); break
            pending += mks
            for row, t in zip(x, ts):
                m = ""
                while pending and pending[0][0] <= t:      # attach marker to nearest following sample
                    m = (m + ";" if m else "") + pending.pop(0)[1]
                w.writerow([f"{t:.6f}"] + [f"{v:.3f}" for v in row] + [m])
            n += len(x)
            if raw_clients:
                raw_q.extend(zip(ts.tolist(), x.round(3).tolist()))
            for f in proc.push(x):
                latest["f"] = f
            if time.time() - last_print > 1 and latest["f"]:
                f = latest["f"]; last_print = time.time(); fh.flush()
                q = " ".join(f"{c}:{'ok' if s == 'good' else s[:5]:5s}" for c, s in zip(CH, f["quality"]))
                pa = f["summary"]["posterior_alpha_rel"]
                print(f"\r{n / FS:6.0f}s  good {f['summary']['n_good']}/8  postα {pa if pa is None else round(pa, 2)}  "
                      f"σ(µV) {f['std_uv']}  {q}", end="", flush=True)
        fh.flush()

    th = threading.Thread(target=acquire, daemon=True); th.start()

    async def serve():
        import websockets

        async def handler(ws):
            clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                clients.discard(ws)
        async def raw_handler(ws):
            raw_clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                raw_clients.discard(ws)

        async def pump_raw():
            while not stop.is_set():
                await asyncio.sleep(0.05)
                if not raw_q:
                    continue
                items = [raw_q.popleft() for _ in range(len(raw_q))]
                msg = json.dumps({"type": "raw", "fs": FS, "ch": CH, "t": [i[0] for i in items], "x": [i[1] for i in items]})
                for c in list(raw_clients):
                    try:
                        await c.send(msg)
                    except Exception:
                        raw_clients.discard(c)

        async with websockets.serve(handler, "0.0.0.0", a.port), websockets.serve(raw_handler, "0.0.0.0", a.port + 1):
            print(f"[ws] features : ws://localhost:{a.port}   -> dashboard/mindcart.html?ws=ws://localhost:{a.port}")
            print(f"[ws] raw EEG  : ws://localhost:{a.port + 1} -> dashboard/pipeline.html?ws=ws://localhost:{a.port + 1}")
            asyncio.get_running_loop().create_task(pump_raw())
            last = None
            while not stop.is_set():
                await asyncio.sleep(0.2)
                f = latest["f"]
                if f is not None and f is not last:
                    last = f; msg = json.dumps(f)
                    for c in list(clients):
                        try:
                            await c.send(msg)
                        except Exception:
                            clients.discard(c)

    try:
        if a.no_ws:
            while th.is_alive():
                time.sleep(0.5)
        else:
            asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); th.join(timeout=2); fh.close()
        if hasattr(src, "close"):
            src.close()
        print(f"\n[record] saved {path.resolve()}")


if __name__ == "__main__":
    main()
