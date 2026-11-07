"""Record the Unicorn EEG stream + MindCart marker stream to CSV, on one shared LSL clock.

Start order on the headset laptop:
  1. Unicorn LSL app -> start streaming (stream type "EEG")
  2. python -m mindcart.record_session --pid P01         (this script; waits for both streams)
  3. python mindcart/experiment.py --pid P01              (PsychoPy; sends markers)
  4. When the experiment ends, the recorder stops by itself (on the "end" marker) or press Ctrl+C.

Output: sessions/P01/eeg.csv  (lsl_time, EEG 1..8)  and  sessions/P01/markers.csv (lsl_time, marker)
"""
import argparse
import csv
import time
from pathlib import Path

from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_byprop


def find(prop, value, timeout):
    print(f"Looking for LSL stream {prop}={value!r} ...")
    s = resolve_byprop(prop, value, timeout=timeout)
    if not s:
        raise SystemExit(f"Not found: {prop}={value!r}. Is it running?")
    print(f"  found: {s[0].name()} ({s[0].channel_count()} ch @ {s[0].nominal_srate()} Hz)")
    return StreamInlet(s[0], max_buflen=60, processing_flags=proc_clocksync | proc_dejitter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--eeg-type", default="EEG")
    ap.add_argument("--marker-name", default="MindCartMarkers")
    ap.add_argument("--n-eeg", type=int, default=8, help="keep first N channels (Unicorn LSL adds accel etc.)")
    ap.add_argument("--out", default="sessions")
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--max-minutes", type=float, default=40)
    a = ap.parse_args()

    out = Path(a.out) / a.pid
    out.mkdir(parents=True, exist_ok=True)
    eeg = find("type", a.eeg_type, a.timeout)
    mk = find("name", a.marker_name, a.timeout)

    n_eeg = n_mk = 0
    t_end = time.time() + a.max_minutes * 60
    with open(out / "eeg.csv", "w", newline="") as fe, open(out / "markers.csv", "w", newline="") as fm:
        we, wm = csv.writer(fe), csv.writer(fm)
        we.writerow(["lsl_time"] + [f"EEG {i + 1}" for i in range(a.n_eeg)])
        wm.writerow(["lsl_time", "marker"])
        print(f"Recording -> {out}  (Ctrl+C to stop)")
        try:
            done = False
            while not done and time.time() < t_end:
                data, ts = eeg.pull_chunk(timeout=0.2, max_samples=512)
                for row, t in zip(data, ts):
                    we.writerow([f"{t:.6f}"] + [f"{v:.4f}" for v in row[:a.n_eeg]])
                n_eeg += len(ts)
                mdata, mts = mk.pull_chunk(timeout=0.0)
                for row, t in zip(mdata, mts):
                    wm.writerow([f"{t:.6f}", row[0]])
                    fm.flush()
                    n_mk += 1
                    if row[0] == "end":
                        done = True
                if n_eeg and n_eeg % 2500 < len(ts):
                    print(f"  {n_eeg / 250:.0f} s EEG, {n_mk} markers")
            # grab trailing EEG after "end"
            time.sleep(1.0)
            data, ts = eeg.pull_chunk(timeout=0.5, max_samples=2048)
            for row, t in zip(data, ts):
                we.writerow([f"{t:.6f}"] + [f"{v:.4f}" for v in row[:a.n_eeg]])
            n_eeg += len(ts)
        except KeyboardInterrupt:
            pass
    print(f"Saved {n_eeg} EEG samples ({n_eeg / 250:.0f} s) and {n_mk} markers to {out}")


if __name__ == "__main__":
    main()
