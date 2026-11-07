# Oxford Neurotech Hackathon: EEG pipeline (Consumer Tech)

A real-time EEG processing pipeline for the **g.tec Unicorn Hybrid Black** (8 channels, 250 Hz). It runs on recorded files, on synthetic data, or on a live LSL stream, using the same code each time. That means the app team can build against it without the headset.

```
Unicorn ─LSL─▶ input adapter ─▶ stateful filter (1–40 Hz + 50 Hz notch, causal)
         CSV ─▶ replay        ─▶ 2 s ring buffer, emits every 1 s
   synthetic ─▶               ─▶ signal-quality flags + Welch band power (δ θ α β, abs + relative)
                              ─▶ JSON feature frame ─▶ stdout / WebSocket ─▶ app
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                                    # 7 tests
python scripts/analyse_recording.py data/raw/2026-11-07/UnicornRecorder_*.csv    # quality report → reports/
python scripts/live.py --synthetic                           # fake live stream
python scripts/live.py --replay data/raw/2026-11-07/UnicornRecorder_07_11_2026_17_30_360.csv --ws 8765
python scripts/live.py --lsl --ws 8765                       # on the headset laptop
```

**Live dashboard:** open `dashboard/index.html` in a browser and click **Connect**, or add `?ws=ws://<laptop-ip>:8765` to the URL. Add `?demo` instead for a fake demo with no backend. It shows electrode quality on a head map, band power per channel, relaxation and engagement meters, and a 60 s trend. It also works on a phone.

The app connects to `ws://<laptop-ip>:8765` and gets one JSON frame per second:

```json
{"t": 12.0, "channels": ["Fz","C3",...], "quality": ["good","noisy",...],
 "alpha": [...], "alpha_rel": [...], "beta": [...], ...,
 "summary": {"n_good": 6, "posterior_alpha_rel": 0.31, "beta_alpha_ratio": 0.8}}
```

## Layout

| File | What it does |
|---|---|
| `eeg_pipeline/config.py` | Sample rate, channel names, filter and window settings, quality thresholds |
| `eeg_pipeline/io.py` | Unicorn CSV loader, synthetic EEG (alpha blocks, blinks, mains), LSL inlet, replay |
| `eeg_pipeline/preprocessing.py` | `StreamingFilter` (causal, keeps state between chunks), `filter_offline` (zero-phase) |
| `eeg_pipeline/features.py` | Welch PSD and band power |
| `eeg_pipeline/quality.py` | Per-channel `good / flat / noisy / artefact` |
| `eeg_pipeline/pipeline.py` | `EEGPipeline.push(chunk)` → feature frames; `process_recording`, `run_live` |
| `dashboard/index.html` | Live web dashboard (no install, reads the WebSocket, has a demo mode) |
| `scripts/` | `analyse_recording.py` (report and plots), `live.py` (live stream and WebSocket) |

## Findings on the 7 Nov recordings

Typical 2 s window amplitude after filtering (median across windows). Scalp EEG is normally about 10–40 µV.

| Session | Length | Typical amplitude | Verdict |
|---|---|---|---|
| 17:18, 17:19 | 1 s, 5 s | n/a | Test clicks, too short to use |
| 17:20 | 125 s | 7,000–28,000 µV | Not EEG: electrodes not making contact |
| 17:24 | 364 s | 1,200–2,100 µV | Not EEG: still poor contact |
| 17:30 | 109 s | **50–275 µV** | Marginal. Fz and PO8 are closest to usable. Large slow (1–2 Hz) waves on every channel point to movement or cable sway |

Other notes:

- **The quality improved a lot between sessions**, so whatever changed in the setup worked. Keep going: add more gel or wetting, press the electrodes down, check the ground and reference ear clips, and keep the cable still.
- **The `.bdf` exports have a broken scaling header**, which gives values around 10¹⁰ µV. Use the CSVs.
- `UnicornRecorder_*` and `UnicornRawDataRecorder_*` differ only slightly, by small offset and filtering differences.
- **There are no triggers or markers in any file.** For the validation runs, press the trigger or note timestamps.
- The cap layout is assumed to be `Fz C3 Cz C4 Pz PO7 Oz PO8` (Unicorn default). Check this against your cap.

## Next recordings (for validation)

1. Do the impedance or signal check in Unicorn Suite until all channels look flat and quiet.
2. Record **eyes open for 60 s, then eyes closed for 60 s**, with the subject sitting still. Posterior alpha (Pz, PO7, Oz, PO8) should clearly go up with eyes closed. Running `analyse_recording.py` on the file will show it.
3. Record 5 deliberate blinks. Fz should show them as `artefact`.

Until step 2 passes, we shouldn't claim the system detects relaxation or attention.

## Notes

- `StreamingFilter` is causal and keeps its state between chunks, so there's no look-ahead and no window-edge artefacts. The tests check that the output doesn't depend on chunk size and that later samples can't change earlier output.
- The first 4 s after starting are skipped while the filter settles.
- Don't commit identifiable participant data.

## Data layout

```
data/raw/2026-11-07/   original Unicorn Recorder exports (.csv + .bdf), 5 sessions × 2 recorders
data/raw/2026-11-08/   pilot 00:07 (150 s): clean signal (~10 µV, 97% clean windows), 11 Hz alpha peak ~95-125 s
data/merged/           all sessions of each recorder in ONE csv (zipped; columns: session_start, source_file, time_s, EEG 1-8, TRIG)
recordings/            new recordings from stream_unicorn.py (not committed)
```
