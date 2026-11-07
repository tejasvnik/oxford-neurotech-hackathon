"""MindCart viewing task (PsychoPy). Sends LSL markers for every event.

Trial: fixation 1 s -> product image 4 s -> "How interested are you? 1-5" (keys 1-5)
       -> jittered blank 0.5-0.9 s. Break screen halfway. Esc quits safely.

    python mindcart/experiment.py --pid P01 --items stimuli/items.csv --img-dir stimuli/norm
    python mindcart/experiment.py --pid TEST --n 5 --windowed     # 5-trial sync check

Markers (string, one per event):
  start | fix|<trial> | img|<trial>|<item_id> | rate|<trial>|<item_id>|<rating>|<rt> | break | end
Also writes sessions/<pid>/behaviour.csv as a backup.
"""
import argparse
import csv
import random
import zlib
from pathlib import Path

from pylsl import StreamInfo, StreamOutlet, local_clock
from psychopy import core, event, visual

FIX_S, IMG_S, RATE_MAX_S = 1.0, 4.0, 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--items", default="stimuli/items.csv")
    ap.add_argument("--img-dir", default="stimuli/norm")
    ap.add_argument("--n", type=int, default=None, help="limit number of trials (testing)")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="sessions")
    a = ap.parse_args()

    with open(a.items) as f:
        items = list(csv.DictReader(f))
    rng = random.Random(a.seed if a.seed is not None else zlib.crc32(a.pid.encode()))  # stable per participant
    rng.shuffle(items)
    if a.n:
        items = items[:a.n]

    outlet = StreamOutlet(StreamInfo("MindCartMarkers", "Markers", 1, 0, "string", f"mindcart_{a.pid}"))
    out = Path(a.out) / a.pid
    out.mkdir(parents=True, exist_ok=True)
    beh = open(out / "behaviour.csv", "w", newline="")
    bw = csv.writer(beh)
    bw.writerow(["trial", "item_id", "onset_lsl", "rating", "rt"])

    win = visual.Window(size=(1280, 800) if a.windowed else None, fullscr=not a.windowed,
                        color=(0, 0, 0), units="height")  # mid-grey, matches image background
    fix = visual.TextStim(win, "+", height=0.08, color="white")
    msg = visual.TextStim(win, "", height=0.04, color="white", wrapWidth=1.2)
    rate = visual.TextStim(win, "How interested are you in this item?\n\n1   2   3   4   5\n\n"
                                "(1 = not at all, 5 = very)", height=0.045, color="white")
    img = visual.ImageStim(win, size=(0.8, 0.8))

    def mark(s):
        outlet.push_sample([s], local_clock())

    def show_msg(text):
        msg.text = text
        msg.draw(); win.flip()
        keys = event.waitKeys(keyList=["space", "escape"])
        if "escape" in keys:
            quit_()

    def quit_():
        mark("end"); beh.close(); win.close(); core.quit()

    show_msg("Look at each clothing item while keeping still and relaxed.\n"
             "Try not to blink while the item is on screen.\n\n"
             "Then rate how interested you are (keys 1-5).\n\nPress SPACE to start.")
    core.wait(1.0)  # give the recorder a moment
    mark("start")
    half = len(items) // 2
    clock = core.Clock()
    for t, it in enumerate(items):
        if t == half and len(items) > 10:
            mark("break")
            show_msg("Short break. Blink, stretch, relax.\n\nPress SPACE to continue.")
        img.image = str(Path(a.img_dir) / it.get("norm_file", it["file"]))
        # fixation
        fix.draw(); win.callOnFlip(mark, f"fix|{t}"); win.flip(); core.wait(FIX_S)
        # product
        img.draw()
        onset = {}
        win.callOnFlip(lambda: onset.setdefault("t", local_clock()))
        win.callOnFlip(mark, f"img|{t}|{it['item_id']}")
        win.flip(); core.wait(IMG_S)
        # rating
        rate.draw(); win.flip(); clock.reset(); event.clearEvents()
        keys = event.waitKeys(maxWait=RATE_MAX_S, keyList=list("12345") + ["escape"], timeStamped=clock)
        if keys and keys[0][0] == "escape":
            quit_()
        r, rt = (int(keys[0][0]), round(keys[0][1], 3)) if keys else ("", "")
        mark(f"rate|{t}|{it['item_id']}|{r}|{rt}")
        bw.writerow([t, it["item_id"], f"{onset.get('t', 0):.6f}", r, rt]); beh.flush()
        win.flip(); core.wait(rng.uniform(0.5, 0.9))
    show_msg("Done, thank you!\n\nPress SPACE to finish.")
    quit_()


if __name__ == "__main__":
    main()
