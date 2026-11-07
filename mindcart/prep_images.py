"""Normalise product images and compute low-level visual features (the confound control).

Every image is fitted onto the same square, same grey background, so size/background
can't drive EEG differences. Brightness/colourfulness are saved so Model A can use them.

    python -m mindcart.prep_images stimuli/raw stimuli/items.csv --out stimuli/norm

items.csv must have at least: item_id,file   (optional: category,colour,price,style)
Writes stimuli/norm/<item_id>.png and adds columns luminance,contrast,colourfulness.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SIZE = 600
BG = (128, 128, 128)


def normalise(src: Path, dst: Path, size=SIZE):
    im = Image.open(src).convert("RGB")
    im.thumbnail((int(size * 0.9), int(size * 0.9)), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), BG)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    canvas.save(dst)
    return canvas


def image_features(im: Image.Image) -> dict:
    a = np.asarray(im, float) / 255.0
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colourfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())  # Hasler & Suesstrunk
    return {"luminance": float(lum.mean()), "contrast": float(lum.std()), "colourfulness": float(colourfulness)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("items_csv")
    ap.add_argument("--out", default="stimuli/norm")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = pd.read_csv(a.items_csv)
    feats = []
    for _, r in items.iterrows():
        im = normalise(Path(a.raw_dir) / r["file"], out / f"{r['item_id']}.png")
        feats.append(image_features(im))
    items = pd.concat([items.drop(columns=[c for c in ("luminance", "contrast", "colourfulness") if c in items]),
                       pd.DataFrame(feats)], axis=1)
    items["norm_file"] = [f"{i}.png" for i in items["item_id"]]
    items.to_csv(a.items_csv, index=False)
    print(f"Normalised {len(items)} images -> {out}; updated {a.items_csv}")


if __name__ == "__main__":
    main()
