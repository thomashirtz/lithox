#!/usr/bin/env python3
"""Regenerate regression PNGs under tests/data/ (dev only).

    python scripts/generate_reference.py

Set LITHOBENCH_METALSET=/path/to/MetalSet to refresh aerial_lithobench.png.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

import lithox as ltx

MASK_URL = (
    "https://raw.githubusercontent.com/thomashirtz/lithox/refs/heads/master/data/mask.png"
)
SIZE = 1024
CELL = "cell0"
OUT = Path(__file__).resolve().parents[1] / "tests" / "data"


def _save_gray(arr: np.ndarray, path: Path) -> None:
    gray = np.clip(arr, 0.0, 1.0)
    Image.fromarray((gray * 255).astype(np.uint8), mode="L").save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(ltx.load_image(MASK_URL, size=SIZE), dtype=np.float32)
    aerial = np.asarray(ltx.LithographySimulator.nominal()(mask).aerial, dtype=np.float32)

    _save_gray(mask, OUT / "mask.png")
    _save_gray(aerial, OUT / "aerial_lithox.png")

    lithobench_root = os.environ.get("LITHOBENCH_METALSET")
    if lithobench_root:
        path = Path(lithobench_root) / "litho" / f"{CELL}.png"
        im = Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        _save_gray(np.array(im, dtype=np.float32) / 255.0, OUT / "aerial_lithobench.png")
        print(f"Updated aerial_lithobench.png from {path}")
    else:
        print("LITHOBENCH_METALSET unset; skipped aerial_lithobench.png")

    print(f"Wrote {OUT}/mask.png and aerial_lithox.png")


if __name__ == "__main__":
    main()
