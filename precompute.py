#!/usr/bin/env python3
import os, io, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from process_module import process_image

INPUT_DIR      = Path("input_images")
OUTPUT_JSON    = Path("segments")
OUTPUT_PREVIEW = Path("previews")

OUTPUT_JSON.mkdir(exist_ok=True)
OUTPUT_PREVIEW.mkdir(exist_ok=True)

for img_path in INPUT_DIR.glob("*.*"):
    name = img_path.stem
    segs, curves = process_image(str(img_path), threshold=128)

    # 1) write JSON of functions
    with open(OUTPUT_JSON / f"{name}.json", "w") as j:
        json.dump(segs, j, indent=2)

    # 2) render preview PNG
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_aspect("equal", "box")
    ax.axis("off")
    for pts in curves:
        t = np.linspace(0,1,50)
        x = (1-t)**3*pts[0][0] + 3*(1-t)**2*t*pts[1][0] + 3*(1-t)*t**2*pts[2][0] + t**3*pts[3][0]
        y = (1-t)**3*pts[0][1] + 3*(1-t)**2*t*pts[1][1] + 3*(1-t)*t**2*pts[2][1] + t**3*pts[3][1]
        ax.plot(x, y, linewidth=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    with open(OUTPUT_PREVIEW / f"{name}.png", "wb") as imgout:
        imgout.write(buf.getvalue())
