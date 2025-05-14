#!/usr/bin/env python3
import io
import os
import base64
import tempfile

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import potrace
import matplotlib.pyplot as plt

app = Flask(__name__)


def process_image(image_path, threshold=128, min_length=0, max_length=float('inf')):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [], []

    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    bmp = potrace.Bitmap(gray < threshold)
    paths = bmp.trace()

    xs, ys = [], []
    for curve in paths:
        start = curve.start_point
        for seg in curve:
            pts = [start] + ([seg.c] if seg.is_corner else [seg.c1, seg.c2]) + [seg.end_point]
            for p in pts:
                xs.append(p.x); ys.append(p.y)
            start = seg.end_point
    if not xs:
        return [], []

    scale = ((max(xs)/w) + (max(ys)/h)) / 2.0 or 1.0

    desmos_segments = []
    preview_curves = []

    for curve in paths:
        start = curve.start_point
        for seg in curve:
            if seg.is_corner:
                p0, p1, p2, p3 = start, start, seg.c, seg.end_point
            else:
                p0, p1, p2, p3 = start, seg.c1, seg.c2, seg.end_point

            def loc(pt):
                x = pt.x/scale - cx
                y = cy - pt.y/scale
                return x, y

            pts_pixel = [loc(p) for p in (p0, p1, p2, p3)]
            (x0, y0), _, _, (x3, y3) = pts_pixel
            length = np.hypot(x3-x0, y3-y0)

            if length < min_length or length > max_length:
                start = seg.end_point
                continue

            preview_curves.append(np.array(pts_pixel))

            Bx = (
                f"(1-t)**3*{pts_pixel[0][0]}"
                f" + 3*(1-t)**2*t*{pts_pixel[1][0]}"
                f" + 3*(1-t)*t**2*{pts_pixel[2][0]}"
                f" + t**3*{pts_pixel[3][0]}"
            )
            By = (
                f"(1-t)**3*{pts_pixel[0][1]}"
                f" + 3*(1-t)**2*t*{pts_pixel[1][1]}"
                f" + 3*(1-t)*t**2*{pts_pixel[2][1]}"
                f" + t**3*{pts_pixel[3][1]}"
            )
            domain = r"\left\{0 \le t \le 1\right\}"
            desmos_segments.append(f"({Bx}, {By}) {domain}")
            start = seg.end_point

    return desmos_segments, preview_curves

@app.route('/')
def index():
    # Serve the standalone HTML file from the templates folder
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    img = request.files.get('image')
    thr = int(request.form.get('threshold', 128))
    mn  = int(request.form.get('minLength', 0))
    mx  = int(request.form.get('maxLength', 1e9))

    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    img.save(path)
    segments, curves = process_image(path, thr, mn, mx)
    os.remove(path)

    # Generate preview image
    fig, ax = plt.subplots(figsize=(4,4))

    # Force the same scale on X and Y so nothing gets distorted
    ax.set_aspect('equal', adjustable='box')

    for pts in curves:
        t = np.linspace(0,1,50)
        x = (1-t)**3*pts[0,0] + 3*(1-t)**2*t*pts[1,0] + 3*(1-t)*t**2*pts[2,0] + t**3*pts[3,0]
        y = (1-t)**3*pts[0,1] + 3*(1-t)**2*t*pts[1,1] + 3*(1-t)*t**2*pts[2,1] + t**3*pts[3,1]
        ax.plot(x, y, linewidth=1)

    # Remove margins and axes
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    preview_b64 = base64.b64encode(buf.read()).decode('ascii')

    return jsonify({ 'segments': segments, 'preview': preview_b64 })

if __name__ == '__main__':
    app.run(debug=True)