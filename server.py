#!/usr/bin/env python3
import io
import os
import base64
import tempfile

from flask import Flask, request, jsonify, render_template, CORS
import cv2
import numpy as np
import potrace
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

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
