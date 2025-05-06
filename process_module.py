# process_module.py
import cv2, numpy as np, potrace

def process_image(image_path, threshold=128, min_length=0, max_length=float('inf')):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [], []
    h, w = gray.shape
    bmp = potrace.Bitmap(gray < threshold)
    paths = bmp.trace()

    desmos_segs = []
    preview_curves = []
    for curve in paths:
        start = curve.start_point
        for seg in curve:
            # corner vs bezier
            if seg.is_corner:
                p0, p1, p2, p3 = start, start, seg.c, seg.end_point
            else:
                p0, p1, p2, p3 = start, seg.c1, seg.c2, seg.end_point

            def loc(pt):
                return pt.x - w/2, h/2 - pt.y

            pts = list(map(loc, (p0, p1, p2, p3)))
            # length filter
            x0, y0 = pts[0]; x3, y3 = pts[3]
            length = np.hypot(x3-x0, y3-y0)
            if not (min_length <= length <= max_length):
                start = seg.end_point
                continue

            preview_curves.append(pts)

            Bx = f"(1-t)^3*{pts[0][0]:.2f} + 3*(1-t)^2*t*{pts[1][0]:.2f} + 3*(1-t)*t^2*{pts[2][0]:.2f} + t^3*{pts[3][0]:.2f}"
            By = f"(1-t)^3*{pts[0][1]:.2f} + 3*(1-t)^2*t*{pts[1][1]:.2f} + 3*(1-t)*t^2*{pts[2][1]:.2f} + t^3*{pts[3][1]:.2f}"
            dom = r"\left\{0 \le t \le 1\right\}"
            desmos_segs.append(f"({Bx}, {By}) {dom}")

            start = seg.end_point

    return desmos_segs, preview_curves
