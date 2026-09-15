"""Отрисовка bbox дефекта на изображении."""
from __future__ import annotations

import cv2
import numpy as np

from . import config


def draw_bbox(image_bgr: np.ndarray, bbox_0_1000: list[int], level: str, caption: str) -> np.ndarray:
    """bbox приходит от Gemini как [ymin, xmin, ymax, xmax] в шкале 0..1000."""
    out = image_bgr.copy()
    if not bbox_0_1000 or len(bbox_0_1000) != 4:
        return out

    h, w = out.shape[:2]
    ymin, xmin, ymax, xmax = bbox_0_1000
    x1, y1 = int(xmin / 1000 * w), int(ymin / 1000 * h)
    x2, y2 = int(xmax / 1000 * w), int(ymax / 1000 * h)
    x1, x2 = sorted((max(0, x1), min(w - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(h - 1, y2)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return out

    hex_color = config.PRIORITY_COLORS.get(level, "#c62828").lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    color = (b, g, r)

    thickness = max(2, int(min(h, w) * 0.005))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    # плашка с подписью
    scale = max(0.5, min(h, w) / 900)
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 12
    cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 12, ty + 6), color, -1)
    cv2.putText(out, caption, (x1 + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out
