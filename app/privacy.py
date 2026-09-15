"""
Анонимизация: размытие лиц и номерных знаков перед обработкой и показом.

Требование ТЗ "Безопасность". Занимает 15 минут, а в критериях — отдельная
строка, поэтому не пропускайте.

Каскады Хаара идут в комплекте с opencv-python, ничего качать не нужно.
"""
from __future__ import annotations

import cv2
import numpy as np

_face_cascade = None
_plate_cascade = None


def _cascades():
    global _face_cascade, _plate_cascade
    if _face_cascade is None:
        base = cv2.data.haarcascades
        _face_cascade = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        _plate_cascade = cv2.CascadeClassifier(base + "haarcascade_russian_plate_number.xml")
    return _face_cascade, _plate_cascade


def anonymize(image_bgr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Вернуть копию изображения с замыленными лицами и номерами + список находок."""
    faces_cc, plates_cc = _cascades()
    out = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    found: list[str] = []

    regions = []
    for cc, label in ((faces_cc, "лицо"), (plates_cc, "автомобильный номер")):
        if cc.empty():
            continue
        for (x, y, w, h) in cc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24)):
            regions.append((x, y, w, h))
            found.append(label)

    for (x, y, w, h) in regions:
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        k = max(11, (max(w, h) // 4) | 1)   # ядро должно быть нечётным
        out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)

    return out, found
