"""
Претест качества фото — детерминированный, без нейросети, ~30 мс.

Закрывает пункт ТЗ "показывать низкую уверенность на некачественном фото".
Важно: это считается ДО вызова VLM, поэтому на мусорном фото вы честно
говорите "не могу оценить" вместо галлюцинации.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config


@dataclass
class QualityReport:
    score: float                      # 0..1, множитель к уверенности модели
    blur_var: float
    brightness: float
    width: int
    height: int
    issues: list[str] = field(default_factory=list)

    @property
    def is_poor(self) -> bool:
        return self.score < 0.7


def assess(image_bgr: np.ndarray) -> QualityReport:
    """Оценить пригодность фото для анализа."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    score = 1.0
    issues: list[str] = []

    # 1. Размытость
    if blur_var < config.BLUR_THRESHOLD:
        penalty = max(0.35, blur_var / config.BLUR_THRESHOLD)
        score *= penalty
        issues.append(f"Фото размыто (резкость {blur_var:.0f} при норме >{config.BLUR_THRESHOLD:.0f})")

    # 2. Освещённость
    if brightness < config.DARK_THRESHOLD:
        score *= 0.6
        issues.append(f"Фото слишком тёмное (яркость {brightness:.0f}/255)")
    elif brightness > config.BRIGHT_THRESHOLD:
        score *= 0.7
        issues.append(f"Фото пересвечено (яркость {brightness:.0f}/255)")

    # 3. Разрешение
    if min(h, w) < config.MIN_RESOLUTION:
        score *= 0.6
        issues.append(f"Низкое разрешение ({w}x{h} px)")

    # 4. Низкий контраст — типично для тумана, снега, съёмки против солнца
    contrast = float(gray.std())
    if contrast < 25:
        score *= 0.75
        issues.append(f"Низкий контраст сцены ({contrast:.0f})")

    return QualityReport(
        score=round(min(1.0, max(0.0, score)), 3),
        blur_var=round(blur_var, 1),
        brightness=round(brightness, 1),
        width=w,
        height=h,
        issues=issues,
    )
