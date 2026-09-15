"""
Самопроверка без обращения к API.

Запуск:  python selftest.py
Проверяет, что движок приоритета ведёт себя так, как обещано судьям.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from app import priority, quality, render

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'✅' if cond else '❌'} {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


print("=== Движок приоритета ===")

school = priority.compute("pothole", 3, 0.9, 1.0, "school_access", "medium", "dry", ["school_zone", "pedestrian_crossing"])
yard = priority.compute("pothole", 3, 0.9, 1.0, "yard", "low", "dry", [])
check("Тот же дефект у школы важнее, чем во дворе",
      school.score > yard.score, f"{school.score} > {yard.score}")
check("У школы приоритет высокий", school.level == "high", school.label)
check("Во дворе приоритет низкий", yard.level == "low", yard.label)

crack = priority.compute("crack", 2, 0.9, 1.0, "secondary", "low", "dry", [])
check("Мелкая трещина на второстепенной — низкий", crack.level == "low", f"{crack.score}")

critical = priority.compute("pothole", 5, 0.95, 1.0, "highway", "high", "wet", ["sharp_turn"])
check("Критическая яма на трассе — высокий", critical.level == "high", f"{critical.score}")

none = priority.compute("none", 0, 0.95, 1.0, "main_road", "high", "dry", [])
check("«Дефекта нет» не создаёт приоритет", none.score == 0.0 and not none.needs_manual_review)

blurry = priority.compute("pothole", 5, 0.5, 0.4, "highway", "high", "dry", [])
check("Низкая уверенность уводит на ручную проверку", blurry.needs_manual_review)
check("Низкая уверенность не даёт высший приоритет", blurry.level != "high", blurry.label)

wet = priority.compute("pothole", 3, 0.9, 1.0, "main_road", "high", "wet", [])
dry = priority.compute("pothole", 3, 0.9, 1.0, "main_road", "high", "dry", [])
check("Дождь повышает балл", wet.score > dry.score, f"{wet.score} > {dry.score}")

check("Факторы раскладываются в объяснение", len(school.factors) >= 4,
      f"{len(school.factors)} факторов")
print("\n  Пример объяснения (школа):")
for line in school.explanation_lines():
    print("    ", line)

print("\n=== Оценка качества фото ===")
sharp = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
blur_img = cv2.GaussianBlur(sharp, (31, 31), 0)
dark = (np.ones((600, 800, 3), dtype=np.uint8) * 20)

check("Резкое фото оценено высоко", quality.assess(sharp).score > 0.9)
check("Размытое фото понижает оценку", quality.assess(blur_img).score < 0.7,
      f"score={quality.assess(blur_img).score}")
check("Тёмное фото помечено проблемой", any("тёмное" in i for i in quality.assess(dark).issues))

print("\n=== Отрисовка bbox ===")
img = np.zeros((480, 640, 3), dtype=np.uint8)
out = render.draw_bbox(img, [300, 200, 700, 600], "high", "HIGH 12.4")
check("bbox отрисован", out.shape == img.shape and out.sum() > 0)
check("Пустой bbox не ломает отрисовку", render.draw_bbox(img, [], "low", "x").shape == img.shape)

print("\n" + ("🎉 Все проверки пройдены" if not FAILS else f"⚠️ Провалено: {FAILS}"))
sys.exit(1 if FAILS else 0)
