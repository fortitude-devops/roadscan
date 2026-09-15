"""
Пакетная проверка без UI — для часа борьбы с false positive.

  python analyze_cli.py demo_images/*.jpg --road main_road --traffic high
  python analyze_cli.py demo_images/normal_*.jpg --expect none

Флаг --expect показывает, на каких фото модель ошиблась: именно так вы
быстро находите, что дописать в стоп-лист промпта.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()

from app import config, priority, quality, vision  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="пути или маски файлов")
    ap.add_argument("--road", default="main_road", choices=list(config.ROAD_TYPES))
    ap.add_argument("--traffic", default="medium", choices=list(config.TRAFFIC_LEVELS))
    ap.add_argument("--weather", default="dry", choices=list(config.WEATHER))
    ap.add_argument("--zone", default="ZONE-CLI")
    ap.add_argument("--expect", default=None, choices=list(config.DEFECT_TYPES),
                    help="ожидаемый класс; в конце выводится точность")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.images:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    hits = total = 0
    print(f"{'файл':<34} {'класс':<14} {'sev':>3} {'conf':>5} {'балл':>6}  приоритет")
    print("-" * 82)

    for p in paths:
        image = cv2.imread(p)
        if image is None:
            print(f"{Path(p).name:<34} НЕ ПРОЧИТАН")
            continue

        qrep = quality.assess(image)
        try:
            det = vision.analyze(
                image,
                {
                    "zone_id": args.zone,
                    "road_type_label": config.ROAD_TYPES[args.road],
                    "weather_label": config.WEATHER[args.weather],
                },
                use_cache=not args.no_cache,
            ).detection
        except Exception as exc:  # noqa: BLE001
            print(f"{Path(p).name:<34} ОШИБКА: {exc}")
            continue

        pres = priority.compute(det.defect_type, det.severity, det.confidence, qrep.score,
                                args.road, args.traffic, args.weather, [])

        mark = ""
        if args.expect:
            total += 1
            ok = det.defect_type == args.expect
            hits += ok
            mark = "  ✅" if ok else f"  ❌ ждали {args.expect}"

        flag = " ⚠️ручная" if pres.needs_manual_review else ""
        print(f"{Path(p).name:<34} {det.defect_type:<14} {det.severity:>3} "
              f"{det.confidence:>5.2f} {pres.score:>6.2f}  {pres.label}{flag}{mark}")

    if args.expect and total:
        print("-" * 82)
        print(f"Точность по классу «{args.expect}»: {hits}/{total} = {hits / total:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
