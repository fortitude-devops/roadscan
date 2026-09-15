"""
Проверка веб-сервиса без обращения к Gemini.

Запуск:  python api_selftest.py

Подменяет слой распознавания заглушкой и прогоняет весь путь:
загрузка фото -> анализ -> очередь -> смена статуса -> экспорт CSV.
Полезно, когда закончилась квота API, а фронтенд надо отлаживать дальше.
"""
from __future__ import annotations

import io
import sys

import cv2
import numpy as np

from app import vision

# --- заглушка модели: подставляем ДО импорта API -------------------------
_FAKE = vision.DefectDetection(
    defect_type="pothole", severity=4, confidence=0.87,
    bbox=[420, 300, 720, 660],
    location_text="Правая полоса, 2 м от бордюра",
    description="Выбоина с рваными краями и видимой глубиной, около 40 см в поперечнике.",
    evidence=["рваная кромка асфальта", "видна глубина и обнажённое основание", "выкрошенный материал по краю"],
)
vision.analyze = lambda image, context, use_cache=True: vision.VisionResult(_FAKE, False, "stub")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from api import store  # noqa: E402

client = TestClient(app)
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'✅' if cond else '❌'} {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def fake_photo() -> bytes:
    img = np.random.randint(60, 190, (720, 1280, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


print("=== Служебные эндпоинты ===")
check("GET /healthz", client.get("/healthz").status_code == 200)
check("GET /docs (Swagger)", client.get("/docs").status_code == 200)

d = client.get("/api/dictionaries").json()
check("GET /api/dictionaries", "road_types" in d and "defect_types" in d,
      f"{len(d['road_types'])} типов дорог")

z = client.get("/api/zones").json()
check("GET /api/zones", len(z) >= 10, f"{len(z)} участков")

print("\n=== Анализ снимка ===")
r = client.post("/api/analyze", files={"file": ("road.jpg", io.BytesIO(fake_photo()), "image/jpeg")},
                data={"zone_id": "ZONE-03", "weather": "dry"})
check("POST /api/analyze (школа)", r.status_code == 200, f"HTTP {r.status_code}")
school = r.json()
check("Контекст взят из справочника зон",
      school["context"]["road_type"] == "school_access", school["context"]["zone_name"])
check("Приоритет у школы высокий", school["priority"]["level"] == "high",
      f"{school['priority']['score']} балла")
check("Факторы приоритета возвращаются", len(school["priority"]["factors"]) >= 4,
      f"{len(school['priority']['factors'])} факторов")
check("Снимок с рамкой отдаётся", client.get(school["image_url"]).status_code == 200)

r2 = client.post("/api/analyze", files={"file": ("road.jpg", io.BytesIO(fake_photo()), "image/jpeg")},
                 data={"zone_id": "ZONE-08"})
yard = r2.json()
check("Тот же дефект во дворе — ниже приоритет",
      yard["priority"]["score"] < school["priority"]["score"],
      f"{yard['priority']['score']} < {school['priority']['score']}")

print("\n=== Пересчёт контекста (what-if) ===")
wi = client.post("/api/what-if", json={"report_id": school["report_id"],
                                       "road_type": "yard", "traffic_level": "low"})
check("POST /api/what-if", wi.status_code == 200)
check("Перенос на дворовой проезд снижает приоритет",
      wi.json()["score"] < school["priority"]["score"],
      f"{wi.json()['score']} < {school['priority']['score']}")

print("\n=== Валидация входа ===")
check("Битый файл отклоняется с 400",
      client.post("/api/analyze", files={"file": ("x.jpg", io.BytesIO(b"not an image"), "image/jpeg")}
                  ).status_code == 400)
check("Неизвестный тип дороги отклоняется с 422",
      client.post("/api/analyze", files={"file": ("r.jpg", io.BytesIO(fake_photo()), "image/jpeg")},
                  data={"zone_id": "", "road_type": "космодром"}).status_code == 422)

print("\n=== Очередь ремонта ===")
rows = client.get("/api/reports").json()
check("GET /api/reports", len(rows) >= 2, f"{len(rows)} заявок")
check("Очередь отсортирована по баллу",
      all(rows[i]["score"] >= rows[i + 1]["score"] for i in range(len(rows) - 1)))
check("Есть срок исполнения", bool(rows[0]["due_date"]), rows[0]["due_date"])

st = client.get("/api/reports/stats").json()
check("GET /api/reports/stats", st["total"] >= 2, str(st["by_level"]))

check("Фильтр по приоритету",
      all(x["priority_level"] == "high" for x in client.get("/api/reports?level=high").json()))

up = client.patch(f"/api/reports/{school['report_id']}/status", json={"status": "in_progress"})
check("PATCH статуса заявки", up.status_code == 200 and up.json()["status"] == "in_progress")
check("Недопустимый статус отклоняется",
      client.patch(f"/api/reports/{school['report_id']}/status",
                   json={"status": "чинится"}).status_code == 422)

csv_resp = client.get("/api/export.csv")
check("GET /api/export.csv", csv_resp.status_code == 200 and "ROAD-" in csv_resp.text)
check("CSV с BOM для Excel", csv_resp.text.startswith("﻿"))

print("\n=== Фронтенд ===")
check("Страница жителя отдаётся", client.get("/static/index.html").status_code == 200)
check("Дашборд отдаётся", client.get("/dashboard").status_code == 200)
check("CSS и JS отдаются",
      client.get("/static/styles.css").status_code == 200
      and client.get("/static/app.js").status_code == 200
      and client.get("/static/dashboard.js").status_code == 200)
check("Корень редиректит на страницу жителя",
      client.get("/", follow_redirects=False).status_code in (307, 302))

store.delete_all()
print("\n" + ("🎉 Все проверки пройдены" if not FAILS else f"⚠️ Провалено: {FAILS}"))
sys.exit(1 if FAILS else 0)
