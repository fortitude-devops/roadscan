"""
RoadScan API — веб-сервис выявления дефектов дорожного покрытия.

Запуск:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

    Страница жителя:  http://localhost:8000/
    Дашборд акимата:  http://localhost:8000/dashboard
    Документация API: http://localhost:8000/docs
"""
from __future__ import annotations

import csv
import io
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from app import config, priority, privacy, providers, quality, render, vision  # noqa: E402

from . import store  # noqa: E402
from .schemas import (  # noqa: E402
    AnalyzeResponse,
    DictionariesOut,
    PriorityOut,
    ReportSummary,
    StatusUpdate,
    WhatIfRequest,
    ZoneOut,
)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

app = FastAPI(
    title="RoadScan API",
    version="1.0.0",
    description=(
        "Сервис выявления дефектов дорожного покрытия по фотографии "
        "и приоритизации ремонта.\n\n"
        "**Архитектура.** Мультимодальная модель отвечает только на вопрос «что видно на снимке» — "
        "тип дефекта и его тяжесть. Приоритет ремонта считает детерминированный движок "
        "на контексте участка: тип дороги, трафик, погода, близость школы или перехода. "
        "Поэтому приоритет воспроизводим, объясним в баллах и настраивается без переобучения модели.\n\n"
        "**Ограничение.** Сервис выдаёт рекомендацию для планирования работ "
        "и не назначает санкции подрядчику."
    ),
    contact={"name": "HackAlem AI — кейс 9"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # хакатон-режим; в проде — список доменов
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Диагностика ошибок
# ---------------------------------------------------------------------------
ERROR_LOG = ROOT / "storage" / "errors.log"
log = logging.getLogger("roadscan")

# Рабочий размер снимка: всё, что больше, уменьшается сразу после загрузки
WORK_IMAGE_SIDE = 1600


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """
    Любое необработанное исключение вместо пустой «500 Internal Server Error»
    возвращает текст причины и дописывает полную трассировку в storage/errors.log.
    На хакатоне это экономит часы: причина видна прямо на странице.
    """
    tb = traceback.format_exc()
    stamp = datetime.now(timezone.utc).isoformat()
    try:
        with ERROR_LOG.open("a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 70}\n{stamp}  {request.method} {request.url.path}\n{tb}")
    except Exception:  # noqa: BLE001 — диагностика не должна ронять сервис
        pass
    log.error("Необработанная ошибка на %s:\n%s", request.url.path, tb)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}",
                 "hint": "Полная трассировка — в консоли сервера и в storage/errors.log"},
    )


class Stage:
    """Помечает этап обработки, чтобы в ошибке было видно, где именно упало."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "Stage":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None or isinstance(exc, HTTPException):
            return False
        raise HTTPException(500, f"Этап «{self.name}»: {type(exc).__name__}: {exc}") from exc


def imwrite_safe(path: Path, image: np.ndarray, quality: int = 85) -> bool:
    """
    cv2.imwrite не умеет пути с кириллицей на Windows (имя пользователя вида
    C:\\Users\\Алмас) и молча возвращает False. Пишем через imencode.
    """
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


# ---------------------------------------------------------------------------
# Справочник участков
# ---------------------------------------------------------------------------
def _load_zones() -> dict[str, dict]:
    df = pd.read_csv(ROOT / "data" / "zones.csv")
    df["context_flags"] = df["context_flags"].fillna("")
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        out[r["zone_id"]] = {
            "zone_id": r["zone_id"],
            "zone_name": r["zone_name"],
            "road_type": r["road_type"],
            "road_type_label": config.ROAD_TYPES.get(r["road_type"], r["road_type"]),
            "traffic_level": r["traffic_level"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "context_flags": [f for f in str(r["context_flags"]).split("|") if f],
        }
    return out


ZONES = _load_zones()


# ---------------------------------------------------------------------------
# Служебное
# ---------------------------------------------------------------------------
BUILD = "2026-09-15.8"   # метка сборки: видно, какая версия реально запущена


@app.get("/healthz", tags=["служебные"], summary="Проверка живости сервиса")
def healthz() -> dict:
    return {
        "status": "ok",
        "build": BUILD,
        "папка_сервиса": str(ROOT),
        "reports": store.stats()["total"],
    }


@app.get("/api/diag", tags=["служебные"], summary="Диагностика: где именно ломается обработка")
def diag() -> dict:
    """
    Прогоняет весь конвейер на сгенерированном снимке и отдельно пробует
    обратиться к модели. Показывает, какой этап падает и с какой ошибкой.

    Откройте http://localhost:8000/api/diag и пришлите вывод — по нему
    причина видна без чтения консоли.
    """
    import os
    import platform
    import sys

    report: dict = {
        "сборка": BUILD,
        "версии": {
            "python": sys.version.split()[0],
            "платформа": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "окружение": {},
        "этапы": {},
    }

    # --- провайдеры и ключи ---
    report["провайдеры"] = {
        "порядок_использования": providers.chain() or "НЕ НАСТРОЕН НИ ОДИН",
        "ключ_задан": providers.available(),
    }
    report["окружение"]["офлайн_режим"] = os.getenv("OFFLINE_MODE") == "1"

    try:
        from app.providers.local_detector import LocalDetectorProvider
        report["провайдеры"]["локальная_модель"] = str(LocalDetectorProvider._find_model())
    except Exception as exc:  # noqa: BLE001
        report["провайдеры"]["локальная_модель"] = f"нет: {exc}"
    report["окружение"]["путь_проекта"] = str(ROOT)
    report["окружение"]["путь_только_латиница"] = ROOT.as_posix().isascii()

    try:
        probe = store.IMAGES / "_diag_probe.jpg"
        probe.write_bytes(b"test")
        probe.unlink()
        report["окружение"]["папка_storage_доступна_на_запись"] = True
    except Exception as exc:  # noqa: BLE001
        report["окружение"]["папка_storage_доступна_на_запись"] = f"НЕТ: {exc}"

    # --- синтетический снимок ---
    img = np.clip(
        np.repeat(np.linspace(70, 135, 480)[:, None], 640, axis=1)
        + np.random.default_rng(0).normal(0, 26, (480, 640)), 0, 255
    ).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def step(name: str, fn):
        try:
            fn()
            report["этапы"][name] = "ok"
        except Exception as exc:  # noqa: BLE001
            report["этапы"][name] = {
                "ошибка": f"{type(exc).__name__}: {exc}",
                "трассировка": traceback.format_exc().splitlines()[-6:],
            }

    step("анонимизация", lambda: privacy.anonymize(img))
    step("оценка качества", lambda: quality.assess(img))
    step("расчёт приоритета", lambda: priority.compute(
        "pothole", 4, 0.9, 1.0, "main_road", "high", "dry", []))
    step("отрисовка рамки", lambda: render.draw_bbox(img, [100, 100, 400, 400], "high", "HIGH 12.0"))
    step("запись файла", lambda: imwrite_safe(store.IMAGES / "_diag.jpg", img))
    step("справочник зон", lambda: len(ZONES))

    # --- реальное обращение к модели ---
    try:
        vres = vision.analyze(
            img,
            {"zone_id": "DIAG", "road_type_label": "Городская магистраль", "weather_label": "Сухо"},
            use_cache=False,
        )
        report["этапы"]["вызов модели"] = {
            "статус": "ok",
            "ответил": vres.model,
            "класс_дефекта": vres.detection.defect_type,
        }
    except Exception as exc:  # noqa: BLE001
        report["этапы"]["вызов модели"] = {
            "ошибка": f"{type(exc).__name__}: {exc}",
            "трассировка": traceback.format_exc().splitlines()[-8:],
        }

    (store.IMAGES / "_diag.jpg").unlink(missing_ok=True)

    bad = [k for k, v in report["этапы"].items() if v != "ok" and not (
        isinstance(v, dict) and v.get("статус") == "ok")]
    report["итог"] = "Все этапы прошли" if not bad else f"Падают этапы: {', '.join(bad)}"
    return report


@app.get("/api/dictionaries", response_model=DictionariesOut, tags=["справочники"],
         summary="Все справочники для фронтенда одним запросом")
def dictionaries() -> DictionariesOut:
    return DictionariesOut(
        defect_types=config.DEFECT_TYPES,
        road_types=config.ROAD_TYPES,
        traffic_levels=config.TRAFFIC_LEVELS,
        weather=config.WEATHER,
        context_flags={k: v[0] for k, v in config.CONTEXT_BONUS.items()},
        priority_labels=config.PRIORITY_LABELS,
        priority_colors=config.PRIORITY_COLORS,
        thresholds=config.PRIORITY_THRESHOLDS,
    )


@app.get("/api/zones", response_model=list[ZoneOut], tags=["справочники"],
         summary="Справочник участков дорожной сети")
def zones() -> list[ZoneOut]:
    return [ZoneOut(**z) for z in ZONES.values()]


# ---------------------------------------------------------------------------
# Анализ снимка
# ---------------------------------------------------------------------------
@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["анализ"],
          summary="Проанализировать фотографию дороги")
def analyze(
    file: UploadFile = File(..., description="Фотография дорожного покрытия"),
    zone_id: str = Form("", description="ID участка из справочника; если задан, контекст берётся из него"),
    road_type: str = Form("main_road", description="Тип дороги, если участок не указан"),
    traffic_level: str = Form("medium", description="Интенсивность движения"),
    weather: str = Form("dry", description="Погода на момент съёмки"),
    context_flags: str = Form("", description="Контекстные флаги через запятую"),
    anonymize: bool = Form(True, description="Размыть лица и автомобильные номера"),
    save: bool = Form(True, description="Сохранить заявку в очередь ремонта"),
) -> AnalyzeResponse:
    """
    Принимает снимок, возвращает тип дефекта, его локализацию, приоритет ремонта
    и полный разбор факторов, повлиявших на приоритет.

    Эндпоинт синхронный (`def`, не `async def`): FastAPI выполняет его в пуле
    потоков, поэтому обращение к модели не блокирует остальные запросы.
    """
    started = time.perf_counter()

    raw = np.frombuffer(file.file.read(), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            400,
            "Не удалось прочитать изображение. Поддерживаются JPEG, PNG и WebP. "
            "Формат HEIC с iPhone не читается — переснимите или сохраните как JPEG.",
        )

    # Снимок с телефона — это 12 мегапикселей и 4 МБ. Каскады Хаара на таком
    # размере работают секундами и могут упереться в память на слабом ноутбуке,
    # а пороги качества в config.py калибруются под предсказуемый масштаб.
    # Поэтому приводим любой снимок к одному размеру до начала обработки.
    with Stage("подготовка изображения"):
        h, w = image.shape[:2]
        if max(h, w) > WORK_IMAGE_SIDE:
            k = WORK_IMAGE_SIDE / max(h, w)
            image = cv2.resize(image, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)

    # --- контекст участка ---
    zone = ZONES.get(zone_id)
    if zone:
        road_type = zone["road_type"]
        traffic_level = zone["traffic_level"]
        flags = zone["context_flags"]
        zone_name, lat, lon = zone["zone_name"], zone["lat"], zone["lon"]
    else:
        flags = [f.strip() for f in context_flags.split(",") if f.strip()]
        zone_name, lat, lon = None, None, None

    if road_type not in config.ROAD_TYPES:
        raise HTTPException(422, f"Неизвестный тип дороги: {road_type}")
    if traffic_level not in config.TRAFFIC_LEVELS:
        raise HTTPException(422, f"Неизвестный уровень трафика: {traffic_level}")
    if weather not in config.WEATHER:
        raise HTTPException(422, f"Неизвестная погода: {weather}")

    # --- анонимизация ---
    blurred: list[str] = []
    if anonymize:
        with Stage("анонимизация лиц и номеров"):
            image, blurred = privacy.anonymize(image)

    # --- качество снимка ---
    with Stage("оценка качества снимка"):
        qrep = quality.assess(image)

    # --- распознавание ---
    try:
        vres = vision.analyze(
            image,
            {
                "zone_id": zone_id or "—",
                "road_type_label": config.ROAD_TYPES[road_type],
                "weather_label": config.WEATHER[weather],
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Модель недоступна: {type(exc).__name__}: {exc}") from exc

    det = vres.detection

    # --- приоритет ---
    with Stage("расчёт приоритета"):
        pres = priority.compute(
            defect_type=det.defect_type,
            severity=det.severity,
            model_confidence=det.confidence,
            quality_score=qrep.score,
            road_type=road_type,
            traffic_level=traffic_level,
            weather=weather,
            context_flags=flags,
        )

    # --- снимок с рамкой ---
    with Stage("отрисовка и сохранение снимка"):
        report_id = store.next_id()
        annotated = render.draw_bbox(
            image, det.bbox, pres.level, f"{pres.level.upper()} {pres.score:.1f}"
        ) if det.bbox else image
        if not imwrite_safe(store.IMAGES / f"{report_id}.jpg", annotated):
            log.warning("Не удалось сохранить снимок для %s", report_id)

    payload = {
        "report_id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detection": {
            **det.model_dump(exclude={"image_issues", "privacy_flags"}),
            "defect_label": config.DEFECT_TYPES.get(det.defect_type, det.defect_type),
        },
        "priority": {
            "level": pres.level,
            "label": pres.label,
            "score": pres.score,
            "sla_days": pres.sla_days,
            "final_confidence": pres.final_confidence,
            "needs_manual_review": pres.needs_manual_review,
            "review_reason": pres.review_reason,
            "factors": [f.__dict__ for f in pres.factors],
        },
        "quality": qrep.__dict__,
        "context": {
            "zone_id": zone_id or "—",
            "zone_name": zone_name,
            "road_type": road_type,
            "road_type_label": config.ROAD_TYPES[road_type],
            "traffic_level": traffic_level,
            "weather": weather,
            "context_flags": flags,
            "lat": lat,
            "lon": lon,
        },
        "privacy_blurred": sorted(set(blurred)),
        "image_url": f"/api/reports/{report_id}/image",
        "model": vres.model,
        "from_cache": vres.from_cache,
        "processing_ms": int((time.perf_counter() - started) * 1000),
    }

    with Stage("сохранение заявки"):
        if save:
            store.save(payload)

    with Stage("формирование ответа"):
        return AnalyzeResponse(**payload)


@app.post("/api/what-if", response_model=PriorityOut, tags=["анализ"],
          summary="Пересчитать приоритет того же дефекта в другом контексте")
def what_if(req: WhatIfRequest) -> PriorityOut:
    """
    Демонстрирует ключевую идею системы: один и тот же дефект получает разный
    приоритет в зависимости от места. Модель повторно НЕ вызывается —
    пересчёт мгновенный и бесплатный.
    """
    r = store.get(req.report_id)
    if r is None:
        raise HTTPException(404, "Заявка не найдена")

    if req.road_type not in config.ROAD_TYPES:
        raise HTTPException(422, f"Неизвестный тип дороги: {req.road_type}")

    ctx = r["context"]
    pres = priority.compute(
        defect_type=r["detection"]["defect_type"],
        severity=r["detection"]["severity"],
        model_confidence=r["detection"]["confidence"],
        quality_score=r["quality"]["score"],
        road_type=req.road_type,
        traffic_level=req.traffic_level or ctx["traffic_level"],
        weather=req.weather or ctx["weather"],
        context_flags=req.context_flags if req.context_flags is not None else [],
    )
    return PriorityOut(
        level=pres.level, label=pres.label, score=pres.score, sla_days=pres.sla_days,
        final_confidence=pres.final_confidence, needs_manual_review=pres.needs_manual_review,
        review_reason=pres.review_reason, factors=[f.__dict__ for f in pres.factors],
    )


# ---------------------------------------------------------------------------
# Очередь ремонта
# ---------------------------------------------------------------------------
def _summary(r: dict) -> ReportSummary:
    return ReportSummary(
        report_id=r["report_id"],
        created_at=r["created_at"],
        zone_id=r["context"]["zone_id"],
        zone_name=r["context"]["zone_name"],
        road_type_label=r["context"]["road_type_label"],
        defect_type=r["detection"]["defect_type"],
        defect_label=r["detection"]["defect_label"],
        priority_level=r["priority"]["level"],
        priority_label=r["priority"]["label"],
        score=r["priority"]["score"],
        sla_days=r["priority"]["sla_days"],
        due_date=store.due_date(r["created_at"], r["priority"]["sla_days"]),
        confidence=r["priority"]["final_confidence"],
        needs_manual_review=r["priority"]["needs_manual_review"],
        status=r.get("status", "new"),
        image_url=r["image_url"],
        lat=r["context"]["lat"],
        lon=r["context"]["lon"],
    )


@app.get("/api/reports", response_model=list[ReportSummary], tags=["очередь ремонта"],
         summary="Очередь ремонта, отсортированная по приоритету")
def list_reports(
    level: str | None = None,
    status: str | None = None,
    only_defects: bool = True,
) -> list[ReportSummary]:
    rs = store.all_reports()
    if only_defects:
        rs = [r for r in rs if r["detection"]["defect_type"] != "none"]
    if level:
        rs = [r for r in rs if r["priority"]["level"] == level]
    if status:
        rs = [r for r in rs if r.get("status", "new") == status]
    return [_summary(r) for r in rs]


@app.get("/api/reports/stats", tags=["очередь ремонта"], summary="Сводка по очереди")
def report_stats() -> dict:
    return store.stats()


@app.get("/api/reports/{report_id}", response_model=AnalyzeResponse, tags=["очередь ремонта"],
         summary="Полная карточка заявки")
def get_report(report_id: str) -> AnalyzeResponse:
    r = store.get(report_id)
    if r is None:
        raise HTTPException(404, "Заявка не найдена")
    return AnalyzeResponse(**r)


@app.get("/api/reports/{report_id}/image", tags=["очередь ремонта"],
         summary="Снимок с отрисованной рамкой дефекта")
def get_image(report_id: str):
    path = store.IMAGES / f"{report_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "Изображение не найдено")
    return FileResponse(path, media_type="image/jpeg")


@app.patch("/api/reports/{report_id}/status", response_model=ReportSummary, tags=["очередь ремонта"],
           summary="Изменить статус заявки")
def update_status(report_id: str, body: StatusUpdate) -> ReportSummary:
    try:
        r = store.set_status(report_id, body.status)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if r is None:
        raise HTTPException(404, "Заявка не найдена")
    return _summary(r)


@app.get("/api/export.csv", tags=["очередь ремонта"], summary="Выгрузить наряд-задание в CSV")
def export_csv() -> StreamingResponse:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["ID", "Дата", "Участок", "Адрес", "Тип дороги", "Дефект",
                "Приоритет", "Балл", "Уверенность", "Срок до", "Ручная проверка", "Статус"])
    for r in store.all_reports():
        if r["detection"]["defect_type"] == "none":
            continue
        s = _summary(r)
        w.writerow([s.report_id, s.created_at.strftime("%Y-%m-%d %H:%M"), s.zone_id,
                    s.zone_name or "—", s.road_type_label, s.defect_label, s.priority_label,
                    f"{s.score:.2f}", f"{s.confidence:.2f}", s.due_date,
                    "да" if s.needs_manual_review else "нет", s.status])
    data = "﻿" + buf.getvalue()          # BOM, чтобы Excel не ломал кириллицу
    return StreamingResponse(
        io.BytesIO(data.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="repair_queue.csv"'},
    )


@app.delete("/api/reports", tags=["очередь ремонта"], summary="Очистить очередь (для демо)")
def clear_reports() -> dict:
    return {"deleted": store.delete_all()}


# ---------------------------------------------------------------------------
# Статика
# ---------------------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "dashboard.html")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse("/static/index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR, html=True), name="static")
