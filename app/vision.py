"""
Слой распознавания: кэш, выбор провайдера, повторы, переключение вендоров.

Устройство:
  * сам запрос к модели делает провайдер из app/providers — Gemini,
    OpenRouter, Groq, OpenAI, Mistral или любой OpenAI-совместимый;
  * этот модуль отвечает за всё остальное: кэш по SHA-256 снимка, повторы
    при временных сбоях и переход к следующему провайдеру, если первый лёг.

Смена вендора — одна строка в .env, код трогать не нужно. На защите это же
и аргумент: решение не привязано к одному поставщику модели.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config, providers
from .detection import DefectDetection, coerce  # noqa: F401  (ре-экспорт для совместимости)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

log = logging.getLogger("roadscan.vision")

RETRIES = int(os.getenv("AI_RETRIES", "2"))            # попыток на одного провайдера
BACKOFF_BASE = float(os.getenv("AI_BACKOFF", "1.5"))   # пауза: 1.5 с, 3 с, 6 с

# Ошибки, которые проходят сами: перегрузка, лимиты, обрыв связи
_TRANSIENT = ("UNAVAILABLE", "503", "RESOURCE_EXHAUSTED", "429", "INTERNAL",
              "500", "502", "DEADLINE_EXCEEDED", "504", "TIMEOUT", "OVERLOADED",
              "CONNECTION", "READ TIMED OUT")
# Ошибки, при которых повторять бессмысленно
_FATAL = ("API_KEY", "PERMISSION_DENIED", "UNAUTHENTICATED", "401", "403", "INVALID_API")


@dataclass
class VisionResult:
    detection: DefectDetection
    from_cache: bool
    model: str
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["detection"] = self.detection.model_dump()
        return d


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------
def resize_for_model(image_bgr: np.ndarray) -> np.ndarray:
    """Ресайз до MAX_IMAGE_SIDE. Ускоряет ответ в 2-4 раза, точность не падает."""
    h, w = image_bgr.shape[:2]
    scale = config.MAX_IMAGE_SIDE / max(h, w)
    if scale >= 1.0:
        return image_bgr
    return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _cache_key(jpeg_bytes: bytes, context: dict) -> str:
    h = hashlib.sha256(jpeg_bytes)
    h.update(json.dumps(context, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:24]


def _is_transient(exc: Exception) -> bool:
    s = str(exc).upper()
    return any(t in s for t in _TRANSIENT)


def _is_fatal(exc: Exception) -> bool:
    s = str(exc).upper()
    return any(t in s for t in _FATAL)


# ---------------------------------------------------------------------------
# Основной вызов
# ---------------------------------------------------------------------------
def analyze(image_bgr: np.ndarray, context: dict, use_cache: bool = True) -> VisionResult:
    """
    context: {"zone_id", "road_type_label", "weather_label"} — справка для модели.
    """
    small = resize_for_model(image_bgr)
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Не удалось закодировать изображение в JPEG")
    jpeg_bytes = buf.tobytes()

    cache_file = CACHE_DIR / f"{_cache_key(jpeg_bytes, context)}.json"

    if use_cache and cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return VisionResult(coerce(data["detection"]), True, data.get("model", "cache"))

    if os.getenv("OFFLINE_MODE") == "1":
        return VisionResult(
            DefectDetection(defect_type="none", severity=0, confidence=0.0,
                            description="Офлайн-режим: нет кэша для этого снимка."),
            False, "offline", error="OFFLINE_MODE=1 и снимок отсутствует в кэше",
        )

    order = providers.chain()
    if not order:
        raise RuntimeError(
            "Не настроен ни один провайдер. Откройте .env и задайте AI_PROVIDER "
            "(например openrouter) вместе с соответствующим ключом."
        )

    errors: list[str] = []

    for name in order:
        try:
            provider = providers.build(name)
        except Exception as exc:  # noqa: BLE001 — нет ключа или кривая настройка
            errors.append(f"{name}: {exc}")
            continue

        for attempt in range(RETRIES):
            try:
                detection, model_used = provider.analyze(jpeg_bytes, context)
                cache_file.write_text(
                    json.dumps({"detection": detection.model_dump(), "model": model_used},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if errors or attempt:
                    log.info("Ответ получен от %s (попытка %d)", model_used, attempt + 1)
                return VisionResult(detection, False, model_used)

            except Exception as exc:  # noqa: BLE001
                msg = str(exc)[:200]
                if _is_fatal(exc):
                    errors.append(f"{name}: проблема с ключом или доступом — {msg}")
                    break
                if _is_transient(exc) and attempt < RETRIES - 1:
                    pause = BACKOFF_BASE * (2 ** attempt)
                    log.warning("%s недоступен (%s). Повтор через %.1f с…", name, msg[:90], pause)
                    time.sleep(pause)
                    continue
                errors.append(f"{name}: {msg}")
                log.warning("%s не ответил: %s. Перехожу к следующему провайдеру…", name, msg[:120])
                break

    raise RuntimeError(
        "Ни один провайдер не ответил.\n" + "\n".join(f"  • {e}" for e in errors)
        + "\n\nЧто можно сделать: подождать минуту и повторить; "
        "задать в .env другой AI_PROVIDER; или загрузить снимок, который уже "
        "анализировался раньше — он возьмётся из кэша."
    )
