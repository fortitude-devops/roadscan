"""
Провайдер для любого OpenAI-совместимого API.

Одним классом покрываются OpenRouter, Groq, OpenAI, Mistral, Together,
DeepInfra и локальные серверы вроде Ollama и LM Studio — у всех один и тот же
эндпоинт /chat/completions и один формат передачи картинки.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re

import httpx

from ..detection import DefectDetection, coerce, strict_json_schema
from ..prompt import SYSTEM_PROMPT, USER_TEMPLATE

log = logging.getLogger("roadscan.provider")

# Готовые настройки провайдеров: ключ -> (переменная с API-ключом, base_url, модель)
PRESETS: dict[str, tuple[str, str, str]] = {
    "openrouter": (
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        # Роутер сам выбирает доступную бесплатную модель с поддержкой картинок.
        # Это и есть защита от «503 high demand»: упала одна — возьмётся другая.
        "openrouter/free",
    ),
    "groq": (
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ),
    "openai": (
        "OPENAI_API_KEY",
        "https://api.openai.com/v1",
        "gpt-4o-mini",
    ),
    "mistral": (
        "MISTRAL_API_KEY",
        "https://api.mistral.ai/v1",
        "pixtral-12b-2409",
    ),
    "custom": (
        "CUSTOM_API_KEY",
        "",
        "",
    ),
}


class OpenAICompatProvider:
    """Распознавание через /chat/completions с картинкой в data:-URL."""

    def __init__(self, preset: str) -> None:
        if preset not in PRESETS:
            raise ValueError(f"Неизвестный провайдер: {preset}")
        key_env, base_url, model = PRESETS[preset]

        self.name = preset
        self.api_key = os.getenv(key_env, "").strip()
        self.base_url = os.getenv(f"{preset.upper()}_BASE_URL", base_url).rstrip("/")
        self.model = os.getenv(f"{preset.upper()}_MODEL", model).strip()
        self.key_env = key_env

        if not self.api_key:
            raise RuntimeError(
                f"Для провайдера «{preset}» не задан {key_env} в файле .env"
            )
        if not self.base_url or not self.model:
            raise RuntimeError(
                f"Для провайдера «{preset}» нужно задать {preset.upper()}_BASE_URL "
                f"и {preset.upper()}_MODEL в файле .env"
            )

    # ------------------------------------------------------------------
    def analyze(self, jpeg_bytes: bytes, context: dict, timeout: float = 90.0) -> tuple[DefectDetection, str]:
        b64 = base64.b64encode(jpeg_bytes).decode()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_TEMPLATE.format(**context)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ]

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.name == "openrouter":
            # OpenRouter просит идентифицировать приложение
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "RoadScan"

        base = {"model": self.model, "messages": messages, "temperature": 0.1, "max_tokens": 1200}

        # Сначала строгая JSON-схема; если провайдер её не принимает — режим
        # json_object; если и он не поддерживается — просто просим JSON текстом.
        attempts = [
            {**base, "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "road_defect", "strict": True, "schema": strict_json_schema()},
            }},
            {**base, "response_format": {"type": "json_object"}},
            base,
        ]

        last_error = ""
        with httpx.Client(timeout=timeout) as client:
            for i, payload in enumerate(attempts):
                r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)

                if r.status_code == 200:
                    body = r.json()
                    text = body["choices"][0]["message"]["content"]
                    used = body.get("model", self.model)
                    return coerce(_extract_json(text)), f"{self.name}:{used}"

                last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                # 4xx на response_format — пробуем более простой режим
                if r.status_code in (400, 404, 422) and i < len(attempts) - 1:
                    log.info("%s: режим ответа не принят, пробую проще", self.name)
                    continue
                raise RuntimeError(f"{self.name} ({self.model}): {last_error}")

        raise RuntimeError(f"{self.name} ({self.model}): {last_error}")


def _extract_json(text: str) -> dict:
    """Достать JSON, даже если модель обернула его в ```json или пояснения."""
    if not text:
        raise RuntimeError("Пустой ответ модели")
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise RuntimeError(f"Ответ не содержит JSON: {text[:200]}")
