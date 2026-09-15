"""Провайдер Google Gemini: нативный SDK со строгой схемой ответа."""
from __future__ import annotations

import json
import logging
import os

from .. import config
from ..detection import DefectDetection, coerce
from ..prompt import SYSTEM_PROMPT, USER_TEMPLATE

log = logging.getLogger("roadscan.provider")

_client = None


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        if not os.getenv("GEMINI_API_KEY", "").strip():
            raise RuntimeError("Для провайдера «gemini» не задан GEMINI_API_KEY в файле .env")

    @staticmethod
    def _get_client():
        global _client
        if _client is None:
            from google import genai
            _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return _client

    def _candidates(self, client) -> list[str]:
        """Основная модель, затем заданные вручную, затем доступные flash аккаунта."""
        out = [config.GEMINI_MODEL]
        out += [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
        try:
            names = [m.name.split("/")[-1] for m in client.models.list()]
            out += sorted(
                [n for n in names if "flash" in n and not any(
                    x in n for x in ("image", "tts", "live", "embedding", "native-audio"))],
                reverse=True,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось получить список моделей Gemini: %s", exc)

        seen: set[str] = set()
        return [n for n in out if n and not (n in seen or seen.add(n))]

    def analyze(self, jpeg_bytes: bytes, context: dict, timeout: float = 90.0) -> tuple[DefectDetection, str]:
        from google.genai import types

        client = self._get_client()
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=DefectDetection,
            temperature=0.1,
            max_output_tokens=1200,
        )
        parts = [
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            USER_TEMPLATE.format(**context),
        ]

        last: Exception | None = None
        for model in self._candidates(client)[:3]:
            try:
                resp = client.models.generate_content(model=model, contents=parts, config=cfg)
                return self._parse(resp), f"gemini:{model}"
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("Gemini %s не ответила: %s", model, str(exc)[:160])
        raise RuntimeError(f"gemini: {last}")

    @staticmethod
    def _parse(response) -> DefectDetection:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, DefectDetection):
            return parsed
        if isinstance(parsed, dict):
            return coerce(parsed)
        return coerce(json.loads(response.text))
