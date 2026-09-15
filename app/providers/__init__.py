"""
Реестр провайдеров распознавания.

Провайдер выбирается переменной AI_PROVIDER в .env:
    openrouter | gemini | groq | openai | mistral | custom

AI_PROVIDER_FALLBACKS задаёт запасные через запятую — если основной лежит,
сервис автоматически переходит к следующему. Именно это спасает демонстрацию,
когда у одного вендора внезапно «high demand».
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("roadscan.provider")

OPENAI_COMPATIBLE = ("openrouter", "groq", "openai", "mistral", "custom")
KNOWN = ("local", "gemini") + OPENAI_COMPATIBLE


def build(name: str):
    """Создать провайдера по имени. Бросает RuntimeError, если нет ключа."""
    name = name.strip().lower()
    if name == "local":
        from .local_detector import LocalDetectorProvider
        return LocalDetectorProvider()
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider()
    if name in OPENAI_COMPATIBLE:
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(name)
    raise ValueError(f"Неизвестный провайдер «{name}». Доступны: {', '.join(KNOWN)}")


def chain() -> list[str]:
    """Порядок провайдеров: основной, затем запасные, затем все с ключами."""
    primary = os.getenv("AI_PROVIDER", "").strip().lower()
    fallbacks = [p.strip().lower() for p in os.getenv("AI_PROVIDER_FALLBACKS", "").split(",") if p.strip()]

    order = [p for p in [primary, *fallbacks] if p in KNOWN]

    # Если явно ничего не задано — берём локальный детектор (если есть веса),
    # затем всех, у кого найден ключ в окружении
    if not order:
        try:
            from .local_detector import LocalDetectorProvider
            LocalDetectorProvider._find_model()
            order.append("local")
        except Exception:  # noqa: BLE001
            pass
        env_key = {
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }
        order += [p for p, k in env_key.items() if os.getenv(k, "").strip()]

    seen: set[str] = set()
    return [p for p in order if not (p in seen or seen.add(p))]


def available() -> dict[str, bool]:
    """Какие провайдеры настроены — для страницы диагностики."""
    env_key = {
        "openrouter": "OPENROUTER_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "custom": "CUSTOM_API_KEY",
    }
    out = {p: bool(os.getenv(k, "").strip()) for p, k in env_key.items()}
    try:
        from .local_detector import LocalDetectorProvider
        out["local"] = bool(LocalDetectorProvider._find_model())
    except Exception:  # noqa: BLE001 — модели нет, это нормально
        out["local"] = False
    return out
