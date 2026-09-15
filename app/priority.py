"""
Движок приоритета ремонта.

Ключевая идея решения: приоритет считает НЕ языковая модель, а прозрачная
формула. Модель отвечает только за severity дефекта на фото, а место,
трафик и погода добавляет этот модуль. Благодаря этому:
  * один и тот же дефект в разном контексте честно меняет приоритет;
  * можно показать судьям вклад каждого фактора в баллах;
  * результат воспроизводим и не "плавает" от запуска к запуску.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config


@dataclass
class Factor:
    name: str
    kind: str          # "base" | "multiplier" | "bonus" | "cap"
    value: float       # множитель или прибавка
    contribution: float  # фактический вклад в баллы
    explanation: str


@dataclass
class PriorityResult:
    level: str                  # low | medium | high
    label: str
    score: float
    factors: list[Factor] = field(default_factory=list)
    needs_manual_review: bool = False
    review_reason: str | None = None
    sla_days: int = 60
    final_confidence: float = 0.0

    @property
    def color(self) -> str:
        return config.PRIORITY_COLORS[self.level]

    def explanation_lines(self) -> list[str]:
        return [f"{f.explanation} → {f.contribution:+.2f} балла" for f in self.factors]


def compute(
    defect_type: str,
    severity: int,
    model_confidence: float,
    quality_score: float,
    road_type: str,
    traffic_level: str,
    weather: str,
    context_flags: list[str] | None = None,
) -> PriorityResult:
    """Посчитать приоритет и разложить его на объяснимые факторы."""
    context_flags = context_flags or []
    final_confidence = round(model_confidence * quality_score, 3)

    # --- Случай "дефекта нет" -------------------------------------------
    if defect_type == "none" or severity <= 0:
        low_conf = final_confidence < config.CONFIDENCE_REVIEW_THRESHOLD
        return PriorityResult(
            level="low",
            label="Дефект не выявлен",
            score=0.0,
            factors=[Factor("Дефект", "base", 0, 0, "Дефекты покрытия на фото не обнаружены")],
            needs_manual_review=low_conf,
            review_reason=("Качество снимка низкое — отсутствие дефекта не гарантировано"
                           if low_conf else None),
            sla_days=0,
            final_confidence=final_confidence,
        )

    factors: list[Factor] = []

    # --- 1. База от тяжести дефекта -------------------------------------
    score = severity * 2.0
    factors.append(Factor(
        "Тяжесть дефекта", "base", severity, score,
        f"Тяжесть дефекта {severity}/5 по снимку (база = {severity}×2)",
    ))

    # --- 2. Тип дефекта --------------------------------------------------
    w = config.DEFECT_WEIGHT.get(defect_type, 1.0)
    before, score = score, score * w
    factors.append(Factor(
        "Тип дефекта", "multiplier", w, score - before,
        f"{config.DEFECT_TYPES.get(defect_type, defect_type)} (коэффициент ×{w})",
    ))

    # --- 3. Тип дороги ---------------------------------------------------
    w = config.ROAD_TYPE_WEIGHT.get(road_type, 1.0)
    before, score = score, score * w
    factors.append(Factor(
        "Тип дороги", "multiplier", w, score - before,
        f"{config.ROAD_TYPES.get(road_type, road_type)} (коэффициент ×{w})",
    ))

    # --- 4. Интенсивность движения ---------------------------------------
    w = config.TRAFFIC_WEIGHT.get(traffic_level, 1.0)
    before, score = score, score * w
    factors.append(Factor(
        "Трафик", "multiplier", w, score - before,
        f"Интенсивность движения: {config.TRAFFIC_LEVELS.get(traffic_level, traffic_level).lower()} (×{w})",
    ))

    # --- 5. Погода -------------------------------------------------------
    bonus = config.WEATHER_BONUS.get(weather, 0.0)
    if bonus:
        score += bonus
        factors.append(Factor(
            "Погода", "bonus", bonus, bonus,
            f"{config.WEATHER.get(weather, weather)}: вода/лёд скрывают глубину и повышают риск",
        ))

    # --- 6. Контекстные флаги --------------------------------------------
    for flag in context_flags:
        if flag in config.CONTEXT_BONUS:
            label, bonus = config.CONTEXT_BONUS[flag]
            score += bonus
            factors.append(Factor(label, "bonus", bonus, bonus, label))

    # --- 7. Уверенность: ограничитель, а не множитель ---------------------
    needs_review = False
    review_reason = None
    if final_confidence < config.CONFIDENCE_REVIEW_THRESHOLD:
        needs_review = True
        review_reason = (
            f"Итоговая уверенность {final_confidence:.2f} ниже порога "
            f"{config.CONFIDENCE_REVIEW_THRESHOLD}: нужен выезд специалиста для подтверждения"
        )
        capped = min(score, config.PRIORITY_THRESHOLDS["high"] - 0.01)
        if capped < score:
            factors.append(Factor(
                "Ограничение по уверенности", "cap", final_confidence, capped - score,
                "Приоритет ограничен уровнем «средний»: данных недостаточно для высшего приоритета",
            ))
            score = capped

    score = round(score, 2)

    if score >= config.PRIORITY_THRESHOLDS["high"]:
        level = "high"
    elif score >= config.PRIORITY_THRESHOLDS["medium"]:
        level = "medium"
    else:
        level = "low"

    return PriorityResult(
        level=level,
        label=config.PRIORITY_LABELS[level],
        score=score,
        factors=factors,
        needs_manual_review=needs_review,
        review_reason=review_reason,
        sla_days=config.SLA_DAYS[level],
        final_confidence=final_confidence,
    )
