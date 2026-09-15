"""
Схема ответа модели — общая для всех провайдеров.

Вынесена отдельно, чтобы провайдеры (Gemini, OpenRouter, Groq, OpenAI)
не зависели друг от друга и от слоя кэширования.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DefectDetection(BaseModel):
    defect_type: str = Field(description="none|pothole|crack|rutting|edge_break|patch_failure")
    severity: int = Field(ge=0, le=5, description="Тяжесть дефекта 1-5, либо 0 если дефекта нет")
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в классе дефекта")
    bbox: list[int] = Field(default_factory=list, description="[ymin,xmin,ymax,xmax] в шкале 0-1000, [] если дефекта нет")
    location_text: str = Field(default="", description="Где на снимке дефект, человеческим языком")
    description: str = Field(default="", description="Описание дефекта в 1-2 предложениях")
    evidence: list[str] = Field(default_factory=list, description="Визуальные признаки, обосновывающие вывод")
    secondary_defects: list[str] = Field(default_factory=list, description="Другие замеченные типы дефектов")
    image_issues: list[str] = Field(default_factory=list, description="Проблемы качества снимка, замеченные моделью")
    privacy_flags: list[str] = Field(default_factory=list, description="face|license_plate|house_number, если видны в кадре")


def strict_json_schema() -> dict:
    """
    JSON Schema в строгом виде для OpenAI-совместимых API:
    все поля обязательны, лишние запрещены. Без этого часть провайдеров
    отказывается принимать response_format=json_schema.
    """
    schema = DefectDetection.model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}).keys())
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        prop.pop("default", None)
    return schema


def coerce(raw: dict) -> DefectDetection:
    """
    Привести сырой словарь к схеме, прощая типичные вольности моделей:
    строку вместо числа, float вместо int в bbox, null вместо списка.
    """
    data = dict(raw or {})

    for key in ("evidence", "secondary_defects", "image_issues", "privacy_flags", "bbox"):
        if data.get(key) is None:
            data[key] = []
        elif isinstance(data[key], str):
            data[key] = [data[key]] if key != "bbox" else []

    if isinstance(data.get("bbox"), list):
        try:
            data["bbox"] = [int(round(float(v))) for v in data["bbox"]][:4]
        except (TypeError, ValueError):
            data["bbox"] = []
        if len(data["bbox"]) != 4:
            data["bbox"] = []

    try:
        data["severity"] = max(0, min(5, int(round(float(data.get("severity", 0))))))
    except (TypeError, ValueError):
        data["severity"] = 0

    try:
        data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        data["confidence"] = 0.0

    dtype = str(data.get("defect_type", "none")).strip().lower()
    allowed = {"none", "pothole", "crack", "rutting", "edge_break", "patch_failure"}
    data["defect_type"] = dtype if dtype in allowed else "none"

    for key in ("location_text", "description"):
        data[key] = str(data.get(key) or "")

    return DefectDetection(**{k: v for k, v in data.items() if k in DefectDetection.model_fields})
