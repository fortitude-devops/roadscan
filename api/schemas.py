"""Контракты REST API. Именно они видны в Swagger по адресу /docs."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FactorOut(BaseModel):
    name: str = Field(description="Название фактора")
    kind: str = Field(description="base | multiplier | bonus | cap")
    value: float = Field(description="Множитель или прибавка")
    contribution: float = Field(description="Фактический вклад в баллы")
    explanation: str = Field(description="Формулировка для человека")


class QualityOut(BaseModel):
    score: float = Field(description="Пригодность снимка 0..1")
    blur_var: float
    brightness: float
    width: int
    height: int
    issues: list[str] = []


class DetectionOut(BaseModel):
    defect_type: str = Field(description="none|pothole|crack|rutting|edge_break|patch_failure")
    defect_label: str = Field(description="Человекочитаемое название дефекта")
    severity: int = Field(description="Тяжесть дефекта 1-5 по снимку")
    confidence: float = Field(description="Уверенность модели в классе")
    bbox: list[int] = Field(default_factory=list, description="[ymin,xmin,ymax,xmax] в шкале 0-1000")
    location_text: str
    description: str
    evidence: list[str] = []
    secondary_defects: list[str] = []


class PriorityOut(BaseModel):
    level: str = Field(description="low | medium | high")
    label: str
    score: float
    sla_days: int = Field(description="Нормативный срок реакции, дней")
    final_confidence: float = Field(description="confidence модели × качество снимка")
    needs_manual_review: bool
    review_reason: str | None = None
    factors: list[FactorOut] = []


class ContextOut(BaseModel):
    zone_id: str
    zone_name: str | None = None
    road_type: str
    road_type_label: str
    traffic_level: str
    weather: str
    context_flags: list[str] = []
    lat: float | None = None
    lon: float | None = None


class AnalyzeResponse(BaseModel):
    """Полный результат анализа одного снимка."""
    report_id: str
    created_at: datetime
    detection: DetectionOut
    priority: PriorityOut
    quality: QualityOut
    context: ContextOut
    privacy_blurred: list[str] = Field(default_factory=list, description="Что анонимизировано в кадре")
    image_url: str = Field(description="Снимок с отрисованной рамкой дефекта")
    model: str
    from_cache: bool
    processing_ms: int


class ReportSummary(BaseModel):
    """Строка очереди ремонта."""
    report_id: str
    created_at: datetime
    zone_id: str
    zone_name: str | None
    road_type_label: str
    defect_type: str
    defect_label: str
    priority_level: str
    priority_label: str
    score: float
    sla_days: int
    due_date: str
    confidence: float
    needs_manual_review: bool
    status: str = Field(description="new | in_progress | done | rejected")
    image_url: str
    lat: float | None = None
    lon: float | None = None


class StatusUpdate(BaseModel):
    status: str = Field(description="new | in_progress | done | rejected")


class ZoneOut(BaseModel):
    zone_id: str
    zone_name: str
    road_type: str
    road_type_label: str
    traffic_level: str
    lat: float
    lon: float
    context_flags: list[str] = []


class WhatIfRequest(BaseModel):
    """Пересчёт приоритета того же дефекта в другом контексте — без повторного вызова модели."""
    report_id: str
    road_type: str
    traffic_level: str | None = None
    weather: str | None = None
    context_flags: list[str] | None = None


class DictionariesOut(BaseModel):
    defect_types: dict[str, str]
    road_types: dict[str, str]
    traffic_levels: dict[str, str]
    weather: dict[str, str]
    context_flags: dict[str, str]
    priority_labels: dict[str, str]
    priority_colors: dict[str, str]
    thresholds: dict[str, float]
