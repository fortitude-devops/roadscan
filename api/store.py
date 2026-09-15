"""
Хранилище заявок.

Потокобезопасный список в памяти + зеркало на диск в JSON, чтобы заявки
пережили перезапуск сервера во время хакатона. Для MVP этого достаточно;
при масштабировании сюда подставляется PostgreSQL без изменения API.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"
IMAGES = STORAGE / "images"
DB_FILE = STORAGE / "reports.json"

STORAGE.mkdir(exist_ok=True)
IMAGES.mkdir(exist_ok=True)

VALID_STATUSES = {"new", "in_progress", "done", "rejected"}

_lock = threading.Lock()
_reports: dict[str, dict] = {}


def _load() -> None:
    if DB_FILE.exists():
        try:
            for r in json.loads(DB_FILE.read_text(encoding="utf-8")):
                _reports[r["report_id"]] = r
        except Exception:  # noqa: BLE001 — битый файл не должен ронять сервис
            pass


def _flush() -> None:
    tmp = DB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(_reports.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DB_FILE)


_load()


def next_id() -> str:
    with _lock:
        return f"ROAD-{len(_reports) + 1:04d}"


def save(report: dict) -> dict:
    report.setdefault("status", "new")
    with _lock:
        _reports[report["report_id"]] = report
        _flush()
    return report


def get(report_id: str) -> dict | None:
    return _reports.get(report_id)


def all_reports() -> list[dict]:
    """Очередь ремонта: сначала высокий балл, затем свежие."""
    return sorted(
        _reports.values(),
        key=lambda r: (-r["priority"]["score"], r["created_at"]),
    )


def set_status(report_id: str, status: str) -> dict | None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Недопустимый статус: {status}")
    with _lock:
        r = _reports.get(report_id)
        if r is None:
            return None
        r["status"] = status
        r["updated_at"] = datetime.now(timezone.utc).isoformat()
        _flush()
        return r


def delete_all() -> int:
    with _lock:
        n = len(_reports)
        _reports.clear()
        _flush()
    for f in IMAGES.glob("*.jpg"):
        f.unlink(missing_ok=True)
    return n


def due_date(created_at: str, sla_days: int) -> str:
    try:
        dt = datetime.fromisoformat(created_at)
    except ValueError:
        dt = datetime.now(timezone.utc)
    return (dt + timedelta(days=sla_days)).date().isoformat()


def stats() -> dict:
    rs = list(_reports.values())
    by_level: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in rs:
        lvl = r["priority"]["level"]
        if r["detection"]["defect_type"] != "none":
            by_level[lvl] = by_level.get(lvl, 0) + 1
    return {
        "total": len(rs),
        "defects": sum(1 for r in rs if r["detection"]["defect_type"] != "none"),
        "by_level": by_level,
        "manual_review": sum(1 for r in rs if r["priority"]["needs_manual_review"]),
        "open": sum(1 for r in rs if r.get("status") in ("new", "in_progress")),
    }
