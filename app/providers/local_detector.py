"""
Локальный детектор дорожных повреждений (ONNX или PyTorch).

Зачем он нужен помимо облачных моделей:
  * работает без интернета и без квот — на демонстрации не бывает «503»;
  * специализированная модель, обученная именно на дорожных дефектах,
    а не общий VLM: выше точность и меньше ложных срабатываний;
  * ответ за 50-200 мс на обычном ноутбуке.

Чего он не делает: не пишет человеческое объяснение. Поэтому описание
и признаки формируются здесь детерминированно из геометрии рамок, а
приоритет по-прежнему считает движок в app/priority.py.

Поддерживает любые веса с таксономией RDD2022 — CamThink, YOLOv8/v11/v12,
обученные на Crowdsensing-based Road Damage Detection Challenge.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from ..detection import DefectDetection

log = logging.getLogger("roadscan.provider")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = ROOT / "models"

# --------------------------------------------------------------------------
# Сопоставление классов детектора с нашими типами дефектов.
# Ключ ищется как подстрока в имени класса, поэтому работает и с кодами
# RDD (D00, D40), и с текстовыми названиями (pothole, longitudinal crack).
# --------------------------------------------------------------------------
CLASS_MAP: list[tuple[str, str]] = [
    ("d40", "pothole"),
    ("pothole", "pothole"),
    ("выбоин", "pothole"),
    ("d20", "crack"),          # сетка трещин (alligator)
    ("alligator", "crack"),
    ("d00", "crack"),          # продольная
    ("longitudinal", "crack"),
    ("d10", "crack"),          # поперечная
    ("transverse", "crack"),
    ("crack", "crack"),
    ("трещин", "crack"),
    ("d30", "edge_break"),
    ("edge", "edge_break"),
    ("кромк", "edge_break"),
    ("rut", "rutting"),
    ("d01", "rutting"),
    ("d11", "rutting"),
    ("колея", "rutting"),
    ("repair", "patch_failure"),
    ("patch", "patch_failure"),
    ("заплат", "patch_failure"),
]

# Человеческие названия классов детектора для поля evidence
RDD_LABELS = {
    "d00": "продольная трещина (D00)",
    "d10": "поперечная трещина (D10)",
    "d20": "сетка трещин (D20)",
    "d40": "яма (D40)",
    "d30": "разрушение кромки (D30)",
    "repair": "ранее отремонтированный участок",
}

# Базовая тяжесть по типу и насколько сильно на неё влияет площадь дефекта
SEVERITY_BASE = {"pothole": 2.5, "crack": 1.2, "rutting": 2.0,
                 "edge_break": 1.8, "patch_failure": 1.5, "none": 0.0}
SEVERITY_AREA_GAIN = {"pothole": 26.0, "crack": 14.0, "rutting": 18.0,
                      "edge_break": 16.0, "patch_failure": 12.0, "none": 0.0}
# Сетка трещин опаснее одиночной — усиливаем отдельно
ALLIGATOR_BONUS = 1.0


class LocalDetectorProvider:
    """Инференс YOLO-подобного детектора на onnxruntime или ultralytics."""

    name = "local"

    def __init__(self) -> None:
        self.conf_threshold = float(os.getenv("LOCAL_CONF", "0.30"))
        self.iou_threshold = float(os.getenv("LOCAL_IOU", "0.45"))
        self.model_path = self._find_model()
        self.backend = "onnx" if self.model_path.suffix.lower() == ".onnx" else "torch"
        self._session = None
        self._yolo = None
        self._names: dict[int, str] = {}
        self._input_size = int(os.getenv("LOCAL_INPUT_SIZE", "640"))

    # ------------------------------------------------------------------
    @staticmethod
    def _find_model() -> Path:
        explicit = os.getenv("LOCAL_MODEL_PATH", "").strip()
        if explicit:
            p = Path(explicit)
            if not p.is_absolute():
                p = ROOT / p
            if not p.exists():
                raise RuntimeError(f"Файл модели не найден: {p}")
            return p

        DEFAULT_MODEL_DIR.mkdir(exist_ok=True)
        found = sorted(DEFAULT_MODEL_DIR.glob("*.onnx")) + sorted(DEFAULT_MODEL_DIR.glob("*.pt"))
        if not found:
            raise RuntimeError(
                f"Положите файл модели (.onnx или .pt) в папку {DEFAULT_MODEL_DIR.name}/ "
                "либо укажите путь в LOCAL_MODEL_PATH в файле .env. "
                "См. models/README.md — там ссылки, откуда скачать."
            )
        return found[0]

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._session is not None or self._yolo is not None:
            return

        if self.backend == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "Не установлен onnxruntime. Выполните: pip install onnxruntime"
                ) from exc

            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            inp = self._session.get_inputs()[0]
            shape = inp.shape
            if isinstance(shape[-1], int) and shape[-1] > 0:
                self._input_size = int(shape[-1])
            self._input_name = inp.name

            # Имена классов часто лежат в метаданных ONNX, положенных ultralytics
            meta = self._session.get_modelmeta().custom_metadata_map or {}
            raw = meta.get("names", "")
            if raw:
                try:
                    import ast
                    parsed = ast.literal_eval(raw)
                    self._names = {int(k): str(v) for k, v in parsed.items()}
                except Exception:  # noqa: BLE001
                    pass
        else:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Для весов .pt нужен ultralytics: pip install ultralytics. "
                    "Либо используйте .onnx — он легче и не тянет PyTorch."
                ) from exc
            self._yolo = YOLO(str(self.model_path))
            self._names = dict(self._yolo.names)

        if not self._names:
            env_names = os.getenv("LOCAL_CLASS_NAMES", "").strip()
            if env_names:
                self._names = {i: n.strip() for i, n in enumerate(env_names.split(","))}
            else:
                # Порядок классов RDD2022 по умолчанию
                self._names = {0: "D00", 1: "D10", 2: "D20", 3: "D40", 4: "Repair"}
            log.info("Имена классов взяты по умолчанию: %s", list(self._names.values()))

    # ------------------------------------------------------------------
    def analyze(self, jpeg_bytes: bytes, context: dict, timeout: float = 90.0
                ) -> tuple[DefectDetection, str]:
        self._load()

        image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Не удалось декодировать изображение для детектора")

        boxes = self._infer_torch(image) if self._yolo is not None else self._infer_onnx(image)
        detection = self._to_detection(boxes, image.shape[:2])
        return detection, f"local:{self.model_path.name}"

    # ------------------------------------------------------------------
    def _infer_torch(self, image: np.ndarray) -> list[dict]:
        res = self._yolo.predict(image, conf=self.conf_threshold, iou=self.iou_threshold,
                                 verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append({"cls": int(b.cls[0]), "conf": float(b.conf[0]),
                        "xyxy": (x1, y1, x2, y2)})
        return out

    def _infer_onnx(self, image: np.ndarray) -> list[dict]:
        h0, w0 = image.shape[:2]
        size = self._input_size

        # letterbox: вписываем с сохранением пропорций
        gain = min(size / h0, size / w0)
        nh, nw = int(round(h0 * gain)), int(round(w0 * gain))
        pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(
            image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self._session.run(None, {self._input_name: blob})[0]

        pred = np.squeeze(raw)
        if pred.ndim != 2:
            raise RuntimeError(f"Неожиданная форма выхода модели: {raw.shape}")
        # приводим к (кандидаты, 4 + классы)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        n_cls = pred.shape[1] - 4
        if n_cls < 1:
            raise RuntimeError(f"В выходе модели нет классов: {pred.shape}")

        scores = pred[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]
        keep = confs >= self.conf_threshold
        if not keep.any():
            return []

        pred, cls_ids, confs = pred[keep], cls_ids[keep], confs[keep]
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - bw / 2 - pad_x) / gain
        y1 = (cy - bh / 2 - pad_y) / gain
        x2 = (cx + bw / 2 - pad_x) / gain
        y2 = (cy + bh / 2 - pad_y) / gain

        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idx = cv2.dnn.NMSBoxes(rects, confs.astype(float).tolist(),
                               self.conf_threshold, self.iou_threshold)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()

        out = []
        for i in idx:
            out.append({
                "cls": int(cls_ids[i]),
                "conf": float(confs[i]),
                "xyxy": (float(np.clip(x1[i], 0, w0)), float(np.clip(y1[i], 0, h0)),
                         float(np.clip(x2[i], 0, w0)), float(np.clip(y2[i], 0, h0))),
            })
        return out

    # ------------------------------------------------------------------
    def _to_detection(self, boxes: list[dict], shape: tuple[int, int]) -> DefectDetection:
        h, w = shape

        if not boxes:
            return DefectDetection(
                defect_type="none", severity=0, confidence=0.85,
                location_text="—",
                description="Детектор не обнаружил дефектов покрытия на снимке.",
                evidence=[f"модель {self.model_path.name} не нашла объектов "
                          f"с уверенностью выше {self.conf_threshold:.2f}"],
            )

        enriched = []
        for b in boxes:
            raw_name = str(self._names.get(b["cls"], f"class_{b['cls']}"))
            dtype = _map_class(raw_name)
            x1, y1, x2, y2 = b["xyxy"]
            rel_area = max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h)
            sev = SEVERITY_BASE.get(dtype, 1.0) + SEVERITY_AREA_GAIN.get(dtype, 10.0) * rel_area
            if "d20" in raw_name.lower() or "alligator" in raw_name.lower():
                sev += ALLIGATOR_BONUS
            enriched.append({**b, "raw_name": raw_name, "type": dtype,
                             "rel_area": rel_area, "sev": min(5.0, sev)})

        enriched = [e for e in enriched if e["type"] != "none"]
        if not enriched:
            return DefectDetection(
                defect_type="none", severity=0, confidence=0.8,
                description="Обнаруженные объекты не относятся к дефектам покрытия.",
            )

        # главный дефект: по тяжести, при равенстве — по уверенности
        main = max(enriched, key=lambda e: (e["sev"], e["conf"]))
        x1, y1, x2, y2 = main["xyxy"]

        bbox = [int(y1 / h * 1000), int(x1 / w * 1000),
                int(y2 / h * 1000), int(x2 / w * 1000)]

        others = sorted({e["type"] for e in enriched if e is not main})
        readable = _readable(main["raw_name"])

        evidence = [
            f"детектор дорожных повреждений: {readable}, уверенность {main['conf']:.2f}",
            f"дефект занимает {main['rel_area'] * 100:.1f}% кадра",
        ]
        if len(enriched) > 1:
            evidence.append(f"всего на снимке обнаружено дефектов: {len(enriched)}")
        evidence.append(f"тяжесть {round(main['sev'])}/5 рассчитана по типу и площади дефекта")

        return DefectDetection(
            defect_type=main["type"],
            severity=int(round(main["sev"])),
            confidence=round(float(main["conf"]), 3),
            bbox=bbox,
            location_text=_position(x1, y1, x2, y2, w, h),
            description=(
                f"Обнаружен дефект: {readable}. "
                f"Занимает около {main['rel_area'] * 100:.1f}% площади кадра"
                + (f", всего дефектов на снимке: {len(enriched)}." if len(enriched) > 1 else ".")
            ),
            evidence=evidence,
            secondary_defects=others,
        )


# --------------------------------------------------------------------------
def _map_class(raw_name: str) -> str:
    low = raw_name.lower()
    for needle, dtype in CLASS_MAP:
        if needle in low:
            return dtype
    return "none"


def _readable(raw_name: str) -> str:
    low = raw_name.lower()
    for code, label in RDD_LABELS.items():
        if code in low:
            return label
    return raw_name


def _position(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    horiz = "левая часть полосы" if cx < 0.38 else ("правая часть полосы" if cx > 0.62 else "центр полосы")
    vert = "ближний план" if cy > 0.62 else ("дальний план" if cy < 0.38 else "средний план")
    return f"{horiz}, {vert}"
