"""
Проверка файла модели: та ли это модель и подойдёт ли она сервису.

    python check_model.py models\8542745...pt

Показывает формат, классы, размер входа и выносит вердикт. Для .pt не требует
PyTorch: если ultralytics не установлен, имена классов вынимаются прямо из
архива чекпоинта.
"""
from __future__ import annotations

import ast
import re
import sys
import zipfile
from pathlib import Path

# Что мы ожидаем увидеть у модели под наш кейс
RDD_MARKERS = {
    "D00": "продольная трещина",
    "D10": "поперечная трещина",
    "D20": "сетка трещин",
    "D40": "яма",
    "D30": "разрушение кромки",
}
KEYWORDS = {
    "pothole": "яма",
    "longitudinal": "продольная трещина",
    "transverse": "поперечная трещина",
    "alligator": "сетка трещин",
    "crack": "трещина",
    "edge": "кромка",
    "rut": "колея",
    "repair": "отремонтированный участок",
}


def main() -> int:
    if len(sys.argv) < 2:
        print("Использование:  python check_model.py <путь к .pt или .onnx>")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        return 1

    size_mb = path.stat().st_size / 1024 / 1024
    print("=" * 62)
    print(f"Файл:    {path.name}")
    print(f"Размер:  {size_mb:.1f} МБ")
    print(f"Формат:  {path.suffix or 'без расширения'}")
    print("=" * 62)

    if size_mb < 1:
        print("\n⚠️  Меньше мегабайта — скорее всего скачалась HTML-страница, "
              "а не веса. Проверьте ссылку.")
        return 1

    if path.suffix.lower() == ".onnx":
        return check_onnx(path)
    if path.suffix.lower() in (".pt", ".pth"):
        return check_pt(path)

    print("\n⚠️  Неизвестное расширение. Сервису нужен .onnx (лучше) или .pt.")
    print("    Если файл скачан без имени — попробуйте переименовать в .pt и запустить снова.")
    return 1


# ---------------------------------------------------------------------------
def check_onnx(path: Path) -> int:
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n❌ Нет onnxruntime. Установите:  pip install onnxruntime")
        return 1

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Файл не открывается как ONNX: {exc}")
        return 1

    inp = sess.get_inputs()[0]
    print(f"\nВход:    {inp.name}  {inp.shape}")
    for out in sess.get_outputs():
        print(f"Выход:   {out.name}  {out.shape}")

    meta = sess.get_modelmeta().custom_metadata_map or {}
    names = {}
    if meta.get("names"):
        try:
            names = ast.literal_eval(meta["names"])
        except Exception:  # noqa: BLE001
            pass
    if meta.get("imgsz"):
        print(f"imgsz:   {meta['imgsz']}")

    return verdict(names, list(names.values()), path, is_onnx=True)


# ---------------------------------------------------------------------------
def check_pt(path: Path) -> int:
    names: dict = {}

    # Способ 1: ultralytics, если установлен — самый точный
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
        names = dict(model.names)
        print("\nПрочитано через ultralytics ✓")
    except ImportError:
        print("\nultralytics не установлен — читаю классы прямо из архива")
    except Exception as exc:  # noqa: BLE001
        print(f"\n⚠️  ultralytics не смог загрузить файл: {str(exc)[:160]}")

    # Способ 2: вынуть строки из чекпоинта
    found_words: list[str] = []
    if not names:
        if not zipfile.is_zipfile(path):
            print("❌ Это не torch-чекпоинт: файл не является zip-архивом.")
            print("   Возможно, скачался повреждённым — загрузите заново.")
            return 1

        blob = b""
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".pkl"):
                    blob += z.read(n)
        text = blob.decode("latin-1", errors="ignore")

        # словарь имён классов часто лежит как {0: 'D00', 1: 'D10', ...}
        m = re.search(r"\{\s*0\s*:\s*'[^']{1,40}'(?:\s*,\s*\d+\s*:\s*'[^']{1,40}')*\s*\}", text)
        if m:
            try:
                names = ast.literal_eval(m.group(0))
            except Exception:  # noqa: BLE001
                pass

        for code in RDD_MARKERS:
            if re.search(rf"\b{code}\b", text):
                found_words.append(code)
        for kw in KEYWORDS:
            if re.search(kw, text, re.I):
                found_words.append(kw)

        arch = sorted(set(re.findall(r"yolo\w{0,6}", text, re.I)))
        if arch:
            print(f"Архитектура: {', '.join(arch[:6])}")

    print("\n⚠️  Внимание: сервису лучше подходит .onnx — он не тянет PyTorch (~2 ГБ).")
    return verdict(names, list(names.values()) + found_words, path, is_onnx=False)


# ---------------------------------------------------------------------------
def verdict(names: dict, evidence: list[str], path: Path, is_onnx: bool) -> int:
    if names:
        print(f"\nКлассы ({len(names)}):")
        for i, n in sorted(names.items()):
            hint = ""
            for code, ru in RDD_MARKERS.items():
                if code.lower() in str(n).lower():
                    hint = f"  — {ru}"
            if not hint:
                for kw, ru in KEYWORDS.items():
                    if kw in str(n).lower():
                        hint = f"  — {ru}"
                        break
            print(f"   {i}: {n}{hint}")
    else:
        print("\nИмена классов явно не найдены.")

    blob = " ".join(str(e).lower() for e in evidence)
    hits = [k for k in list(RDD_MARKERS) + list(KEYWORDS) if k.lower() in blob]

    print("\n" + "=" * 62)
    if hits:
        print("✅ ПОХОЖЕ НА МОДЕЛЬ ДОРОЖНЫХ ПОВРЕЖДЕНИЙ")
        print(f"   Найдены признаки: {', '.join(sorted(set(hits))[:8])}")
        print("\nЧто делать дальше:")
        if is_onnx:
            print(f"   1. Переложите файл в папку models/ под понятным именем,")
            print(f"      например models/road_damage.onnx")
            print("   2. В .env:  AI_PROVIDER=local")
            print("   3. Перезапустите start.bat")
        else:
            print("   1. Сконвертируйте в ONNX (один раз):")
            print("        .venv\\Scripts\\python -m pip install ultralytics")
            print(f"        .venv\\Scripts\\yolo export model={path} format=onnx imgsz=640")
            print("   2. Полученный .onnx положите в models/")
            print("   3. В .env:  AI_PROVIDER=local")
            print("\n   Либо оставьте .pt как есть — сервис его прочитает,")
            print("   но понадобится ultralytics с PyTorch (~2 ГБ).")
    else:
        print("❓ НЕ УДАЛОСЬ ПОДТВЕРДИТЬ, что это модель дорожных повреждений")
        print("   Классы не содержат ни кодов RDD (D00/D10/D20/D40),")
        print("   ни слов pothole / crack / edge.")
        print("\n   Это может быть модель под другую задачу либо просто другой")
        print("   набор имён классов. Посмотрите список выше: если классы")
        print("   называются по-своему, но по смыслу это дефекты дороги —")
        print("   допишите их в CLASS_MAP в app/providers/local_detector.py")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
