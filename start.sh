#!/usr/bin/env bash
# RoadScan — запуск веб-сервиса (macOS / Linux)
set -e
cd "$(dirname "$0")"

echo
echo "================================================"
echo "   RoadScan — запуск веб-сервиса"
echo "================================================"
echo

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
    echo "[ОШИБКА] Python не найден. Установите Python 3.12:"
    echo "   macOS:  brew install python@3.12"
    echo "   Ubuntu: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[1/4] Создаю виртуальное окружение..."
    "$PY" -m venv .venv
else
    echo "[1/4] Виртуальное окружение готово."
fi

VPY=".venv/bin/python"

if [ ! -f ".venv/.installed" ]; then
    echo "[2/4] Устанавливаю зависимости. Это займёт 1-3 минуты..."
    "$VPY" -m pip install --upgrade pip --quiet
    "$VPY" -m pip install -r requirements.txt
    touch .venv/.installed
else
    echo "[2/4] Зависимости уже установлены."
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo
    echo "------------------------------------------------"
    echo " ВНИМАНИЕ: создан файл .env"
    echo " Впишите в него ключ: GEMINI_API_KEY=ваш_ключ"
    echo " Ключ бесплатно: https://aistudio.google.com/apikey"
    echo "------------------------------------------------"
    echo
fi

echo "[3/4] Проверяю логику приоритета..."
"$VPY" selftest.py

echo
echo "[4/4] Запускаю сервис..."
echo
echo "   Страница жителя:  http://localhost:8000/"
echo "   Очередь ремонта:  http://localhost:8000/dashboard"
echo "   Документация API: http://localhost:8000/docs"
echo
echo "   Остановить сервис: Ctrl+C"
echo

# браузер откроется сам через 6 секунд, когда сервер поднимется
( sleep 6
  if command -v open >/dev/null 2>&1; then
      open http://localhost:8000/ >/dev/null 2>&1
  elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open http://localhost:8000/ >/dev/null 2>&1
  fi ) >/dev/null 2>&1 &

exec "$VPY" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
