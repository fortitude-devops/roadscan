@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo.
echo ================================================
echo    RoadScan - запуск веб-сервиса
echo ================================================
echo.

REM ---------- 1. Ищем Python ----------
set "PY="
where py >nul 2>&1
if not errorlevel 1 set "PY=py"
if defined PY goto havepy
where python >nul 2>&1
if not errorlevel 1 set "PY=python"
:havepy

if not defined PY (
    echo [ОШИБКА] Python не найден в системе.
    echo.
    echo Установите Python 3.12 отсюда:
    echo     https://www.python.org/downloads/windows/
    echo.
    echo При установке ОБЯЗАТЕЛЬНО поставьте галочку
    echo     "Add python.exe to PATH"
    echo.
    echo Затем закройте это окно и запустите start.bat заново.
    echo.
    pause
    exit /b 1
)

REM ---------- 2. Виртуальное окружение ----------
if exist ".venv\Scripts\python.exe" goto havevenv
echo [1/4] Создаю виртуальное окружение...
%PY% -m venv .venv
if errorlevel 1 goto fail
goto venvdone
:havevenv
echo [1/4] Виртуальное окружение готово.
:venvdone

set "VPY=.venv\Scripts\python.exe"

REM ---------- 3. Зависимости ----------
if exist ".venv\.installed" goto havedeps
echo [2/4] Устанавливаю зависимости. Это займёт 1-3 минуты...
echo.
%VPY% -m pip install --upgrade pip --quiet
%VPY% -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo ok>".venv\.installed"
goto depsdone
:havedeps
echo [2/4] Зависимости уже установлены.
:depsdone

REM ---------- 4. Файл .env ----------
if exist ".env" goto haveenv
copy ".env.example" ".env" >nul
echo.
echo ------------------------------------------------
echo  ВНИМАНИЕ: создан файл .env
echo.
echo  Откройте его Блокнотом и впишите ключ:
echo      GEMINI_API_KEY=ваш_ключ
echo.
echo  Ключ бесплатно: https://aistudio.google.com/apikey
echo.
echo  Без ключа сервис откроется, но анализ фото
echo  вернёт ошибку "Модель недоступна".
echo ------------------------------------------------
echo.
:haveenv

REM ---------- 5. Самопроверка ----------
echo [3/4] Проверяю логику приоритета...
%VPY% selftest.py
if errorlevel 1 goto testfail

REM ---------- 6. Запуск ----------
echo.
echo [4/4] Запускаю сервис...
echo.
echo     Страница жителя:  http://localhost:8000/
echo     Очередь ремонта:  http://localhost:8000/dashboard
echo     Документация API: http://localhost:8000/docs
echo.
echo     Остановить сервис: Ctrl+C
echo.

REM браузер откроется сам через 6 секунд, когда сервер поднимется
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 6; Start-Process 'http://localhost:8000/'"

%VPY% -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
goto end

:testfail
echo.
echo [ОШИБКА] Самопроверка не прошла. Причина в выводе выше.
pause
exit /b 1

:fail
echo.
echo [ОШИБКА] Установка не удалась. Прокрутите вывод выше - там причина.
pause
exit /b 1

:end
echo.
echo Сервис остановлен.
pause
