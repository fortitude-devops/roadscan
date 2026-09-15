FROM python:3.12-slim

# libgl/libglib нужны opencv даже в headless-сборке
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY api/ ./api/
COPY web/ ./web/
COPY data/ ./data/

RUN mkdir -p storage/images cache

ENV PYTHONUNBUFFERED=1 PORT=8000
EXPOSE 8000

# Render/Railway/Fly подставляют свой $PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
