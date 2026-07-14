# Image สำหรับ serving API บน Cloud Run
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ต้อง copy src/ ด้วย — pickle ของโมเดลอ้างคลาส src.preprocess.SimplePreprocessor
# (ไม่มี src/ = โหลดโมเดลพังทันทีด้วย ModuleNotFoundError)
COPY src/ src/
COPY api/ api/

ENV PYTHONBUFFERED=1

CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:8080}
