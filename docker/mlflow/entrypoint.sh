#!/bin/sh
# ประกอบ backend-store-uri ตอน runtime เพราะ:
# - DB_PASS มาจาก Secret Manager (ฉีดเป็น env ตอน deploy — ห้าม hardcode)
# - ต่อ Cloud SQL ผ่าน unix socket /cloudsql/<instance> ที่ Cloud Run mount ให้
#   (จาก flag --add-cloudsql-instances — ไม่ต้องเปิด public IP ของ DB เลย)
set -e

exec mlflow server \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --backend-store-uri "postgresql+psycopg2://${DB_USER}:${DB_PASS}@/${DB_NAME}?host=/cloudsql/${CLOUDSQL_INSTANCE}" \
    --artifacts-destination "${ARTIFACT_ROOT}" \
    --workers 2