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
    --allowed-hosts "${MLFLOW_SERVER_ALLOWED_HOSTS:-*.run.app,localhost,localhost:*,127.0.0.1}" \
    --cors-allowed-origins "${MLFLOW_SERVER_CORS_ALLOWED_ORIGINS:-*}" \
    --workers 1

# --cors-allowed-origins จำเป็นเพราะ MLflow 3.5+ บล็อก cross-origin API ทุก ajax path
# → UI (React) โหลด experiment ไม่ได้ ("Cross-origin request blocked") ถ้าไม่ allow origin
# CORS ไม่รองรับ wildcard subdomain (*.run.app ใช้ไม่ได้เหมือน allowed-hosts) → default เป็น *
# override ให้แคบได้ผ่าน env MLFLOW_SERVER_CORS_ALLOWED_ORIGINS (comma-list ของ origin เต็ม)