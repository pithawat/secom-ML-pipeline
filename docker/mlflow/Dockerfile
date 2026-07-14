# MLflow tracking server สำหรับ Cloud Run
# ⚠️ tag ต้องตรงกับ mlflow ใน requirements.txt (client/server คนละรุ่นห่างกันมาก = REST เพี้ยน)
FROM ghcr.io/mlflow/mlflow:v3.14.0


# image ทางการไม่มี driver postgres และ lib ของ GCS — ต้องลงเพิ่มเอง
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 google-cloud-storage

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
