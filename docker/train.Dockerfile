# Image สำหรับรันเทรนบน Vertex AI Custom Job
# python 3.11-slim: เบา + ทุก lib ใน requirements มี wheel รองรับ
# (เครื่อง dev ใช้ 3.10 ได้ตามปกติ — pickle จากเครื่อง dev ไม่เคยถูก deploy
#  เพราะโมเดล production ทุกตัวเกิดจาก image นี้เท่านั้น)
FROM python:3.11-slim
# libgomp1 = OpenMP runtime ที่ lightgbm/xgboost ต้องใช้ (ไม่มี = ImportError ตอนรัน)
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง dependencies ก่อน copy โค้ด → docker cache layer นี้ไว้
# แก้โค้ดกี่ครั้งก็ไม่ต้องลง lib ใหม่ (build เร็วขึ้นมาก)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV PYTHONUNBUFFERED=1
# container ไม่มี git → mlflow autolog พยายามอ่าน git SHA แล้ว spam warning ยาว ๆ
# เรา capture git_sha ผ่าน env GIT_SHA เองอยู่แล้ว (ดู _git_sha) → ปิด mlflow git ให้เงียบ
ENV GIT_PYTHON_REFRESH=quiet

# รันเป็น module จาก /app เสมอ → pickle อ้างคลาสเป็น src.preprocess.* (ดู README §4)
ENTRYPOINT ["python", "-m", "src.train"]
# default: ลงทะเบียน best model เป็น @challenger (Vertex override args ได้)
CMD ["--register"]