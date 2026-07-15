"""
SECOM fault-detection serving API

ทำไมโหลดโมเดลจาก "release bundle" ใน GCS (MODEL_URI) แทนที่จะถาม MLflow:
- Cloud Run scale-to-zero: ทุก cold start ต้องโหลดโมเดลใหม่ — ถ้าไปพึ่ง MLflow
  (ซึ่งก็ scale-to-zero เหมือนกัน) จะเกิด cold start ซ้อนสองชั้น + เพิ่มจุดพัง
- MODEL_URI ถูก pin ตายตัวกับ 1 revision ของ Cloud Run ตอน deploy
  → รู้แน่นอนว่า revision ไหนใช้โมเดล version อะไร และ rollback ทำได้ด้วย
  การย้าย traffic กลับ revision เก่า (ไม่ต้องยุ่งกับโมเดลเลย)
- MLflow server ล่ม/ถูกลบ ก็ไม่กระทบ API ที่รันอยู่

การที่โมเดลเป็น sklearn Pipeline (preprocessor + classifier) ทำให้ API
รับ "ข้อมูลดิบ 590 ค่า (มี null ได้)" ตรงจากหน้างาน — ไม่ต้องมีโค้ด
preprocessing ซ้ำสองที่ (ถ้าซ้ำเมื่อไหร่ training/serving skew ตามมาแน่)
"""
import json
import os
import time
from contextlib import asynccontextmanager

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.evaluate import predict_positive_proba

MODEL_URI = os.environ.get("MODEL_URI")          # เช่น gs://bucket/releases/secom-fault/v7/model
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # โหลดครั้งเดียวตอน start (fail fast: ถ้าโหลดไม่ได้ให้ container ตายเลย
    # Cloud Run จะถือว่า deploy ล้มเหลว — ดีกว่าปล่อยให้รับ traffic แล้วค่อยพัง)
    if not MODEL_URI:
        raise RuntimeError("ต้องตั้ง env MODEL_URI (เช่น gs://…/releases/secom-fault/v7/model)")
    t0 = time.time()
    model = mlflow.sklearn.load_model(MODEL_URI)
    _state["model"] = model
    # จำนวนฟีเจอร์ดิบที่โมเดลคาดหวัง — อ่านจาก preprocessor ที่ fit มาแล้ว (=590)
    _state["n_features"] = int(len(model.named_steps["pre"].means_))
    print(f"loaded model v{MODEL_VERSION} from {MODEL_URI} in {time.time() - t0:.1f}s")
    yield


app = FastAPI(title="SECOM fault detection API", lifespan=lifespan)

# CORS: ให้หน้า test UI (เปิดจากไฟล์ local / โดเมนอื่น) ยิง /predict ได้
# ไม่งั้น browser บล็อก cross-origin. เดโม/งานเรียนใช้ "*" ได้ — งานจริงล็อกเป็น origin ที่รู้จัก
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    # แต่ละ instance = ค่า sensor 590 ตัว, ใส่ null ได้ (SECOM มี missing values เป็นปกติ
    # — preprocessor ในตัวโมเดลจะ impute ด้วยค่าเฉลี่ยที่เรียนจาก train เอง)
    instances: list[list[float | None]] = Field(min_length=1)


class Prediction(BaseModel):
    fault_probability: float
    is_fault: bool


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # อนุญาต field ชื่อ model_*
    predictions: list[Prediction]
    model_version: str
    threshold: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_uri": MODEL_URI,
        "n_features": _state["n_features"],
        "threshold": settings.decision_threshold,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    n = _state["n_features"]
    for i, row in enumerate(req.instances):
        if len(row) != n:
            raise HTTPException(
                status_code=422,
                detail=f"instance {i}: ต้องมี {n} ค่า แต่ได้ {len(row)}",
            )

    # dtype=float ทำให้ null → NaN, คอลัมน์เป็นเลข 0..589 ตรงกับตอนเทรนพอดี
    X = pd.DataFrame(req.instances, dtype=float)
    proba = predict_positive_proba(_state["model"], X)

    # structured log → Cloud Logging เก็บให้อัตโนมัติ
    # เป็นข้อมูลดิบสำหรับวิเคราะห์ drift ภายหลัง (ดู README §10)
    print(json.dumps({
        "event": "predict",
        "n_instances": len(req.instances),
        "mean_fault_proba": float(np.mean(proba)),
        "model_version": MODEL_VERSION,
    }))

    thr = settings.decision_threshold
    return PredictResponse(
        predictions=[
            Prediction(fault_probability=float(p), is_fault=bool(p >= thr))
            for p in proba
        ],
        model_version=MODEL_VERSION,
        threshold=thr,
    )