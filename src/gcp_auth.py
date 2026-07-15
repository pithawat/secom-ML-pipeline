"""
ทำไมต้องมีไฟล์นี้:
MLflow server ของเรา deploy บน Cloud Run แบบ --no-allow-unauthenticated
(ไม่เปิด public — ใครก็ได้บนอินเทอร์เน็ตไม่ควรเขียน experiment เราได้)
=> ทุก request ต้องแนบ Google ID token ที่มี audience = URL ของ service
MLflow client รองรับอยู่แล้วผ่าน env MLFLOW_TRACKING_TOKEN (แนบเป็น Bearer ให้เอง)

ลำดับการหา token (เรียงตาม environment ที่โค้ดนี้จะไปรัน):
1. มี MLFLOW_TRACKING_TOKEN ใน env อยู่แล้ว
   → GitHub Actions ตั้งให้ผ่าน google-github-actions/auth (ดู workflow)
2. metadata server (มีเฉพาะบนเครื่องใน GCP: Vertex AI / Cloud Run / GCE)
   → mint token สดจาก service account ที่ผูกกับ job/service — ไม่มี key ไฟล์ใด ๆ
3. ชี้ MLflow ที่ localhost (dev บนเครื่อง) → ไม่ต้องใช้ token เลย
"""

import os 
import time
import urllib.parse

import requests

import mlflow
from src.config import settings

_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/identity?audience={audience}"
)

_REFRESH_AFTER_SEC = 50 * 60  # ID token อายุ 60 นาที — ต่ออายุก่อนหมดที่ ~50 นาที


_token_minted_at: float | None = None  # None = token มาจากภายนอก เราไม่ยุ่ง lifecycle

def _is_local(uri: str) -> bool:
    host = urllib.parse.urlparse(uri).hostname
    return host in ("127.0.0.1", "localhost")

def _token_from_metadata(audience: str) -> str | None:
    try:
        r = requests.get(
            _METADATA_URL.format(audience=audience),
            headers={"Metadata-Flavor":"Google"},
            timeout=3,
        )
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        return None
    
def ensure_mlflow_auth() -> None:
     """เตรียม MLFLOW_TRACKING_TOKEN ให้พร้อมใช้ — เรียกซ้ำได้เรื่อย ๆ (idempotent)

    งานยาว (เช่น optuna หลายร้อย trial) ควรเรียกทุกครั้งก่อนคุยกับ MLflow:
    ถ้า token ที่ mint เองใกล้หมดอายุจะขอใหม่ให้อัตโนมัติ
    """
     
     global _token_minted_at

     if _is_local(settings.mlflow_uri):
         return # MLflow local ไม่มี auth
     
     have_token = bool(os.environ.get("MLFLOW_TRACKING_TOKEN"))
     if have_token and _token_minted_at is None:
          return  # token จากภายนอก (CI) — จัดการ lifecycle ไม่ได้ ใช้ตามที่ให้มา
     if have_token and time.time() - _token_minted_at < _REFRESH_AFTER_SEC:
        return  # token ที่ mint เองยังไม่ใกล้หมดอายุ
     
     token = _token_from_metadata(audience=settings.mlflow_uri)
     if token:
         os.environ["MLFLOW_TRACKING_TOKEN"] =token
         _token_minted_at = time.time()
         return
     
     raise RuntimeError(
         "เชื่อม MLflow แบบ authenticated ไม่ได้: ไม่มี MLFLOW_TRACKING_TOKEN "
        "และไม่ได้รันอยู่บน GCP\n"
        "ถ้ารันจากเครื่อง local ให้ตั้ง token เองก่อน:\n"
        "  PowerShell: $env:MLFLOW_TRACKING_TOKEN = gcloud auth print-identity-token\n"
        "  bash:       export MLFLOW_TRACKING_TOKEN=$(gcloud auth print-identity-token)"
     )

def configure_mlflow() -> None:
    """setup มาตรฐานก่อนใช้งาน mlflow ทุกครั้ง: auth → tracking uri → experiment
    รวมไว้ที่เดียวเพื่อให้ train / tune / promote / gate ทำเหมือนกันเป๊ะ"""
    ensure_mlflow_auth()
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.experiment)