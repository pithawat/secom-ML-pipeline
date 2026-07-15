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

_METADATA_IDENTITY_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/identity"
)

_REFRESH_AFTER_SEC = 50 * 60  # ID token อายุ 60 นาที — ต่ออายุก่อนหมดที่ ~50 นาที


_token_minted_at: float | None = None  # None = token มาจากภายนอก เราไม่ยุ่ง lifecycle

def _is_local(uri: str) -> bool:
    host = urllib.parse.urlparse(uri).hostname
    return host in ("127.0.0.1", "localhost")

def _decode_jwt_claims(token: str) -> dict | None:
    """decode payload ของ JWT (ไม่ verify signature) — ใช้ debug ว่า aud/email ที่ metadata
    คืนมาตรงกับที่ Cloud Run คาดหวังจริงไหม. คืน None ถ้า token ไม่ใช่ JWT 3 ส่วน."""
    import base64
    import json

    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)  # เติม padding ให้ base64 ครบ
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None


def _token_from_metadata(audience: str) -> str | None:
    print(f"[gcp_auth] requesting ID token for audience={audience!r}")
    try:
        r = requests.get(
            _METADATA_IDENTITY_URL,
            params={"audience": audience},  # ให้ requests encode ครั้งเดียว ถูกต้องเสมอ
            headers={"Metadata-Flavor": "Google"},
            timeout=3,
        )
        r.raise_for_status()
    except requests.RequestException:
        return None  # ไม่ได้รันบน GCP (เช่น เครื่อง dev) — ให้ชั้นถัดไปตัดสินใจ

    token = r.text.strip()  # กัน newline/space แฝง ที่ทำให้ Bearer header เพี้ยน → "Invalid bearer token"
    claims = _decode_jwt_claims(token)
    if claims is None:
        print(f"[gcp_auth] ⚠ metadata ไม่ได้คืน JWT! len={len(token)} head={token[:48]!r}")
    else:
        print(f"[gcp_auth] token claims: aud={claims.get('aud')!r} "
              f"email={claims.get('email')!r} iss={claims.get('iss')!r}")
    return token
    
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
     
     # audience = service URL แบบไม่มี trailing slash (ตามหลักฐานจริงจากการทดสอบ curl มือ
     # ด้วย impersonation ที่ผ่าน audience check ได้) — decode ด้านบนจะยืนยัน aud จริงให้เห็น
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