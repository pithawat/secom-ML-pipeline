"""
Champion vs Challenger — ตัดสินว่าโมเดลใหม่ได้ขึ้น production หรือไม่

หลักการที่เลือกใช้ (และเหตุผล):
1. ประเมิน "ทั้งสองตัว" ใหม่สด ๆ บน holdout ชุดเดียวกัน ณ ตอนนี้
   — ไม่เอา metric เก่าที่บันทึกไว้คนละรอบมาเทียบกัน เพราะถ้า dataset ถูกอัปเดต
   ตัวเลขจากคนละข้อมูลจะเทียบกันไม่ได้ (ต้อง apples-to-apples เสมอ)
2. challenger ต้องชนะ >= min_delta ถึงจะ promote
   — กัน churn: ถ้าดีกว่ากัน 0.0001 การสลับโมเดล (และ redeploy) ไม่คุ้มความเสี่ยง
3. ยังไม่มี champion (deploy ครั้งแรก) → challenger ขึ้นทันที
   (มาถึงขั้นนี้ได้แปลว่าผ่าน model gate มาแล้ว — ดูลำดับ job ใน workflow)
4. challenger เป็นตัวเดียวกับ champion อยู่แล้ว → ถือว่า promote (idempotent)
   เคสนี้เกิดตอน rerun pipeline หลัง deploy รอบก่อน fail กลางทาง
   ถ้าไม่ทำแบบนี้ rerun จะข้าม deploy แล้ว API ค้างอยู่กับโมเดลเก่า

รันโดย GitHub Actions หลัง gate ผ่าน:
    python -m src.promote --challenger-version 7
ผลลัพธ์ส่งต่อให้ job ถัดไปทาง $GITHUB_OUTPUT (promoted=true/false, champion_version=N)
"""

import argparse
import json
import math
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import settings
from src.data import chronological_split, data_fingerprint, load_raw
from src.evaluate import compute_metrics, predict_positive_proba
from src.gcp_auth import configure_mlflow

def _evaluate(model_uri: str, X_test, y_test) -> dict:
     """โหลดโมเดลจาก registry แล้ววัดบน holdout — ใช้ code path เดียวกับตอนเทรน
    (compute_metrics เดียวกัน threshold เดียวกัน) เพื่อให้ตัวเลขเทียบกันได้จริง"""
     model = mlflow.sklearn.load_model(model_uri)
     proba = predict_positive_proba(model, X_test)
     return compute_metrics(y_test, proba, settings.decision_threshold)

def _write_github_output(**kv) -> None:
     """ส่งค่าให้ step ถัดไปใน GitHub Actions — ถ้ารันนอก Actions จะไม่ทำอะไร"""
     path = os.getenv("GITHUB_OUTPUT")
     if not path:
          return 
     with open(path, "a", encoding="utf-8") as f:
          for k, v in kv.items():
               f.write(f"{k}={v}\n")

def main(challenger_version: str, min_delta: float) -> None:
    configure_mlflow()
    client = MlflowClient()
    name = settings.registered_model_name
    metric = settings.primary_metric

    X, y = load_raw()
    _, X_test, _, y_test = chronological_split(X, y)

    ch_metrics = _evaluate(f"models:/{name}/{challenger_version}", X_test, y_test)
    print(f"challenger v{challenger_version}: {json.dumps(ch_metrics, default=float)}")

    try:
          champ_mv = client.get_model_version_by_alias(name, settings.champion_alias)
    except MlflowException:
         champ_mv =None

    if champ_mv is None:
         promoted, reason = True, "first champion "
    elif champ_mv.version == str(challenger_version):
         promoted, reason = True, "challenger เป็น champion อยู่แล้ว (rerun) — deploy ซ้ำให้ตรงกัน"
    else:
        champ_metrics = _evaluate(f"models:/{name}/{champ_mv.version}", X_test, y_test)
        print(f"champion   v{champ_mv.version}: {json.dumps(champ_metrics, default=float)}")
        ch_score, champ_score = ch_metrics[metric], champ_metrics[metric]
    
        promoted = (not math.isnan(ch_score)) and (
             math.isnan(champ_score) or ch_score >= champ_score + min_delta
        )

        reason =(
            f"{metric}: challenger={ch_score:.4f} vs "
            f"champion(v{champ_mv.version})={champ_score:.4f}, min_delta={min_delta}"
        )

    if promoted:
        client.set_registered_model_alias(
              name, settings.champion_alias, challenger_version
         )
        client.set_model_version_tag(name, str(challenger_version), "promoted_reason", reason)
        client.set_model_version_tag(
             name, str(challenger_version), "eval_data_hash", data_fingerprint()
        )
        champion_version = str(challenger_version)
        print(f"PROMOTED -> v{challenger_version} คือ @champion ตัวใหม่ ({reason})")
    else:
         champion_version = champ_mv.version
         print(f"KEPT → champion v{champ_mv.version} ยังอยู่ ({reason})")

    _write_github_output(
        promoted=str(promoted).lower(), champion_version=champion_version
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenger-version", required=True,
        help="เลข version (ไม่ใช่ alias) — pin โดย scripts/resolve_version.py ตั้งแต่ต้น pipeline",
    )
    parser.add_argument("--min-delta", type=float, default=settings.promote_min_delta)
    args = parser.parse_args()
    main(args.challenger_version, args.min_delta)