"""
แปลง alias → เลข version แล้ว "pin" ไว้ใช้ตลอดทั้ง pipeline run

ทำไมต้อง pin: alias (@challenger) เป็น pointer ที่ขยับได้ — ถ้ามีคน trigger
เทรนรอบใหม่ระหว่างที่ gate ของรอบนี้กำลังรัน alias จะชี้ไปตัวใหม่กลางคัน
=> job ถัด ๆ ไป (gate/promote/deploy) ต้องอ้าง "เลข version" ที่ resolve ครั้งเดียว
   ตอนต้น ไม่ใช่ alias   (ใน workflow ยังมี concurrency group กันอีกชั้นหนึ่ง)

ใช้: python -m scripts.resolve_version challenger
พิมพ์เลข version ทาง stdout และเขียน version=N ลง $GITHUB_OUTPUT (ถ้ารันใน Actions)
"""

import os
import sys

from mlflow.tracking import MlflowClient

from src.config import settings
from src.gcp_auth import configure_mlflow


def main(alias: str) -> str:
    configure_mlflow()
    mv = MlflowClient().get_model_version_by_alias(
        settings.registered_model_name, alias
    )
    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"version={mv.version}\n")
    print(mv.version)
    return mv.version


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else settings.challenger_alias)