import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
import argparse
import math
import os
import subprocess

from src.config import settings
from src.preprocess import SimplePreprocessor
from src.models import LOGGED_PARAMS
from src.data import load_raw, data_fingerprint, chronological_split
from src.models import make_models
from src.evaluate import compute_metrics, predict_positive_proba
from src.gcp_auth import configure_mlflow

def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("pre", SimplePreprocessor()),
         ("clf", model)
         ])

def _log_params(model) -> None:
    if hasattr(model, "get_params"):
        p = model.get_params()
        mlflow.log_params({k: v for k, v in p.items() if k in LOGGED_PARAMS})

def _git_sha() -> str:
    """commit ที่ใช้เทรน — บน CI/Vertex ส่งผ่าน env GIT_SHA (ใน image ไม่มี .git)
    บนเครื่อง dev ถามจาก git ตรง ๆ"""
    sha = os.getenv("GIT_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()[:12]
    except Exception:
        return "unknown"

def train_all(register: bool = False) -> dict:
    configure_mlflow()  # auth (ถ้าจำเป็น) + tracking uri + experiment

    X, y = load_raw()
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    lineage_tags = {"git_sha": _git_sha(), "data_hash": data_fingerprint()}

    results: dict[str, dict] = {}
    best: tuple[float, str, str] | None = None  # (score, ชื่อ algo, model_uri)

    for name, model in make_models().items():
        pipe = build_pipeline(model)
        with mlflow.start_run(run_name=name):
            mlflow.set_tags(lineage_tags)
            _log_params(model)

            pipe.fit(X_train, y_train)
            proba = predict_positive_proba(pipe, X_test)

            metrics = compute_metrics(y_test, proba, settings.decision_threshold)
            mlflow.log_metrics(metrics)

            info = mlflow.sklearn.log_model(
                pipe,
                name="model",
                # signature + input_example = สัญญา (contract) ของ input:
                # 590 คอลัมน์ float มี NaN ได้ — คนโหลดโมเดลไปใช้เห็นทันทีว่าต้องป้อนอะไร
                signature=infer_signature(X_test.head(5), proba[:5]),
                input_example=X_test.head(3),
            )

            results[name] = metrics
            print(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            # เลือก best ด้วย primary metric; 
            score = metrics.get(settings.primary_metric)
            if score is not None and not math.isnan(score):
                if best is None or score > best[0]:
                    best = (score, name, info.model_uri)
    
    if register:
        if best is None:
            raise RuntimeError(
                 f"ทุกโมเดลได้ {settings.primary_metric} = nan — "
                "test window อาจมีคลาสเดียว เช็คข้อมูล/การ split ก่อน"
            )
        score, algo, model_uri = best
        # register เฉพาะ best-of-run ตัวเดียว → เป็นผู้ท้าชิง (@challenger)
        # การตัดสินว่าได้ขึ้น @champion หรือไม่ เป็นหน้าที่ของ src/promote.py
        # (หลังผ่าน model gate แล้วเท่านั้น — ดู .github/workflows/train-deploy.yml)
        mv = mlflow.register_model(model_uri, settings.registered_model_name)
        client = MlflowClient()
        client.set_registered_model_alias(
            settings.registered_model_name, settings.challenger_alias, mv.version
        )

        for k, v in {**lineage_tags, "algo": algo,
                     settings.primary_metric: f"{score:.4f}"}.items():
            client.set_model_version_tag(
                settings.registered_model_name, mv.version, k, str(v)
            )
        print(f"registered challenger: v{mv.version} ({algo}, "
              f"{settings.primary_metric}={score:.4f})")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register", action="store_true",
        help="ลงทะเบียน best model ของรอบนี้เป็น @challenger (ใช้ตอนรันบน Vertex)",
    )
    train_all(register=parser.parse_args().register)