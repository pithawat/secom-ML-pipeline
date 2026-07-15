"""Model registry. เพิ่ม/ลบโมเดลได้ที่เดียว."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import settings

# params ที่อยากให้ mlflow log (กัน UI รก)
LOGGED_PARAMS = {"class_weight", "n_estimators", "learning_rate", "max_depth"}


def make_models() -> dict:
    rs = settings.random_state
    return {
        # LogisticRegression = โมเดลที่ดีสุดของ SECOM นี้ (recall 0.41, PR-AUC 0.21,
        # ROC-AUC 0.79) — tree models (RF/LGBM/XGB) ให้ proba เกาะกลุ่มต่ำ recall=0
        # ที่ threshold 0.5. ห้าม comment ทิ้งอีก ไม่งั้น challenger จะเป็นโมเดลอ่อนที่ gate บล็อก
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="liblinear"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=rs
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced", random_state=rs, verbose=-1
        ),
        "XGB": XGBClassifier(
            n_estimators=300,
            max_depth=4,            # ตื้นลงกัน overfit บน window เล็ก
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=rs,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        ),
    }
