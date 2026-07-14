"""
MODEL GATE — ด่านสุดท้ายก่อนโมเดลจะถูก promote/deploy ไปให้ API ใช้

ปรัชญา: gate "ไม่ได้" มีหน้าที่เลือกโมเดลที่ดีที่สุด (นั่นคืองานของ promote.py)
gate มีหน้าที่กัน "โมเดลพัง/เพี้ยน" ไม่ให้หลุดไป production เช่น
  - ทายเป็นคลาสเดียวล้วน (recall=0 หรือ specificity=0)
  - crash เมื่อเจอ NaN (ข้อมูลจริงจากโรงงานมี NaN แน่นอน)
  - ผลไม่ deterministic (ทายสองครั้งได้คนละค่า)
  - แย่กว่าการสุ่ม (PR-AUC ต่ำกว่า base rate)

ใช้ "ข้อมูลจริง" จาก test window ของ chronological split (~314 แถวท้าย:
pass ~296 / fail ~17) — มีทั้ง label ดีและเสียครบตามโจทย์
ค่า floor ทั้งหมดตั้งใน src/config.py (gate_min_*) — ปรับผ่าน env ได้ ดู README §9

การรัน (ใน GitHub Actions job `gate` — ตั้ง MODEL_URI ให้อัตโนมัติ):
    MODEL_URI="models:/secom-fault/7" pytest tests/gate -m gate -v
ถ้าไม่ตั้ง MODEL_URI จะ skip ทั้งไฟล์ (ทำให้ `pytest` เฉย ๆ บนเครื่อง dev ไม่พัง)
"""

import os 

import numpy as np
import pandas as pd
import pytest

import mlflow.sklearn
from src.config import settings
from src.data import chronological_split, load_raw
from src.evaluate import compute_metrics, predict_positive_proba
from src.gcp_auth import configure_mlflow

pytestmark = pytest.mark.gate


MODEL_URI = os.getenv("MODEL_URI")
if MODEL_URI is None:
    pytest.skip(
        "ไม่ได้ตั้ง MODEL_URI (เช่น models:/secom-fault/7 หรือ gs://…/model) — ข้าม gate",
        allow_module_level=True,
    )

@pytest.fixture(scope="module")
def model():
    configure_mlflow()  # ถ้า MODEL_URI เป็น models:/ ต้อง auth กับ MLflow ก่อน
    return mlflow.sklearn.load_model(MODEL_URI)

@pytest.fixture(scope="module")
def holdout():
    X, y = load_raw()
    _, X_test, _, y_test = chronological_split(X, y)
    return X_test, np.asarray(y_test)

@pytest.fixture(scope="module")
def proba(model, holdout):
    return predict_positive_proba(model, holdout[0])

def test_proba_is_valid_probability_vector(proba, holdout):
    assert proba.shape == (len(holdout[0]),)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()

def test_handles_row_of_all_nan(model, holdout):
    """เคสสุดขั้วที่หน้างานเจอได้: sensor หลุดทั้งแถว — ต้องได้ค่า valid ไม่ใช่ 500"""
    X_test, _ = holdout
    row = pd.DataFrame([[np.nan] * X_test.shape[1]], columns=X_test.columns)
    p = predict_positive_proba(model, row)
    assert 0.0 <= p[0] <= 1.0

def test_predictions_are_deterministic(model, holdout):
    a = predict_positive_proba(model, holdout[0])
    b = predict_positive_proba(model, holdout[0])
    assert np.allclose(a,b), "ทายสองครั้งได้คนละค่า — ห้ามปล่อยไป production"

# ── พฤติกรรมบน "ของเสียจริง" และ "ของดีจริง" 

def test_catches_real_fault_lots(proba, holdout):
     """ชิ้นงานที่เสียจริงในไลน์ผลิต — โมเดลต้องจับได้อย่างน้อยตาม floor"""
     _, y = holdout
     fault_mask = y == 1
     assert fault_mask.sum() >= 5, "test window ต้องมี fault จริงพอให้ประเมิน"
     recall = float((proba[fault_mask] >= settings.decision_threshold).mean())
     assert recall >= settings.gate_min_recall_fault, (
         f"จับของเสียจริงได้แค่ {recall:.2%}"
         f"(floor={settings.gate_min_recall_fault:.0%})  — โมเดลนี้ห้ามผ่านไป deploy"
     )

def test_passes_real_good_lots(proba, holdout):
    """ชิ้นงานดีจริง — ห้ามเหมาว่าเสียเกิน floor (ไม่งั้น production หยุดทั้งไลน์)"""
    _, y = holdout
    good_mask = y == 0
    assert good_mask.sum() >= 50, "test window ต้องมีชิ้นงานดีพอให้ประเมิน"
    specificity =float((proba[good_mask] < settings.decision_threshold).mean())
    assert specificity >= settings.gate_min_specifically, (
        f"ทายชิ้นงานดีถูกแค่ {specificity:.2%} "
        f"(floor={settings.gate_min_specificity:.0%})"
    )

def test_better_than_random(proba, holdout):
    """กันเคสโมเดลเพี้ยนแบบเนียน ๆ: AUC ต้องดีกว่าการสุ่มชัดเจน"""
    _, y = holdout
    m = compute_metrics(y, proba, settings.decision_threshold)
    assert not np.isnan(m["PR_AUC"]), "PR_AUC วัดไม่ได้ — holdout ผิดปกติ"
    assert m["PR_AUC"] >= settings.gate_min_pr_auc, (
         f"PR_AUC={m['PR_AUC']:.4f} < floor {settings.gate_min_pr_auc} "
        f"(base rate ~0.066 — ค่านี้แปลว่าแทบไม่ต่างจากสุ่ม)"
    )
    assert m["ROC-AUC"] >= settings.gate_min_roc_auc, (
         f"ROC-AUC={m['ROC-AUC']:.4f} < floor {settings.gate_min_roc_auc}"
    )