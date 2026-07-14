import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import settings


# def compute_metrics(y_true, proba, threshold: float | None = None) -> dict:
#     """คืน dict ของ metric หลัก. Fault = class 1."""
#     threshold = settings.decision_threshold if threshold is None else threshold
#     y_true = np.asarray(y_true)
#     pred = (np.asarray(proba) >= threshold).astype(int)
#     return {
#         "Recall_Fault": recall_score(y_true, pred, zero_division=0),
#         "Precision_Fault": precision_score(y_true, pred, zero_division=0),
#         "ROC-AUC": roc_auc_score(y_true, proba),
#         "PR_AUC": average_precision_score(y_true, proba),
#     }

def predict_positive_proba(pipe, X) -> np.ndarray:
    """ดึงความน่าจะเป็นของคลาส fault (label=1) แบบทนทาน — คืน 1-D array เสมอ.
 
    ทำไมต้องมี: `predict_proba(X)[:, 1]` พังได้ถ้าโมเดลเห็นคลาสเดียวตอนเทรน
    (SECOM non-stationary: บาง fold ไม่มี fault เลย) เพราะ predict_proba
    อาจคืน (n, 1) -> index [:, 1] error, หรือบางเวอร์ชันยุบเป็น scalar.
    วิธีถูกคือหา index ของคลาส 1 จาก classes_ ไม่ hardcode เลข 1.
    """
    proba = np.asarray(pipe.predict_proba(X))
    clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    classes = list(clf.classes_)
 
    if 1 in classes:
        col = proba[:, classes.index(1)]
    else:
        # โมเดลไม่เคยเห็นคลาส fault เลย -> ความน่าจะเป็น fault = 0 ทุกแถว
        col = np.zeros(proba.shape[0])
    return np.ravel(col)
 
 
def compute_metrics(y_true, proba, threshold: float | None = None) -> dict:
    """คืน dict ของ metric หลัก. Fault = class 1.
 
    ป้องกัน 2 เคสที่ SECOM เจอบ่อยและทำให้ sklearn โยน error ปริศนา:
    1. proba ยุบเป็น scalar/0-d  -> np.ravel บังคับเป็น 1-D เสมอ
       (ต้นเหตุของ 'got scalar array(0)' คือ pred กลายเป็น 0-d array)
    2. y_true มีคลาสเดียว (chronological fold ที่ไม่มี fault เลย)
       -> ROC-AUC / PR-AUC ไม่นิยาม คืน nan แทนที่จะ crash
    """
    threshold = settings.decision_threshold if threshold is None else threshold
    y_true = np.ravel(np.asarray(y_true))
    proba = np.ravel(np.asarray(proba, dtype=float))
 
    if proba.shape != y_true.shape:
        raise ValueError(
            f"y_true/proba length mismatch: {y_true.shape} vs {proba.shape}. "
            "เช็คขั้น extract proba (predict_proba(...)[:, idx_ของคลาส 1])."
        )
 
    pred = (proba >= threshold).astype(int)
    metrics = {
        "Recall_Fault": recall_score(y_true, pred, zero_division=0),
        "Precision_Fault": precision_score(y_true, pred, zero_division=0),
    }
    if len(np.unique(y_true)) < 2:
        # fold มีคลาสเดียว: AUC ประเมินไม่ได้
        metrics["ROC-AUC"] = float("nan")
        metrics["PR_AUC"] = float("nan")
    else:
        metrics["ROC-AUC"] = roc_auc_score(y_true, proba)
        metrics["PR_AUC"] = average_precision_score(y_true, proba)
    return metrics