import numpy as np
import pytest

from src.evaluate import compute_metrics, predict_positive_proba


def test_perfect_separation_gives_perfect_metrics():
    y = [0, 0, 1, 1]
    proba = [0.1, 0.2, 0.8, 0.9]
    m = compute_metrics(y, proba, threshold=0.5)
    assert m["Recall_Fault"] == 1.0
    assert m["Precision_Fault"] == 1.0
    assert m["ROC-AUC"] == 1.0
    assert m["PR_AUC"] == 1.0


def test_single_class_returns_nan_auc_not_crash():
    """SECOM non-stationary: บางช่วงเวลาไม่มี fault เลย — ต้องได้ nan ไม่ใช่ exception"""
    m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert np.isnan(m["ROC-AUC"]) and np.isnan(m["PR_AUC"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.1, 0.2, 0.3])


class _OneClassModel:
    """โมเดลที่เห็นแต่คลาส 0 ตอนเทรน — predict_proba คืน (n, 1) ไม่ใช่ (n, 2)"""
    classes_ = [0]

    def predict_proba(self, X):
        return np.ones((len(X), 1))


def test_predict_positive_proba_when_model_never_saw_fault():
    proba = predict_positive_proba(_OneClassModel(), np.zeros((4, 3)))
    assert proba.shape == (4,)
    assert (proba == 0).all()  # ไม่เคยเห็น fault → P(fault)=0 ไม่ใช่ index error