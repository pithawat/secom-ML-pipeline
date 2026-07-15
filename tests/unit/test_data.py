"""ทดสอบ data loader ด้วยไฟล์สังเคราะห์ใน tmp_path — ไม่พึ่ง dataset จริง/เน็ต"""
import pytest

from src.data import chronological_split, load_raw


def _write_files(tmp_path, n_rows=10, n_label_rows=None):
    data = tmp_path / "d.data"
    labels = tmp_path / "l.data"
    data.write_text(
        "\n".join(f"{i} {i + 0.5} NaN" for i in range(n_rows)), encoding="utf-8"
    )
    n_label_rows = n_rows if n_label_rows is None else n_label_rows
    # สลับ -1 (pass) กับ 1 (fail) เหมือน format ของไฟล์ secom_labels.data
    labels.write_text(
        "\n".join(f"{-1 if i % 3 else 1} ts{i}" for i in range(n_label_rows)),
        encoding="utf-8",
    )
    return data, labels


def test_load_raw_maps_labels_to_binary(tmp_path):
    data, labels = _write_files(tmp_path)
    X, y = load_raw(data, labels)
    assert X.shape == (10, 3)
    assert set(y.unique()) <= {0, 1}   # -1 → 0 เรียบร้อย
    assert X[2].isna().all()           # "NaN" ในไฟล์ต้อง parse เป็น missing จริง


def test_load_raw_row_mismatch_raises(tmp_path):
    data, labels = _write_files(tmp_path, n_rows=10, n_label_rows=9)
    with pytest.raises(ValueError, match="row mismatch"):
        load_raw(data, labels)


def test_chronological_split_never_shuffles(tmp_path):
    data, labels = _write_files(tmp_path)
    X, y = load_raw(data, labels)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2)
    # test ต้องเป็น "ท้ายตาราง" เสมอ — ถ้า shuffle เมื่อไหร่คือ temporal leakage
    assert max(X_train.index) < min(X_test.index)
    assert list(X_test.index) == list(X.index[-len(X_test):])