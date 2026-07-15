import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import hashlib
import fsspec

from src.config import settings

def load_raw(
        data_path: Path | str |None = None,
        labels_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    
    # str() ไม่ใช่ Path() — Path("gs://bucket/x") จะ normalize เหลือ "gs:/bucket/x"
    # (ตัด // เหลือ /) ทำให้ gcsfs หา path ไม่เจอตอนรันบน cloud (SECOM_DATA_DIR=gs://...)
    data_path = str(data_path or settings.data_file)
    labels_path = str(labels_path or settings.labels_file)

    X = pd.read_csv(data_path, sep=r"\s+", header=None)

    raw = pd.read_csv(labels_path, sep=r"\s+", header=None)
    y = raw[0].replace(-1, 0).astype(int)
    y.name = "label"

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"row mismatch: features={X.shape[0]} labels={y.shape[0]}"
        )
    return X, y

def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_size: float | None = None
):
    """Forward-in-time split. NEVER shuffle — จะทำให้เกิด temporal leakage."""
    test_size = settings.test_size if test_size is None else test_size
    return train_test_split(X, y, test_size=test_size, shuffle=False)

def data_fingerprint(
    data_path: str | None = None, labels_path: str | None = None
) -> str:
    """md5 (ย่อ 12 ตัว) ของไฟล์ data+labels — ใช้ tag ลง MLflow run/model version

    เหตุผล: การเปรียบเทียบ champion vs challenger จะแฟร์ก็ต่อเมื่อวัดบนข้อมูลชุดเดียวกัน
    fingerprint ทำให้ตรวจย้อนหลังได้เสมอว่าโมเดลไหนเทรน/ถูกวัดบนข้อมูลชุดไหน (lineage)
    """
    data_path = str(data_path or settings.data_file)
    labels_path = str(labels_path or settings.labels_file)
    h = hashlib.md5()
    for p in (data_path, labels_path):
        with fsspec.open(p, "rb") as f:  # fsspec เปิดได้ทั้ง local และ gs://
            h.update(f.read())
    return h.hexdigest()[:12]