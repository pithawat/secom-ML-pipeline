"""ทดสอบ preprocessor ด้วยข้อมูลสังเคราะห์เล็ก ๆ — เร็ว รันได้ทุกที่ ไม่ต้องมี dataset"""
import numpy as np
import pandas as pd

from src.preprocess import SimplePreprocessor


def _toy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        0: rng.normal(size=60),
        1: rng.normal(size=60),
        2: np.ones(60),  # variance = 0 → ต้องถูกตัดโดย variance filter
    })
    df[3] = df[0] * 2 + 0.001 * rng.normal(size=60)  # corr ~1 กับ col 0 → ต้องถูกตัด
    df.iloc[::7, 1] = np.nan  # โปรย NaN เลียนแบบ SECOM
    return df


def test_fit_transform_drops_bad_columns_and_fills_nan():
    pre = SimplePreprocessor(corr_threshold=0.95, var_threshold=1e-5)
    out = pre.fit_transform(_toy_frame())
    assert list(out.columns) == [0, 1]      # col 2 (var ต่ำ) และ 3 (corr สูง) หายไป
    assert not out.isna().any().any()       # NaN ถูก impute หมด
    assert np.isfinite(out.to_numpy()).all()


def test_transform_is_deterministic():
    df = _toy_frame()
    pre = SimplePreprocessor().fit(df)
    a, b = pre.transform(df), pre.transform(df)
    pd.testing.assert_frame_equal(a, b)


def test_transform_handles_all_nan_row():
    """แถวที่ sensor หลุดหมดทุกตัว — ระบบจริงเจอได้ ต้องไม่ crash"""
    df = _toy_frame()
    pre = SimplePreprocessor().fit(df)
    row = pd.DataFrame([[np.nan] * df.shape[1]], columns=df.columns)
    out = pre.transform(row)
    assert np.isfinite(out.to_numpy()).all()  # ถูกเติมด้วย mean ของ train