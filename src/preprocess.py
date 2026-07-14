"""
fit-on-train pipeline: impute -> IQR clip -> variance filter -> corr filter -> scale.
ทำเป็น BaseEstimator/TransformerMixin เพื่อยัดเข้า sklearn.Pipeline ได้
=> preprocessor fit บน train fold เท่านั้นโดยอัตโนมัติ ไม่มีทาง leak เข้า test.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.config import settings

class SimplePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        corr_threshold: float | None = None,
        var_threshold: float | None = None,
    ):
        self.corr_threshold = (
            settings.corr_threshold if corr_threshold is None else corr_threshold
        ) 
        self.var_threshold = (
            settings.var_threshold if var_threshold is None else var_threshold
        )
                 
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        # 1) mean สำหรับเติม NaN (เรียนจาก train เท่านั้น)
        self.means_ = X.mean()
        X = X.fillna(self.means_)

        # 2) ขอบ IQR สำหรับ clip outlier
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X = X.clip(self.lower_, self.upper_, axis=1)

        # 3) ตัดฟีเจอร์ variance ต่ำ
        X = X[X.columns[X.std() > self.var_threshold]]

        # 4) ตัดฟีเจอร์ correlation สูง
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if (upper[c] > self.corr_threshold).any()]
        self.columns_ = [c for c in X.columns if c not in drop]

        # 5) scaler
        self.scaler_ = StandardScaler().fit(X[self.columns_])
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy().fillna(self.means_)
        X = X.clip(self.lower_, self.upper_, axis=1)
        X = X[self.columns_]
        return pd.DataFrame(
            self.scaler_.transform(X), columns=self.columns_, index=X.index
        )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)