
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import mlflow
import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.config import settings
from src.data import chronological_split, load_raw
from src.evaluate import compute_metrics, predict_positive_proba
from src.train import build_pipeline
from src.gcp_auth import configure_mlflow, ensure_mlflow_auth

# จำนวน fold สำหรับ TimeSeriesSplit (override ได้ผ่าน SECOM_CV_SPLITS ถ้าจะเพิ่มใน config)
N_SPLITS = 5
 
"""
train.py วัดครั้งเดียวบน X_test ที่ fix อยู่แล้ว — holdout เดียวพอ
tune.py วัดซ้ำเป็นร้อยครั้ง ถ้าใช้ val-fold เดียว (1) เสี่ยงโดนช่วงไม่มี fault → พัง (อย่างที่เจอ) 
(2) PR-AUC เหวี่ยงตาม noise ของ fold เดียว → เลือก hyperparameter ผิด TimeSeriesSplit เฉลี่ยหลาย fold เลยได้สัญญาณนิ่งกว่าสำหรับการ "เลือก"
"""
def _usable_folds(y_train, n_splits: int = N_SPLITS):
    """คืน list ของ (train_idx, val_idx) เฉพาะ fold ที่ val มีทั้ง 2 คลาส.
 
    TimeSeriesSplit = forward-in-time เสมอ (val อยู่หลัง train ตามเวลา) ไม่ leak.
    SECOM non-stationary: บาง fold ช่วงท้ายไม่มี fault -> ข้าม fold นั้นไป
    ไม่งั้น PR-AUC/ROC-AUC ประเมินไม่ได้.
    """
    y = np.asarray(y_train)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for tr_idx, val_idx in tscv.split(y):
        if len(np.unique(y[val_idx])) >= 2:
            folds.append((tr_idx, val_idx))
    return folds
 
 
def _suggest_model(trial):
    rs = settings.random_state
    name = trial.suggest_categorical("model", ["rf", "lgbm", "xgb", "logreg"])
 
    if name == "rf":
        return RandomForestClassifier(
            max_depth=trial.suggest_int("rf_max_depth", 2, 32),
            n_estimators=trial.suggest_int("rf_n_estimators", 50, 300, step=10),
            max_features=trial.suggest_float("rf_max_features", 0.2, 1.0),
            class_weight="balanced",
            random_state=rs,
        )
    if name == "lgbm":
        return LGBMClassifier(
            num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 127),
            learning_rate=trial.suggest_float("lgbm_lr", 1e-3, 0.3, log=True),
            n_estimators=trial.suggest_int("lgbm_n_est", 100, 500, step=50),
            class_weight="balanced",
            random_state=rs,
            verbose=-1,
        )
    if name == "xgb":
        return XGBClassifier(
            max_depth=trial.suggest_int("xgb_max_depth", 2, 8),
            learning_rate=trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
            n_estimators=trial.suggest_int("xgb_n_est", 100, 500, step=50),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),
            random_state=rs,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )
    return LogisticRegression(
        C=trial.suggest_float("lr_C", 1e-3, 10, log=True),
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )
 
 
def make_objective(X_train, y_train, folds):
    """objective เฉลี่ย PR-AUC ข้าม fold ของ TimeSeriesSplit.
 
    ใช้ .iloc ด้วย positional index จาก tscv เพราะ index เดิมของ X_train
    อาจไม่เรียง 0..n หลัง chronological split.
    """
    def objective(trial):
        ensure_mlflow_auth()  # study ยาว ๆ อาจเกินอายุ ID token (~1 ชม.) — ต่ออายุให้เอง
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            model = _suggest_model(trial)
            fold_pr = []
            for tr_idx, val_idx in folds:
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
 
                pipe = build_pipeline(model)
                pipe.fit(X_tr, y_tr)
                # ดึง proba คลาส fault แบบทนทาน (คืน 1-D เสมอ ไม่ hardcode index 1)
                proba = predict_positive_proba(pipe, X_val)
                metrics = compute_metrics(y_val, proba)
                if not np.isnan(metrics["PR_AUC"]):
                    fold_pr.append(metrics["PR_AUC"])
 
            if not fold_pr:
                raise optuna.TrialPruned()  # ไม่มี fold ไหนประเมินได้เลย
 
            mean_pr = float(np.mean(fold_pr))
            mlflow.log_params(trial.params)
            mlflow.log_metric("PR_AUC_mean", mean_pr)
            mlflow.log_metric("PR_AUC_std", float(np.std(fold_pr)))
            mlflow.log_metric("n_usable_folds", len(fold_pr))
            return mean_pr  # maximize
 
    return objective
 
 
def run_study(n_trials: int = 50) -> optuna.Study:
    X, y = load_raw()
    X_train, _X_test, y_train, _y_test = chronological_split(X, y)
 
    folds = _usable_folds(y_train, N_SPLITS)
    if not folds:
        raise ValueError(
            "ไม่มี TimeSeriesSplit fold ไหนที่ validation มีทั้ง 2 คลาสเลย. "
            "SECOM ช่วงท้ายอาจ fault เบาบางมาก ลองลด N_SPLITS หรือรวม train+test "
            "แล้ว split ใหม่ให้แต่ละ fold มี fault พอ."
        )
    fault_counts = [int(np.sum(np.asarray(y_train)[v])) for _, v in folds]
    print(f"TimeSeriesSplit: ใช้ได้ {len(folds)}/{N_SPLITS} folds  "
          f"(faults ต่อ val fold: {fault_counts})")
 
    configure_mlflow()
    study = optuna.create_study(direction="maximize")
    with mlflow.start_run(run_name="optuna_search"):
        study.optimize(make_objective(X_train, y_train, folds), n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_PR_AUC_mean", study.best_value)
 
    print("best mean PR-AUC:", round(study.best_value, 4))
    print("best params:", study.best_params)
    return study
 
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_study(n)