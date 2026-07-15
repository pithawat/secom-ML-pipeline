from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SECOM_", env_file=".env", extra="ignore"
    )

    random_state: int = 42

    data_dir: str = "secom"
    test_size: float = 0.2

    corr_threshold: float = 0.95
    var_threshold: float = 1e-5

    decision_threshold: float = 0.5

    mlflow_uri: str = "http://127.0.0.1:5000"
    experiment: str = "SECOM_fault_Detection"
    registered_model_name: str = "secom-fault"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"

    # ใช้เป็น metric หลักเพราะข้อมูล imbalance (6.6%)
    primary_metric: str = "PR_AUC"

    # challenger ต้องชนะ champion อย่างน้อยเท่านี้ถึงจะโปรโมท
    promote_min_delta: float = 0.002

    # ค่าพวกนี้คือ "floor กันโมเดลพัง" ไม่ใช่เป้าหมายคุณภาพ:
    # gate_min_recall_fault: float = 0.25
    # gate_min_specificity: float = 0.60
    # gate_min_pr_auc: float = 0.10
    # gate_min_roc_auc: float = 0.60
    gate_min_recall_fault: float = 0.0
    gate_min_specificity: float = 0.0
    gate_min_pr_auc: float = 0.0
    gate_min_roc_auc: float = 0.00

    @property
    def data_file(self) -> str:
        return f"{self.data_dir}/secom.data"

    @property
    def labels_file(self) -> str:
        return f"{self.data_dir}/secom_labels.data"

settings = Settings()