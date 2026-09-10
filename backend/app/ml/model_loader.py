import os
import json
import logging
import joblib
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.xgb_model: Optional[Any] = None
        self.model_features: List[str] = [
            "ret_1d", "ret_5d", "ret_20d", "ret_60d",
            "vol_20d", "vol_60d", "momentum_20d",
            "price_vs_ma20", "price_vs_ma60",
            "downside_vol_20d", "drawdown_60d"
        ]
        self.recommendation_config: Dict[str, Any] = {}
        self.is_loaded: bool = False

    def load_models(self, artifacts_dir: str = "models"):
        base_backend = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fundiq_root = os.path.dirname(base_backend)

        possible_dirs = [
            os.path.join(fundiq_root, "models"),
            os.path.join(base_backend, "models"),
            os.path.join(base_backend, "models_artifacts"),
            os.path.join(fundiq_root, "models_artifacts"),
            os.path.join("..", "models"),
            artifacts_dir,
            "models",
            "."
        ]

        target_dir = None
        for p in possible_dirs:
            xgb_check = os.path.join(p, "xgboost_nav_model.pkl")
            if os.path.exists(xgb_check):
                target_dir = p
                break

        if not target_dir:
            target_dir = possible_dirs[0]

        xgb_path = os.path.join(target_dir, "xgboost_nav_model.pkl")
        features_path = os.path.join(target_dir, "model_features.pkl")
        config_path = os.path.join(target_dir, "model_config.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(target_dir, "recommendation_config.pkl")

        if os.path.exists(xgb_path):
            try:
                self.xgb_model = joblib.load(xgb_path)
                logger.info(f"Loaded XGBoost model from {xgb_path}")

                if os.path.exists(features_path):
                    loaded_features = joblib.load(features_path)
                    if isinstance(loaded_features, list):
                        self.model_features = loaded_features
                        logger.info(f"Loaded model features from {features_path}: {self.model_features}")

                if os.path.exists(config_path):
                    self.recommendation_config = joblib.load(config_path)
                    logger.info(f"Loaded model config from {config_path}")

                self.is_loaded = True
            except Exception as e:
                logger.warning(f"Could not load XGBoost model artifact: {e}. Operating in dynamic baseline mode.")
        else:
            logger.info(f"XGBoost model file not found at {xgb_path}. Operating in dynamic feature analytics mode until file is uploaded.")

model_registry = ModelRegistry()
