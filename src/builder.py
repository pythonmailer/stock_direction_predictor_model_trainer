from typing import Literal, List, Optional, Dict, Any, overload, Tuple
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .models import TransformerEncoder, LSTM

# ==========================================
# 1. Centralized Defaults (Source of Truth)
# ==========================================

DEFAULT_CONFIGS = {
    "lstm": {
        "d_in": None,
        "hidden_size": 128,
        "n_layers": 2,
        "dropout": 0.1
    },
    "transformer": {
        "d_in": None,
        "hidden_size": 128,
        "nhead": 4,
        "n_layers": 2,
        "d_out": 1,
        "max_len": 100,
        "dropout": 0.1,
        "pos_weight_val": None
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "random_state": 42
    },
    "rf": {
        "n_estimators": 500,
        "max_depth": 6,
        "random_state": 42,
    }
}

def get_template(model_type: Literal["lstm", "transformer", "rf", "xgboost"]) -> Dict[str, Any]:
    """
    Returns a COPY of the default configuration.
    """

    if model_type not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    return DEFAULT_CONFIGS[model_type].copy()

# ==========================================
# 2. The Refactored Factory (Uses the Config)
# ==========================================

@overload
def build_model(
    model_type: Literal["lstm"], 
    feature_cols: Optional[List[str]] = None,
    d_in: Optional[int] = None,
    hidden_size: int = 128, 
    n_layers: int = 2, 
    dropout: float = 0.1
) -> Tuple[LSTM, Dict[str, Any]]: ...

# Signature for Transformer
@overload
def build_model(
    model_type: Literal["transformer"], 
    feature_cols: Optional[List[str]], 
    d_in: Optional[int] = None,
    hidden_size: int = 128,
    nhead: int = 4,
    n_layers: int = 2,
    d_out: int = 1,
    max_len: int = 100,
    dropout: float = 0.1,
    pos_weight_val: Optional[float] = None
) -> Tuple[TransformerEncoder, Dict[str, Any]]: ...

# Signature for XGBoost
@overload
def build_model(
    model_type: Literal["xgboost"], 
    feature_cols: Optional[List[str]],
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    objective: str = "binary:logistic",
    eval_metric: str = "auc",
    tree_method: str = "hist",
    early_stopping_rounds: int = 50,
    random_state: int = 42
) -> Tuple[XGBClassifier, Dict[str, Any]]: ...

# Signature for Random Forest
@overload
def build_model(
    model_type: Literal["rf"], 
    feature_cols: Optional[List[str]],
    n_estimators: int = 500, 
    max_depth: int = 6,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, Any]]: ...

def build_model(model_type: Literal["lstm", "transformer", "rf", "xgboost"], feature_cols: Optional[List[str]] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Factory function to instantiate a model.
    NOW utilizes get_model_config to ensure parameters are consistent.
    """

    model_type = model_type.lower()
    
    if model_type not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    hp = DEFAULT_CONFIGS[model_type].copy()
    hp.update(kwargs)
    hp["model_type"] = model_type

    feature_cols = hp.get("feature_cols")

    if model_type in ["lstm", "transformer"]:
        if hp.get("d_in") is not None:
            pass
        elif feature_cols is not None:
            hp["d_in"] = len(feature_cols)
        else:
            raise ValueError(f"For {model_type}, you must provide 'feature_cols' or 'd_in'.")    

    if model_type == "lstm":
        model = LSTM(
            d_in=hp["d_in"],
            hidden_size=hp["hidden_size"],
            n_layers=hp["n_layers"],
            dropout=hp["dropout"]
        )
    
    elif model_type == "transformer":
        model = TransformerEncoder(
            d_in=hp["d_in"],
            d_model=hp["hidden_size"],
            nhead=hp["nhead"],  
            n_layers=hp["n_layers"],
            d_out=hp["d_out"],
            max_len=hp["max_len"],
            dropout=hp["dropout"],
            pos_weight_val=hp["pos_weight_val"]
        )

    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            random_state=hp["random_state"],
            n_jobs=-1
        )
        
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            subsample=hp["subsample"],
            colsample_bytree=hp["colsample_bytree"],
            n_jobs=-1,
            objective=hp["objective"],
            tree_method=hp["tree_method"],
            eval_metric=hp["eval_metric"],
            early_stopping_rounds=hp.get("early_stopping_rounds"),
            random_state=hp["random_state"],
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, hp