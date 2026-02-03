import pytest
import polars as pl
import numpy as np
import torch
from src import DataProcessor, build_model, LSTM

# ==========================================
# 1. FIXTURES (Setup Data)
# ==========================================

@pytest.fixture
def sample_data():
    """Creates a dummy Polars DataFrame with 100 rows of stock data."""
    dates = pl.datetime_range(
        start=pl.datetime(2023, 1, 1),
        end=pl.datetime(2023, 4, 10),
        interval="1d",
        eager=True
    )
    n = len(dates)
    
    # Random price data
    df = pl.DataFrame({
        "Stock ID": ["TEST_STOCK"] * n,
        "datetime": dates,
        "open": np.random.uniform(100, 200, n),
        "high": np.random.uniform(200, 210, n),
        "low": np.random.uniform(90, 100, n),
        "close": np.random.uniform(100, 200, n),
        "volume": np.random.uniform(1000, 5000, n)
    })
    
    # Cast to Float32 as expected by your processor
    cols = ["open", "high", "low", "close", "volume"]
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in cols])
    return df

# ==========================================
# 2. UNIT TESTS
# ==========================================

def test_data_processor_indicators(sample_data):
    """Test if DataProcessor correctly adds indicators like RSI."""
    dp = DataProcessor(mode="train")
    
    # Manually inject the dataframe (skipping load_data from file)
    dp.df = sample_data
    dp.stocks = ["TEST_STOCK"]
    
    # Calculate specific features
    features_to_test = ["rsi", "roc_5"]
    df_result, features = dp.calculate_indicators(features_to_test)
    
    # Assertions
    assert "rsi" in df_result.columns
    assert "roc_5" in df_result.columns
    # Check if features list updated correctly
    assert "rsi" in features
    # Check data is not empty
    assert df_result.height > 0

def test_target_creation(sample_data):
    """Test if binary target column is created."""
    dp = DataProcessor()
    dp.df = sample_data
    
    # Run target creation
    df_target = dp.create_binary_target(dp.df, profit_pct=0.01, stop_pct=0.01, time_horizon=2)
    
    assert "target" in df_target.columns
    # Target should be 0 or 1
    unique_targets = df_target["target"].unique().to_list()
    assert set(unique_targets).issubset({0, 1, -1}) # depending on your logic (binary or triple barrier)

def test_model_builder_lstm():
    """Test if LSTM builds with correct dimensions."""
    features = ["f1", "f2", "f3", "f4", "f5"]
    
    # FIX: Change n_layers from 1 to 2 to silence the dropout warning
    model, hp = build_model("lstm", feature_cols=features, hidden_size=64, n_layers=2)
    
    assert isinstance(model, LSTM)
    assert hp["d_in"] == 5  
    assert hp["hidden_size"] == 64

def test_model_forward_pass():
    """Test if the model accepts input tensors and outputs correct shape."""
    # Setup
    batch_size = 32
    seq_len = 30
    n_features = 10
    
    model, _ = build_model("lstm", d_in=n_features, hidden_size=32)
    
    # Create dummy input tensor (Batch, Seq, Features)
    dummy_input = torch.randn(batch_size, seq_len, n_features)
    
    # Run model
    output = model(dummy_input)
    
    # Expect output shape (Batch, 1) for binary classification
    assert output.shape == (batch_size, 1)

def test_xgboost_builder():
    """Test if XGBoost builder works (it returns a different object type)."""
    model, hp = build_model("xgboost", n_estimators=100, max_depth=4)
    
    assert hp["model_type"] == "xgboost"
    assert model.n_estimators == 100
    assert model.max_depth == 4