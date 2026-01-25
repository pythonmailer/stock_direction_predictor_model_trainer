import pytest
import polars as pl
import numpy as np
import torch
from src.data_processor import DataProcessor
from src.builder import CompactStockEncoder

# ==========================================
# 1. TEST DATA PROCESSING
# ==========================================
def test_indicators_calculation():
    """
    Verify that indicators are calculated and no NaNs are left at the end
    """
    # Create a dummy dataframe (simulating 100 days of data)
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        "Stock ID": ["TEST"] * 100,
        "datetime": dates,
        "close": np.random.rand(100) * 100 + 100,
        "open": np.random.rand(100) * 100 + 100,
        "high": np.random.rand(100) * 100 + 100,
        "low": np.random.rand(100) * 100 + 100,
        "volume": np.random.randint(1000, 10000, 100)
    }
    df = pl.DataFrame(data)
    
    # Run your function
    processed_df = calculate_indicators_polars(df)
    
    # Assertions (The "Test")
    assert "feat_rsi" in processed_df.columns
    assert "feat_macd" in processed_df.columns
    # Check that after row 50 (when indicators stabilize), there are no nulls
    assert processed_df.slice(50).null_count().sum(axis=1).sum() == 0

# ==========================================
# 2. TEST MODEL ARCHITECTURE
# ==========================================
def test_model_forward_pass():
    """
    Verify the model accepts input (Batch, Seq, Feat) and outputs (Batch, 1)
    """
    BATCH_SIZE = 8
    SEQ_LEN = 30
    N_FEATURES = 13  # Your relative feature count
    
    # Create fake random tensor
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
    
    # Initialize Model
    model = CompactStockEncoder(
        d_in=N_FEATURES, 
        d_model=16, 
        nhead=2, 
        n_layers=1, 
        d_out=1
    )
    
    # Run forward pass
    output = model(dummy_input)
    
    # Check shape
    assert output.shape == (BATCH_SIZE, 1)