from src import DataProcessor, DeepLearningTrainer, Backtester, train_ml_model, build_model, save_json
from datetime import datetime
import polars as pl
import os
import torch
import torch.nn as nn
import numpy as np

import sys

if __name__ == "__main__":
    
    # Set variables
    data_filename = "nifty500_no_adani_kite.parquet"
    profit_pct = 0.03
    stop_pct = 0.01
    time_horizon = 5
    seq_len = 30

    test_data_path = "nifty500_no_adani_kite_test.parquet"

    # Load and process data
    dp = DataProcessor()
    data, features = dp.load_and_process(data_filename)
    data_tg = dp.create_binary_target(data, profit_pct, stop_pct, time_horizon)
    train, val, split_date = dp.time_based_split(data_tg, seq_len)
    scaler = dp.fit_scaler(train, features)

    # Save data and metadata
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metadata_path = dp.save_data(train, val, run_id, scaler)

    # metadata = {
    #         "run_id": run_id,
    #         "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),

    #         "split_date": self.split_date,
    #         "feature_cols": self.feature_cols,
    #         "train_shape": self.train.shape,
    #         "val_shape": self.val.shape,
    #         "init_cols": self.init_cols,
    #         "train_stocks": self.train_stocks,
    #         "seq_len": self.seq_len,
    #         "time_horizon": self.time_horizon,
    #         "profit_pct": self.profit_pct,
    #         "stop_pct": self.stop_pct,
            
    #         "paths": {
    #             "train_data_path": train_data_full_path,
    #             "val_data_path": val_data_full_path,
    #             "scaler_path": scaler_full_path,
    #             "raw_data_used": self.full_path,
    #         }
    # }
    
    # Scale data, create windows and prepare for training
    train_scaled = dp.transform_scaler(train, scaler, features)
    val_scaled = dp.transform_scaler(val, scaler, features)
    train_X, train_y = dp.create_windows(train_scaled, features, seq_len)
    val_X, val_y = dp.create_windows(val_scaled, features, seq_len)
    train_loader, val_loader = dp.to_loaders(train_X, train_y, val_X, val_y)

    # Build Model
    learning_rate = 0.01
    epochs = 100
    patience = 10

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_pos = np.sum(train_y == 1).item()
    num_neg = np.sum(train_y == 0).item()

    ratio_val = num_neg / (num_pos + 1e-6)
    pos_weight = torch.tensor(ratio_val, dtype=torch.float32).to(device)

    model, hp = build_model("lstm", feature_cols=features, hidden_size=128, n_layers=2, dropout=0.1)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train Model
    trainer = DeepLearningTrainer(model, criterion, optimizer, device, epochs, patience, run_id, hp.get("model_type", "dl_model"))
    model, result = trainer.train(train_loader, val_loader)

    # Save Report
    report = {
        "run_id": run_id,
        "model_type": result["model_type"],
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "metadata_path": metadata_path,
        "model": {
            "hyperparameters": hp,
            "ratio_val": ratio_val if ratio_val else None,
            "pos_weight": pos_weight.item() if pos_weight else None,
            "learning_rate": learning_rate,
            "criterion": criterion.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "training_result": result,
        }
    }

    report_base_path = f"artifacts/reports"

    train_report_filename = f"{run_id}_{result['model_type']}_report.json"
    report_save_path = os.path.join(report_base_path, "train", train_report_filename)
    save_json(report_save_path, report)

    bt = Backtester(report_save_path)
    loaded_model =bt.load_model()
    
    test_data, test_features =bt.prepare_data(test_data_path)
    predictions_df = bt.make_predictions(test_data, 0.55)
    df_pnl = bt.run_simulation(predictions_df, capital=100000, time_horizon=5, profit_pct=0.03, brokerage=0.05, stop_pct=0.01, investment=10000)
