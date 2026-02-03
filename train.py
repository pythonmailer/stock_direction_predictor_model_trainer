from src import DataProcessor, DeepLearningTrainer, Backtester, train_ml_model, build_model, save_json, setup_logging
from datetime import datetime
import polars as pl
import os
import torch
import torch.nn as nn
import numpy as np
import dagshub
import mlflow
import sys

def get_date():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":

    setup_logging()
    
    # === CONFIG VARIABLES ===
    data_filename = "nifty500_no_adani_kite.parquet"
    test_data_path = "nifty500_no_adani_kite_test.parquet"

    n_stocks = 5
    
    # Hyperparameters & Settings
    profit_pct = 0.03
    stop_pct = 0.01
    time_horizon = 5
    seq_len = 30
    learning_rate = 0.01
    epochs = 1
    patience = 10
    model_architecture = "lstm"

    dagshub.init(repo_owner='pythonmailer156', repo_name='stock_direction_predictor_model_trainer', mlflow=True)
    mlflow.set_experiment("Stock_Direction_Predictor")

    run_id = get_date()
    
    # with mlflow.start_run(run_name=f"Train_Data_Prep_{run_id}") as train_data_run:

    #     dp = DataProcessor()
    #     dp.load_data(data_filename, n_stocks)
    #     data, features = dp.calculate_indicators() 
        
    #     # Create Target
    #     data_tg = dp.create_binary_target(data, profit_pct, stop_pct, time_horizon)
        
    #     # Split & Fit Scaler
    #     train, val, split_date = dp.time_based_split(seq_len)
    #     scaler = dp.fit_scaler(train, features)

    #     # Save data and metadata
    #     dp.save_data(run_id)

    #     train_data_run_id = train_data_run.info.run_id

    # # 3. Transform & Windowing
    # train_scaled = dp.transform_scaler(train)
    # val_scaled = dp.transform_scaler(val)
        
    # train_X, train_y = dp.create_windows(train_scaled, features, seq_len)
    # val_X, val_y = dp.create_windows(val_scaled, features, seq_len)

    # with mlflow.start_run(run_name=f"Model_Training_{model_architecture}_{run_id}") as training_run:

    #     mlflow.set_tag("train_data_run_id", train_data_run_id) 

    #     # 4. Build Model & Training Setup
    #     if torch.backends.mps.is_available():
    #         device = torch.device("mps")
    #     elif torch.cuda.is_available():
    #         device = torch.device("cuda")
    #     else:
    #         device = torch.device("cpu")

    #     # Handle Class Imbalance
    #     num_pos = np.sum(train_y == 1).item()
    #     num_neg = np.sum(train_y == 0).item()
    #     ratio_val = num_neg / (num_pos + 1e-6)
    #     pos_weight = torch.tensor(ratio_val, dtype=torch.float32).to(device)

    #     model, hp = build_model(model_architecture, feature_cols=features, hidden_size=128, n_layers=2, dropout=0.1)
    #     mlflow.log_params(hp) # Log internal model architecture params

    #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #     train_loader, val_loader = dp.to_loaders(train_X, train_y, val_X, val_y)

    #     input_example = np.random.rand(1, seq_len, len(features)).astype(np.float32)

    #     threshold = 0.55
        
    #     trainer = DeepLearningTrainer(model, criterion, optimizer, device, epochs, patience, input_example, hp.get("model_type", "dl_model"), threshold)
        
    #     model, result = trainer.train(train_loader, val_loader)

    #     mlflow.log_params(hp)
    #     mlflow.log_params({
    #         "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    #         "ratio_val": ratio_val if ratio_val else None,
    #         "pos_weight": pos_weight.item() if pos_weight else None,
    #         "criterion": criterion.__class__.__name__,
    #         "optimizer": optimizer.__class__.__name__,
    #         "learning_rate": learning_rate,
    #         "threshgold": result.get("threshhold", None),
    #     })
    #     mlflow.log_metrics(result.get("metrics", {}))

    #     training_run_id = training_run.info.run_id

    with mlflow.start_run(run_name=f"Backtesting_{run_id}") as backtest_run:
        training_run_id = "0a9b1908523d40eca0b37261138cafb2"

        bt = Backtester(training_run_id)
        loaded_model = bt.load_model()
        test_data, test_feature_cols = bt.prepare_data(test_data_path)
        predictions = bt.make_predictions(test_data, threshhold=0.55)
        pnl_df, bt_result = bt.run_simulation(predictions, time_horizon=time_horizon, stop_pct=stop_pct, \
            profit_pct=profit_pct, brokerage=0.01, investment_per_stock=5000)
        mlflow.set_tag("training_run_id", training_run_id) 
        bt.save_results()
