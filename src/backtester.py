import polars as pl
from .utils import get_logger, load_json, save_json
from .data_processor import DataProcessor
from torch.utils.data import DataLoader, TensorDataset
from .builder import build_model
import torch
from datetime import datetime
import torch.nn as nn
import joblib

class Backtester:
    def __init__(self, report_path: str):

        self.logger = get_logger(__name__)
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.train_report_path = report_path
        self.report = load_json(self.train_report_path)

        self.dp = DataProcessor(mode="test")
        self.metadata = self.dp.load_metadata(self.report["metadata_path"])

        self.model_type = self.report["model_type"].lower()
        self.seq_len = self.metadata["seq_len"]
        self.model_path = self.report["model"]["training_result"]["model_path"]
        self.feature_cols = self.metadata["features"]

        self.device = torch.device("cpu")
        self.model = None
        self.test_metadata_path = None

        self.threshhold = None

        self.time_horizon = None
        self.profit_pct = None
        self.stop_pct = None
        self.brokerage = None
        self.investment_per_stock = None
        self.backtest_results = None

        self.df_pnl = None

    def load_model(self):
        self.logger.info(f"Loading {self.model_type} model from {self.model_path}...")
        
        try:
            if self.model_type in ["rf", "xgboost"]:
                self.model = joblib.load(self.model_path)
            
            elif self.model_type in ["lstm", "transformer"]:
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                elif torch.cuda.is_available():
                    self.device = torch.device("cuda")

                hp = self.report['model']['hyperparameters']

                hp_args = hp.copy()
                hp_args.pop("model_type", None) 
                self.model, _ = build_model(model_type=self.model_type, **hp_args)

                state_dict = torch.load(self.model_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                
                self.model.to(self.device)
                
            self.logger.info(f"{self.model_type} model loaded successfully on {self.device}")
            return self.model

        except Exception as e:
            msg = f"Error while loading {self.model_type} model: {str(e)}"
            self.logger.error(msg)
            raise Exception(msg)

    def prepare_data(self, test_data_path: str):
        self.data = self.dp.load_data(test_data_path)
        self.data, features_cols = self.dp.calculate_indicators()
        self.test_metadata_path = self.dp.save_data(self.run_id)
        self.transformed_data = self.dp.transform_scaler(self.data)
        final_data = self.dp.create_windows(self.transformed_data, self.seq_len)

        if self.model_type == "rf" or self.model_type == "xgboost":
            return dp.reshape_for_ml(final_data), self.feature_cols
        else:
            tensor_data = torch.from_numpy(final_data).float()
            dataset = TensorDataset(tensor_data)
            return DataLoader(dataset, batch_size=128, shuffle=False), self.feature_cols

    def make_predictions(self, test_data, threshold=0.55):
        self.threshold = threshold
        if self.model_type in ["rf", "xgboost"]:
            return self._make_predictions_ml(test_data, threshold)

        probs_list = []
        self.model.eval()
    
        with torch.no_grad():
            for batch in test_data:
                x_batch = batch[0].to(self.device)
                logits = self.model(x_batch)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())

        probs_arr = np.concatenate(probs_list, axis=0).flatten()
        predictions = (probs_arr >= threshold).astype(int)
        
        self.logger.info(f"Inference complete. Generated {len(predictions)} predictions.")
        return self._get_predictions_df(probs_arr, predictions)
        
    def _make_predictions_ml(self, X_2d_numpy, threshold=0.55):
        """
        Specific for Random Forest and XGBoost (Sklearn Wrappers)
        """
        self.logger.info("Starting ML Inference...")

        if X_2d_numpy.ndim != 2:
            raise ValueError(f"ML models expect 2D input, got shape {X_2d_numpy.shape}")

        try:
            probs_arr = self.model.predict_proba(X_2d_numpy)[:, 1]
        except AttributeError:
            self.logger.warning("Model has no predict_proba, using predict instead.")
            probs_arr = self.model.predict(X_2d_numpy)

        predictions = (probs_arr >= threshold).astype(int)

        self.logger.info(f"Inference complete. Generated {len(predictions)} predictions.")
        return self._get_predictions_df(probs_arr, predictions)

    def _get_predictions_df(self, probs_arr, predictions):
        self.logger.info("Aligning predictions with original data...")

        aligned_df = self.data.group_by("Stock ID", maintain_order=True).map_groups(
            lambda df: df.slice(self.seq_len - 1, len(df)))

        if len(aligned_df) != len(predictions) or len(aligned_df) != len(probs_arr):
            self.logger.error(f"Mismatch! Aligned Data: {len(aligned_df)}, Preds: {len(predictions)}, Probs: {len(probs_arr)}")
            raise ValueError("Data alignment failed. Check your windowing logic vs slicing logic.")

        return aligned_df.with_columns([
            pl.Series("probability", probs_arr),
            pl.Series("prediction", predictions)
        ])

    def run_simulation(self, data, time_horizon, profit_pct, brokerage, stop_pct, investment_per_stock):
        self.logger.info("Starting simulation...")

        self.time_horizon = time_horizon
        self.profit_pct = profit_pct
        self.brokerage = brokerage
        self.stop_pct = stop_pct
        self.investment_per_stock = investment_per_stock

        df = self.dp.create_triple_barrier_target(data, profit_pct, stop_pct, time_horizon)

        selling_price_expr = (
            pl.col("close").shift(-time_horizon).over("Stock ID")
        ).alias("selling_price")

        entry_price_expr = (
            pl.col("open").shift(-1).over("Stock ID")
        ).alias("entry_price")

        df = df.with_columns([selling_price_expr, entry_price_expr])

        pnl_expr = (
            pl.when((pl.col("prediction") == 1) & (pl.col("target") == 1))
            .then(profit_pct)  # Hit Profit Target
            .when((pl.col("prediction") == 1) & (pl.col("target") == -1))
            .then(-stop_pct) # Hit Stop Loss
            .when((pl.col("prediction") == 1) & (pl.col("target") == 0))
            .then((pl.col("selling_price") - pl.col("entry_price")) / pl.col("entry_price"))
            .otherwise(0.0) # No trade
        ).alias("pnl_pct")

        brokerage_expr = (
            pl.when(pl.col("prediction") == 1)
            .then(investment * brokerage) 
            .otherwise(0.0)
        ).alias("brokerage_cost")

        self.df_pnl = df.with_columns([
            pnl_expr,
            brokerage_expr
        ]).with_columns([
            (pl.col("pnl_pct") * investment).alias("gross_pnl")
        ]).with_columns([
            (pl.col("gross_pnl") - pl.col("brokerage_cost")).alias("net_pnl")
        ])

        total_trades = self.df_pnl.filter(pl.col("prediction") == 1).height
        total_wins = self.df_pnl.filter((pl.col("prediction") == 1) & (pl.col("target") == 1)).height
        total_pnl = self.df_pnl["net_pnl"].sum()
        neg_trades = self.df_pnl.filter((pl.col("prediction") == 1) & (pl.col("target") == -1)).height
        
        self.backtest_results = {
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_pnl": total_pnl,
            "neg_trades": neg_trades
        }
        return self.df_pnl, self.backtest_results

    def save_results(self):

        df_pnl_path = os.path.join("artifacts", "data", "simulation")
        os.makedirs(df_pnl_path, exist_ok=True)
        df_pnl_filename = f"{self.run_id}_df_pnl.parquet"
        df_pnl_full_path = os.path.join(df_pnl_path, df_pnl_filename)
        self.df_pnl.write_parquet(df_pnl_full_path)
        self.logger.info(f"Saved PnL Data to {df_pnl_full_path}")

        backtest_reports_path = os.path.join("artifacts", "reports", "backtest")
        os.makedirs(backtest_reports_path, exist_ok=True)
        backtest_report_filename = f"{self.run_id}_{self.model_type}_backtest_report.json"
        backtest_report_full_path = os.path.join(backtest_reports_path, backtest_report_filename)

        self.backtest_report = {
            "run_id": self.run_id,
            "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model_type": self.model_type,
            "backtest_results": self.backtest_results,
            "time_horizon": self.time_horizon,
            "profit_pct": self.profit_pct,
            "brokerage": self.brokerage,
            "stop_pct": self.stop_pct,
            "investment_per_stock": self.investment_per_stock,
            "paths": {
                "df_pnl_path": df_pnl_full_path,
                "test_metadata_path": self.test_metadata_path,
                "train_report_path": self.train_report_path,
            }
        }

        save_json(backtest_report_full_path, self.backtest_report)
        self.logger.info(f"Saved Backtest Report to {backtest_report_full_path}")

        return self.backtest_report
        