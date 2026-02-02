import polars as pl
from .utils import get_logger, load_json, save_json
from .data_processor import DataProcessor
from torch.utils.data import DataLoader, TensorDataset
from .builder import build_model
import torch
import plotly.graph_objects as go
import os
import mlflow
import numpy as np
from datetime import datetime
import torch.nn as nn
import joblib

class Backtester:
    def __init__(self):

        self.logger = get_logger(__name__)
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.run_dict = {}
        
        self.dp = DataProcessor(mode="test")

        self.model_id = None
        self.training_params = None
        self.training_metrics = None
        self.common_stocks = None

        self.model_type = None
        self.data = None

        self.device = torch.device("cpu")
        self.model = None
        self.test_metadata_path = None

        self.threshold = None

        self.results_df = None

        self.time_horizon = None
        self.profit_pct = None
        self.stop_pct = None
        self.brokerage = None
        self.investment_per_stock = None
        self.backtest_results = None

        self.df_pnl = pl.DataFrame()

    def get_all_models(self, author, purpose):
        query = f"tags.author = '{author}' AND tags.purpose = '{purpose}'"
        runs = mlflow.search_runs(filter_string=query, output_format="list")
        self.run_dict = {run.info.run_name: run.info.run_id for run in runs}
        options = list(self.run_dict.keys())
        return options

    def load_using_dp(self, dp, model):
        self.dp = dp
        self.dp.mode = "test"
        self.model = model

    def load_using_mlrun_id(self, training_run_id):
        training_run = mlflow.get_run(training_run_id)
        self.dp = DataProcessor(mode="test")
        self.dp.load_config(training_run.data.tags['train_data_run_id'])
        self.model_type = training_run.data.params['model_type']
        model_id = training_run.outputs.model_outputs[0].model_id
        self.training_params = training_run.data.params
        self.training_metrics = training_run.data.metrics

        if self.model_type in ["lstm", "transformer"]:
            self.model = mlflow.pytorch.load_model(f"models:/{model_id}")
        elif self.model_type == "xgboost":
            self.model = mlflow.xgboost.load_model(f"models:/{model_id}")
        else:
            self.model = mlflow.sklearn.load_model(f"models:/{model_id}")
        

    def load_model(self):
        self.logger.info(f"Loading {self.model_type} model from Dagshub...")
        
        try:
            # if self.model_type in ["rf", "xgboost"]:
            #     self.model = joblib.load(self.model_path)
            
            # elif self.model_type in ["lstm", "transformer"]:
            #     if torch.backends.mps.is_available():
            #         self.device = torch.device("mps")
            #     elif torch.cuda.is_available():
            #         self.device = torch.device("cuda")

            #     hp = self.report['model']['hyperparameters']

            #     hp_args = hp.copy()
            #     hp_args.pop("model_type", None) 
            #     self.model, _ = build_model(model_type=self.model_type, **hp_args)

            #     state_dict = torch.load(self.model_path, map_location="cpu")
            #     self.model.load_state_dict(state_dict)
                
            #     self.model.to(self.device)
                
            # self.logger.info(f"{self.model_type} model loaded successfully on {self.device}")
            # return self.model

            if self.model_type in ["lstm", "transformer"]:
                self.model = mlflow.pytorch.load_model(f"models:/{self.model_id}")
            elif self.model_type == "xgboost":
                self.model = mlflow.xgboost.load_model(f"models:/{self.model_id}")
            else:
                self.model = mlflow.sklearn.load_model(f"models:/{self.model_id}")
                
        except Exception as e:
            msg = f"Error while loading {self.model_type} model: {str(e)}"
            self.logger.error(msg)
            raise Exception(msg)

    def run_inference(self, test_data_path: str, train_ratio):
        """
        Loads data, preprocesses it, and runs model inference to get probabilities.
        Returns the DataFrame with a 'probability' column (but NO 'prediction' column yet).
        """
        self.logger.info(f"Starting inference pipeline on {test_data_path}...")

        self.data = self.dp.load_data(test_data_path, len(self.dp.stocks), self.dp.stocks)
        self.data, features = self.dp.calculate_indicators(self.dp.feature_cols)
        _, self.data = self.dp.time_based_split(self.dp.seq_len, train_ratio)
        self.transformed_data = self.dp.transform_scaler(self.data)
        final_data = self.dp.create_windows(self.transformed_data, features, self.dp.seq_len)

        probs_arr = None

        if self.model_type in ["rf", "xgboost"]:

            X_input = self.dp.reshape_for_ml(final_data)
            
            self.logger.info(f"Running ML Inference on shape {X_input.shape}...")
            try:
                probs_arr = self.model.predict_proba(X_input)[:, 1]
            except AttributeError:
                self.logger.warning("Model has no predict_proba, using hard predictions as probs.")
                probs_arr = self.model.predict(X_input).astype(float)

        else:

            tensor_data = torch.from_numpy(final_data).float()
            dataset = TensorDataset(tensor_data)
            loader = DataLoader(dataset, batch_size=512, shuffle=False)
            
            self.logger.info(f"Running DL Inference on {len(dataset)} samples...")
            
            probs_list = []
            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    x_batch = batch[0].to(self.device)
                    logits = self.model(x_batch)
                    probs = torch.sigmoid(logits)
                    probs_list.append(probs.cpu().numpy())
            
            probs_arr = np.concatenate(probs_list, axis=0).flatten()

        aligned_df = self.data.group_by("Stock ID", maintain_order=True).map_groups(
            lambda df: df.slice(self.dp.seq_len - 1, len(df))
        )

        if len(aligned_df) != len(probs_arr):
            raise ValueError(f"Shape Mismatch: Data {len(aligned_df)} != Probs {len(probs_arr)}")

        self.results_df = aligned_df.with_columns(
            pl.Series("probability", probs_arr)
        )
        
        self.logger.info("Inference pipeline complete.")
        return self.results_df

    def update_target(self, df: pl.DataFrame, profit_pct, stop_pct, time_horizon):

        self.logger.info(f"Updating target with profit_pct: {profit_pct}, stop_pct: {stop_pct}, time_horizon: {time_horizon}...")

        self.time_horizon = time_horizon
        self.profit_pct = profit_pct
        self.stop_pct = stop_pct
        
        self.results_df = self.dp.create_triple_barrier_target(df, self.profit_pct, self.stop_pct, self.time_horizon)

        return self.results_df

    def update_threshold(self, df: pl.DataFrame, threshold: float = 0.55):
        """
        Takes the dataframe with 'probability' and adds/updates the 'prediction' column
        based on the new threshold. Does NOT re-run the model.
        """
        if "probability" not in df.columns:
            raise ValueError("DataFrame must contain 'probability' column. Run 'run_inference' first.")
            
        self.logger.info(f"Applying threshold: {threshold}")
        self.threshold = threshold

        self.results_df = df.with_columns(
            pl.when(pl.col("probability") >= threshold)
            .then(1)
            .otherwise(0)
            .alias("prediction")
        )
        return self.results_df

    def get_common_stocks(self, filename):
        all_stocks = self.dp.get_all_stocks(filename)

        self.common_stocks = [s for s in all_stocks if s in self.dp.stocks]

        return self.common_stocks

    def analyze_thresholds(self, df: pl.DataFrame, min_thresh=0.5, max_thresh=0.95, step=0.01):
        """
        Calculates Win Rate and Trade Count for every threshold step.
        Returns a DataFrame ready for plotting.
        """
        results = []
        
        if "target" not in df.columns:
            raise ValueError("DataFrame must have a 'target' column to calculate wins.")

        thresholds = np.arange(min_thresh, max_thresh, step)
        
        for t in thresholds:
            trades = df.filter(pl.col("probability") >= t)
            
            total_trades = len(trades)
            
            if total_trades > 0:
                wins = trades.filter(pl.col("target") == 1).height
                win_rate = (wins / total_trades) * 100
            else:
                wins = 0
                win_rate = 0.0
            
            results.append({
                "Threshold": round(t, 2),
                "Total Trades": total_trades,
                "Win Count": wins,
                "Win Rate (%)": win_rate
            })
            
        return pl.DataFrame(results)

    def run_simulation(self, data, time_horizon, profit_pct, brokerage, stop_pct, investment_per_stock, threshold):
        self.logger.info("Starting simulation...")

        self.time_horizon = time_horizon
        self.profit_pct = profit_pct
        self.brokerage = brokerage
        self.threshold = threshold
        self.stop_pct = stop_pct
        self.investment_per_stock = investment_per_stock

        data = self.update_threshold(data, threshold)
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
            .then(investment_per_stock * brokerage) 
            .otherwise(0.0)
        ).alias("brokerage_cost")

        self.df_pnl = df.with_columns([
            pnl_expr,
            brokerage_expr
        ]).with_columns([
            (pl.col("pnl_pct") * investment_per_stock).alias("gross_pnl")
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

    def plot_equity_curve(self, df_pnl):
        """
        Aggregates PnL by date and plots cumulative Gross vs Net PnL.
        """
        # 1. Aggregate PnL by Date (Summing up multiple trades on the same day)
        # Ensure "date" column exists (adjust case if it's "Date")
        date_col = "datetime" if "datetime" in df_pnl.columns else "Date"
        
        daily_pnl = (
            df_pnl.group_by(date_col)
            .agg([
                pl.col("gross_pnl").sum(),
                pl.col("net_pnl").sum()
            ])
            .sort(date_col) # Sort by time so the line flows correctly
        )

        # 2. Calculate Cumulative Sum (The Equity Curve)
        equity_curve = daily_pnl.with_columns([
            pl.col("gross_pnl").cum_sum().alias("cum_gross"),
            pl.col("net_pnl").cum_sum().alias("cum_net")
        ])

        # 3. Convert to Pandas for Plotly
        plot_data = equity_curve.to_pandas()

        # 4. Create Plotly Chart
        fig = go.Figure()

        # Gross PnL Line (Dashed, Lighter)
        fig.add_trace(go.Scatter(
            x=plot_data[date_col], 
            y=plot_data['cum_gross'],
            mode='lines',
            name='Gross PnL (Pre-Fees)',
            line=dict(color='gray', width=2, dash='dash')
        ))

        # Net PnL Line (Solid, Colored based on profit)
        final_pnl = plot_data['cum_net'].iloc[-1]
        line_color = '#00FA9A' if final_pnl >= 0 else '#FF4B4B'  # Green or Red
        
        fig.add_trace(go.Scatter(
            x=plot_data[date_col], 
            y=plot_data['cum_net'],
            mode='lines',
            name='Net PnL (Realized)',
            fill='tozeroy', # Fill area under curve for visual impact
            line=dict(color=line_color, width=3)
        ))

        fig.update_layout(
            title="Strategy Equity Curve (Net vs Gross)",
            xaxis_title="Date",
            yaxis_title="Total PnL (₹)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        return fig

    def save_results(self):
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        df_pnl_path = os.path.join("artifacts", "data", "simulation")
        os.makedirs(df_pnl_path, exist_ok=True)
        df_pnl_filename = f"{run_id}_df_pnl.parquet"
        df_pnl_full_path = os.path.join(df_pnl_path, df_pnl_filename)
        self.df_pnl.write_parquet(df_pnl_full_path)
        self.logger.info(f"Saved PnL Data to {df_pnl_full_path}")

        mlflow.log_params({
            "created_at": run_id,
            "time_horizon": self.time_horizon,
            "profit_pct": self.profit_pct,
            "brokerage": self.brokerage,
            "stop_pct": self.stop_pct,
            "investment_per_stock": self.investment_per_stock,
            "threshhold": self.threshhold,
            "model_type": self.model_type,
        })

        mlflow.log_metrics(self.backtest_results)

        metadata = {"df_pnl_path": df_pnl_full_path,}

        mlflow.log_dict(metadata, "test_metadata.json")

        return