import os
import torch
import random
import mlflow
import numpy as np
import polars as pl
from datetime import datetime
from .utils import get_logger, save_json, load_json
from torch.utils.data import DataLoader, TensorDataset
from numpy.lib.stride_tricks import sliding_window_view


class DataProcessor:
    def __init__(self, mode="train"):
        self.logger = get_logger(__name__)

        self.mode = mode

        self.full_path = None
        self.req_cols = ['Stock ID', 'datetime', 'close', 'volume', 'high', 'low', 'open']
        self.stocks = []
        self.train_shape = None
        self.val_shape = None

        self.feature_cols = []

        self.time_horizon = None
        self.profit_pct = None
        self.stop_pct = None
        self.train_ratio = None

        self.seq_len = None

        self.train_pos_count = None
        self.train_neg_count = None
        self.val_pos_count = None
        self.val_neg_count = None
        
        self.df = pl.DataFrame()
        self.val = pl.DataFrame()
        self.train = pl.DataFrame()
        self.scaler = pl.DataFrame()

    # ==========================================
    # 1. Data Loading & Feature Engineering
    # ==========================================
    def load_data(self, data_filename: str, n_stocks=1, stocks=[]):
        """
        Loads data and filters for specific stocks.
        To filter stocks, pass a list of stock IDs.
        To select random n stocks, pass an int.
        Data should be located in data folder.
        Data should be in parquet format.
        """
        self.full_path = os.path.join("data", data_filename)
        self.logger.info(f"Loading data from {self.full_path}...")
        
        try:
            q = pl.scan_parquet(self.full_path)
        except FileNotFoundError:
            self.logger.error(f"❌ File not found: {self.full_path}")
            raise

        if len(stocks) >= n_stocks:
            self.stocks = stocks[:n_stocks]
        
        if len(stocks) < n_stocks:
            temp_stocks = self.get_all_stocks(data_filename)

            needed = n_stocks - len(stocks)

            available_pool = list(set(temp_stocks) - set(stocks))

            if len(available_pool) > needed:
                extras = random.sample(available_pool, needed)
            else:
                extras = available_pool
                
            self.stocks = stocks + extras

        q = q.filter(pl.col("Stock ID").is_in(self.stocks))
        df = q.collect()

        if not all(c in df.columns for c in self.req_cols):
            raise ValueError(f"Missing required columns: {self.req_cols}")

        cols_to_cast = ['close', 'high', 'low', 'open', 'volume']

        self.df = df.with_columns([pl.col(c).cast(pl.Float32) for c in cols_to_cast])
        
        print(f"Loaded {len(self.df)} rows.")
        return self.df

    def calculate_indicators(self, features_cols: list[str] = None) -> pl.DataFrame:
        """
        Computes technical indicators using Polars expressions dynamically.
        Only calculates what is requested in features_cols to save memory.
        """
        try:
            self.logger.info("Calculating technical indicators...")
            df = self.df.sort(["Stock ID", "datetime"])
            EPS = 1e-9
            
            # Fallback to self.feature_cols if none provided
            features_cols = features_cols if features_cols else self.feature_cols
            features_cols = features_cols if features_cols else self.all_features()
            
            # Convert to set for O(1) lookup
            req_feats = set(features_cols)

            # --- 1. Identify Necessary Intermediates ---
            # We define boolean flags to check if a "base" calculation is needed
            
            # prev_close is needed for: rel_open, rel_high, rel_low, rel_close, atr
            need_prev_close = bool(req_feats & {"rel_open", "rel_high", "rel_low", "rel_close", "atr"})
            
            # prev_vol is needed for: rel_vol
            need_prev_vol = "rel_vol" in req_feats
            
            # sma_20 is needed for: bb_width, dist_sma20
            need_sma_20 = bool(req_feats & {"bb_width", "dist_sma20"})
            
            # bb_std is needed for: bb_width
            need_bb_std = "bb_width" in req_feats
            
            # min/max windows needed for: stoch
            need_min_max = "stoch" in req_feats
            
            # Build list of intermediate expressions
            intermediates = []
            
            if need_prev_close:
                intermediates.append(pl.col("close").shift(1).over("Stock ID").alias("prev_close"))
            if need_prev_vol:
                intermediates.append(pl.col("volume").shift(1).over("Stock ID").alias("prev_vol"))
            if need_sma_20:
                intermediates.append(pl.col("close").rolling_mean(20).over("Stock ID").alias("sma_20"))
            if need_bb_std:
                intermediates.append(pl.col("close").rolling_std(20).over("Stock ID").alias("bb_std"))
            if need_min_max:
                intermediates.append(pl.col("low").rolling_min(14).over("Stock ID").alias("low_min"))
                intermediates.append(pl.col("high").rolling_max(14).over("Stock ID").alias("high_max"))

            # Calculate intermediates in one optimized batch
            if intermediates:
                df = df.with_columns(intermediates)

            # --- 2. Calculate Final Features ---
            expressions = []

            # Relative Price Features
            if "rel_open" in req_feats:
                expressions.append(((pl.col("open") / pl.col("prev_close")) - 1).cast(pl.Float32).alias("rel_open"))
            if "rel_high" in req_feats:
                expressions.append(((pl.col("high") / pl.col("prev_close")) - 1).cast(pl.Float32).alias("rel_high"))
            if "rel_low" in req_feats:
                expressions.append(((pl.col("low") / pl.col("prev_close")) - 1).cast(pl.Float32).alias("rel_low"))
            if "rel_close" in req_feats:
                expressions.append(((pl.col("close") / pl.col("prev_close")) - 1).cast(pl.Float32).alias("rel_close"))
            
            # Relative Volume
            if "rel_vol" in req_feats:
                expressions.append((pl.col("volume") / (pl.col("prev_vol") + EPS)).log().cast(pl.Float32).alias("rel_vol"))

            # ATR (Normalized) - Note: We calc True Range inline to save an intermediate column
            if "atr" in req_feats:
                tr1 = pl.col("high") - pl.col("low")
                tr2 = (pl.col("high") - pl.col("prev_close")).abs()
                tr3 = (pl.col("low") - pl.col("prev_close")).abs()
                true_range = pl.max_horizontal(tr1, tr2, tr3)
                expressions.append((true_range.rolling_mean(14).over("Stock ID") / (pl.col("close") + EPS)).cast(pl.Float32).alias("atr"))

            # Stochastic
            if "stoch" in req_feats:
                stoch_k = 100 * ((pl.col("close") - pl.col("low_min")) / (pl.col("high_max") - pl.col("low_min") + EPS))
                expressions.append((stoch_k / 100.0).cast(pl.Float32).alias("stoch"))

            # RSI
            if "rsi" in req_feats:
                delta = pl.col("close").diff()
                up = delta.clip(lower_bound=0)
                down = delta.clip(upper_bound=0).abs()
                roll_up = up.ewm_mean(span=14, adjust=False).over("Stock ID")
                roll_down = down.ewm_mean(span=14, adjust=False).over("Stock ID")
                rsi_val = 100.0 - (100.0 / (1.0 + (roll_up / (roll_down + EPS))))
                expressions.append((rsi_val / 100.0).cast(pl.Float32).alias("rsi"))

            # Rate of Change (ROC)
            if "roc_5" in req_feats:
                expressions.append(pl.col("close").pct_change(5).over("Stock ID").cast(pl.Float32).alias("roc_5"))
            if "roc_10" in req_feats:
                expressions.append(pl.col("close").pct_change(10).over("Stock ID").cast(pl.Float32).alias("roc_10"))

            # Bollinger Width
            if "bb_width" in req_feats:
                expressions.append(((4 * pl.col("bb_std")) / (pl.col("sma_20") + EPS)).cast(pl.Float32).alias("bb_width"))

            # Distance to SMA
            if "dist_sma20" in req_feats:
                expressions.append(((pl.col("close") / pl.col("sma_20")) - 1).cast(pl.Float32).alias("dist_sma20"))

            # Apply batch
            if expressions:
                df = df.with_columns(expressions)

            # --- 3. MACD Special Handling ---
            # MACD involves sequential steps that are cleaner to keep separate
            if "macd" in req_feats:
                # 1. EMAs
                df = df.with_columns([
                    pl.col("close").ewm_mean(span=12, adjust=False).over("Stock ID").alias("_ema_12"),
                    pl.col("close").ewm_mean(span=26, adjust=False).over("Stock ID").alias("_ema_26"),
                ])
                # 2. MACD Line
                df = df.with_columns([
                    (pl.col("_ema_12") - pl.col("_ema_26")).alias("_macd_line")
                ])
                # 3. Signal Line & Final Normalized MACD
                df = df.with_columns([
                    pl.col("_macd_line").ewm_mean(span=9, adjust=False).over("Stock ID").alias("_signal_line")
                ])
                df = df.with_columns([
                    ((pl.col("_macd_line") - pl.col("_signal_line")) / (pl.col("close") + EPS)).cast(pl.Float32).alias("macd")
                ])

            # --- 4. Final Cleanup ---
            # Only select the columns we actually wanted + essential data
            
            # Find which requested features actually exist (safety check)
            self.feature_cols = [f for f in features_cols if f in df.columns]
            
            # Combine without duplicates
            final_cols = list(dict.fromkeys(self.req_cols + self.feature_cols))
            
            self.df = df.select(final_cols).drop_nulls()

            self.logger.info(f"Added Indicators successfully. Shape: {self.df.shape}")

            return self.df, self.feature_cols

        except Exception as e:
            self.logger.error(f"Error in indicators: {e}")
            raise

    # ==========================================
    # 2. Target Creation
    # ==========================================
    def create_binary_target(self, df, profit_pct=0.03, stop_pct=0.015, time_horizon=5):
        self.logger.info("Creating Target...")

        df = self.create_triple_barrier_target(df, profit_pct, stop_pct, time_horizon)

        self.df = df.with_columns((pl.col("target") == 1).cast(pl.Int8).alias("target"))

        return self.df

    def create_triple_barrier_target(self, df, profit_pct=0.03, stop_pct=0.015, time_horizon=5):
        self.logger.info("Creating Triple Barrier Target...")
        df = df.sort(["Stock ID", "datetime"])

        self.profit_pct = profit_pct
        self.stop_pct = stop_pct
        self.time_horizon = time_horizon
        
        future_windows = []
        for i in range(1, time_horizon + 1):
            future_windows.append(pl.col("high").shift(-i).over("Stock ID").alias(f"h_{i}"))
            future_windows.append(pl.col("low").shift(-i).over("Stock ID").alias(f"l_{i}"))
        
        # Add next day open as entry price (realistic backtest assumption)
        entry_price = pl.col("open").shift(-1).over("Stock ID")
        
        df = df.with_columns(future_windows + [entry_price.alias("entry_price")])

        outcome = pl.lit(0)
        
        for i in reversed(range(1, time_horizon + 1)):
            stop_hit = pl.col(f"l_{i}") <= (pl.col("entry_price") * (1 - stop_pct))
            profit_hit = pl.col(f"h_{i}") >= (pl.col("entry_price") * (1 + profit_pct))
            
            outcome = pl.when(stop_hit).then(-1)\
                        .when(profit_hit).then(1)\
                        .otherwise(outcome)

        # 3. Finalize
        df = df.with_columns(outcome.alias("target"))
        
        # Drop rows where we don't have enough future data (the last 'time_horizon' rows)
        df = df.filter(pl.col(f"l_{time_horizon}").is_not_null())
        
        # Cleanup
        cols_to_drop = [f"h_{i}" for i in range(1, time_horizon+1)] + \
                       [f"l_{i}" for i in range(1, time_horizon+1)] + ["entry_price"]
        
        self.df = df.drop(cols_to_drop).with_columns(pl.col("target").cast(pl.Int8))

        return self.df

    # ==========================================
    # 3. Splitting & Scaling (Corrected)
    # ==========================================
    def time_based_split(self, seq_len=30, train_ratio=0.8):
        self.logger.info("Performing Time-Based Split...")
        self.train_ratio = train_ratio

        dates = self.df.select("datetime").unique().sort("datetime")["datetime"].to_list()
        split_idx = int(len(dates) * train_ratio)

        split_date = dates[split_idx]
        self.seq_len = seq_len
        
        self.logger.info(f"Split Date: {split_date}")
        
        self.train = self.df.filter(pl.col("datetime") < split_date)
        
        # Include buffer for validation to allow windowing at the boundary
        buffer_date = dates[max(0, split_idx - seq_len + 1)]
        self.val = self.df.filter(pl.col("datetime") >= buffer_date)

        if self.mode == "train":

            self.train_pos_count = self.train["target"].sum()  # Sum of 1s = count of 1s
            self.train_neg_count = self.train.height - self.train_pos_count

            self.val_pos_count = self.val["target"].sum()
            self.val_neg_count = self.val.height - self.val_pos_count

            self.train_shape = self.train.shape
            self.val_shape = self.val.shape
        
        return self.train, self.val

    def fit_scaler(self, df, features):
        """
        Fits scaler on TRAIN, applies to TRAIN and VAL. 
        This prevents data leakage.
        """
        self.logger.info("Fitting Scaler on Data...")
        
        # 1. Calculate Stats on Train (Median & IQR per Stock)
        self.scaler = df.group_by("Stock ID").agg([
            pl.col(c).median().alias(f"{c}_med") for c in features
        ] + [
            (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)).alias(f"{c}_iqr") for c in features
        ])
        
        return self.scaler

    def transform_scaler(self, df_pl: pl.DataFrame):

        self.logger.info("Transforming Data using scaler...")

        df = df_pl.join(self.scaler, on="Stock ID", how="left")
        exprs = []
        for c in self.feature_cols:
            # (Val - Median) / IQR
            scaled = (pl.col(c) - pl.col(f"{c}_med")) / (pl.col(f"{c}_iqr") + 1e-9)
            exprs.append(scaled.clip(-3.0, 3.0).alias(c))
            
        # Select only original columns + scaled features (drop stat cols)
        keep = [col for col in df.columns if "_med" not in col and "_iqr" not in col]
        return df.with_columns(exprs).select(keep)

    def inverse_transform(self, df_scaled: pl.DataFrame, features: list[str]):
        """
        Reverses the Robust Scaling to recover the original values.
        """
        self.logger.info("Inverse transforming (unscaling) data...")
        
        # 1. Join the stats (Median & IQR) back to the data
        df = df_scaled.join(self.scaler, on="Stock ID", how="left")
        
        exprs = []
        for c in features:
            if c in df.columns:
                original_value = (pl.col(c) * (pl.col(f"{c}_iqr") + 1e-9)) + pl.col(f"{c}_med")
                exprs.append(original_value.alias(c))
        
        keep_cols = [col for col in df.columns if "_med" not in col and "_iqr" not in col]
        
        return df.with_columns(exprs).select(keep_cols)

    # ==========================================
    # 4. Windowing & Loader Creation
    # ==========================================
    def create_windows(self, df, features, seq_len=None):
        """
        Unified window creation for both Train and Test.
        mode: 'train' (returns X, y) or 'test' (returns X)
        """
        self.logger.info(f"Generating sliding windows (Mode: {self.mode})...")
        
        if seq_len is None:
            seq_len = self.seq_len
        # Ensure data is sorted
        df = df.sort(["Stock ID", "datetime"])
        
        all_X, all_y = [], []
        
        # Iterate by Stock ID to prevent windows crossing different stocks
        for _, group in df.group_by("Stock ID", maintain_order=True):
            data = group.select(features).to_numpy()
            
            if len(data) <= seq_len:
                continue

            windows = sliding_window_view(data, window_shape=seq_len, axis=0)
            windows = np.moveaxis(windows, 2, 1)

            if self.mode == "train":
                try:
                    targets = group.select("target").to_numpy().flatten()
                except Exception as e:
                    self.logger.error(f"Failed to extract targets: {e}")
                    raise ValueError(f"Failed to extract targets: {e}")
                aligned_targets = targets[seq_len-1:] 
                
                min_len = min(len(windows), len(aligned_targets))
                all_X.append(windows[:min_len])
                all_y.append(aligned_targets[:min_len])
            else:
                all_X.append(windows)

        if not all_X:
            raise ValueError(f"No valid windows generated! Seq_len ({seq_len}) > Data Length?")

        X_out = np.concatenate(all_X, axis=0)
        
        if self.mode == "train":
            y_out = np.concatenate(all_y, axis=0)
            return X_out, y_out
        
        return X_out

    def save_data(self, run_id):

        base_path = "artifacts/data"

        if self.mode == "test":

            test_data_path = os.path.join(base_path, "test")
            os.makedirs(test_data_path, exist_ok=True)

            test_data_filename = f"test_{run_id}.parquet"
            test_data_full_path = os.path.join(test_data_path, test_data_filename)
            self.df.write_parquet(test_data_full_path)
            self.logger.info(f"Test Data Saved to {test_data_full_path}")

            config = {
                "data_shape": self.df.shape,
            }

            mlflow.log_params(config)

            metadata = {
                "run_id": run_id,
                "raw_data_path": self.full_path,
                "stocks": self.stocks,
                "test_data_path": test_data_full_path
            }

            mlflow.log_dict(metadata, "test_metadata.json")

            return config

        # Define file names
        val_data_name = f"val_{run_id}.parquet"
        train_data_name = f"train_{run_id}.parquet"
        scaler_file_name = f"scaler_{run_id}.parquet"

        # Define paths
        paths = {
            "train": os.path.join(base_path, "train"),
            "val": os.path.join(base_path, "validation"),
            "scaler": os.path.join(base_path, "scalers"),
        }

        # Create directories
        for p in paths.values():
            os.makedirs(p, exist_ok=True)

        # Save data
        train_data_full_path = os.path.join(paths["train"], train_data_name)
        self.train.write_parquet(train_data_full_path)
        self.logger.info(f"Train Data Saved to {train_data_full_path}")

        val_data_full_path = os.path.join(paths["val"], val_data_name)
        self.val.write_parquet(val_data_full_path)
        self.logger.info(f"Validation Data Saved to {val_data_full_path}")

        if self.scaler is not None:
            scaler_full_path = os.path.join(paths["scaler"], scaler_file_name)
            self.scaler.write_parquet(scaler_full_path)
            self.logger.info(f"Scaler Saved to {scaler_full_path}")
        else:
            scaler_full_path = None

        config = {
            "train_shape": self.train.shape,
            "val_shape": self.val.shape,
            "seq_len": self.seq_len,
            "n_stocks": len(self.stocks),
            "train_ratio": self.train_ratio,
            "time_horizon": self.time_horizon,
            "profit_pct": self.profit_pct,
            "stop_pct": self.stop_pct,
            "raw_data_path": self.full_path,
            "train_pos_count": self.train_pos_count,
            "train_neg_count": self.train_neg_count,
            "val_pos_count": self.val_pos_count,
            "val_neg_count": self.val_neg_count,
        }

        mlflow.log_params(config)

        # Save metadata

        metadata = {
            "run_id": run_id,
            "feature_cols": self.feature_cols,
            "stocks": self.stocks,
            "train_data_path": train_data_full_path,
            "val_data_path": val_data_full_path,
            "scaler_path": scaler_full_path,
        }

        mlflow.log_dict(metadata, "train_metadata.json")

        return config
    
    def load_config(self, train_data_run_id):
        train_data_run = mlflow.get_run(train_data_run_id)
        config = train_data_run.data.params

        self.seq_len = int(config["seq_len"])

        artifact_path = "train_metadata.json"
        artifact_uri = f"runs:/{train_data_run_id}/{artifact_path}"
        metadata = mlflow.artifacts.load_dict(artifact_uri)

        try:
            self.scaler = pl.read_parquet(metadata["scaler_path"])
        except FileNotFoundError:
            self.logger.error(f"Scaler file not present at {metadata['scaler_path']}")
            raise

        self.feature_cols = metadata["feature_cols"]
        self.stocks = metadata["stocks"]

        self.time_horizon = int(config["time_horizon"])
        self.profit_pct = float(config["profit_pct"])
        self.stop_pct = float(config["stop_pct"])
        self.train_shape = config["train_shape"]
        self.train_ratio = float(config["train_ratio"])
        self.val_shape = config["val_shape"]
        self.train_pos_count = config["train_pos_count"]
        self.train_neg_count = config["train_neg_count"]
        self.val_pos_count = config["val_pos_count"]
        self.val_neg_count = config["val_neg_count"]
        self.full_path = config["raw_data_path"]

        if self.mode == "train":

            try:
                self.train = pl.read_parquet(metadata["train_data_path"])
            except FileNotFoundError:
                self.logger.error(f"Train Data file not present at {metadata['train_data_path']}")
                raise
            
            try:
                self.val = pl.read_parquet(metadata["val_data_path"])
            except FileNotFoundError:
                self.logger.error(f"Validation Data file not present at {metadata['val_data_path']}")
                raise

        return config

    def reshape_for_ml(self, X: np.ndarray) -> np.ndarray:
        """
        Reshapes 3D windows into 2D for ML models (XGBoost/RF).
        Input:  (Batch_Size, Seq_Len, Features)
        Output: (Batch_Size, Seq_Len * Features)
        """
        # 1. Type Safety
        if not isinstance(X, np.ndarray):
            self.logger.error(f"Expected numpy array, got {type(X)}")
            raise TypeError(f"Expected numpy array, got {type(X)}")

        # 2. Dimension Check
        if X.ndim != 3:
            self.logger.error(f"Expected 3D input (Batch, Time, Feat), got shape {X.shape}")
            raise ValueError(f"Expected 3D input (Batch, Time, Feat), got shape {X.shape}")

        # 3. Reshape
        n_samples, seq_len, n_feats = X.shape
        return X.reshape((n_samples, seq_len * n_feats))

    def to_loaders(self, X_train, y_train, X_val, y_val, batch_size=32, num_workers=0):
        self.logger.info("Converting to PyTorch DataLoaders...")
        
        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )

    def all_features(self):
        features = [
            'rel_open', 'rel_high', 'rel_low', 'rel_close', 'rel_vol', 
            'macd', 'atr', 'stoch', 
            'roc_5', 'roc_10', 
            'rsi', 'bb_width', 'dist_sma20'
        ]
        return features

    def get_all_stocks(self, data_filename: str):
        """
        Returns a list of all stock IDs in the dataset.
        """
        try:
            q = pl.scan_parquet(os.path.join("data", data_filename))
        except FileNotFoundError:
            self.logger.error(f"❌ File not found: {self.full_path}")
            raise

        return q.select("Stock ID").unique().collect().to_series().to_list()

    def get_all_dates(self, data_filename: str):
        """
        Returns a list of all dates in the dataset.
        """
        try:
            q = pl.scan_parquet(os.path.join("data", data_filename))
        except FileNotFoundError:
            self.logger.error(f"❌ File not found: {self.full_path}")
            raise

        return q.select("datetime").unique().collect().to_series().to_list()

    def exists_config(self, config, author, purpose):

        full_raw_file_path = os.path.join("data", config['raw_file_name'])

        query = f"tags.author = '{author}' AND tags.purpose = '{purpose}' AND \
            params.seq_len = '{config['seq_len']}' AND params.time_horizon = '{config['time_horizon']}' AND \
                params.profit_pct = '{config['profit_pct']}' AND params.stop_pct = '{config['stop_pct']}' AND \
                    params.raw_data_path = '{full_raw_file_path}' AND params.train_ratio = '{config['train_ratio']}' AND \
                        params.n_stocks = '{config['n_stocks']}'"

        runs = mlflow.search_runs(filter_string=query, output_format="list")

        if len(runs) > 0:
            return runs[0].info.run_id
        else:
            return None

    def has_nan(self):
        if self.df.height == 0 or self.df.width == 0:
            return False
        return self.df.select(pl.all().null_count().sum()).sum_horizontal().item() > 0        