import streamlit as st
import time  # Added for simulation delays
from src import (
    DataProcessor, DeepLearningTrainer, Backtester, train_ml_model, 
    build_model, save_json, setup_logging, get_files_in_dir, is_parquet_file
)
from datetime import datetime
import polars as pl
import os
import torch
import torch.nn as nn
import numpy as np
import dagshub
import mlflow
import sys

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AlphaPredict",
    layout="wide"
)

setup_logging()

# Initialize DagsHub & MLflow
try:
    dagshub.init(repo_owner='pythonmailer156', repo_name='stock_direction_predictor_model_trainer', mlflow=True)
    mlflow.set_experiment("Stock_Direction_Predictor")
except Exception as e:
    st.warning("Could not initialize DagsHub/MLflow. Continuing in offline mode.")

author = "pythonmailer156"

# ======================================================
# SESSION STATE
# ======================================================
if "page" not in st.session_state:
    st.session_state.page = "Train New Model"

if "config" not in st.session_state:
    st.session_state.config = {}

# ======================================================
# GLOBAL CSS
# ======================================================
st.markdown("""
<style>
/* ---------- REMOVE EXCESS TOP PADDING ---------- */
.block-container {
    padding-top: 1rem !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem !important;
}

/* ---------- Sidebar Brand ---------- */
.sidebar-brand {
    font-size: 2.2rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 16px;
}

/* ---------- Sidebar Nav Buttons ---------- */
.nav-btn button {
    width: 100%;
    border-radius: 14px;
    background-color: black;
    color: white;
    border: 1px solid white;
    padding: 0.55em 0.8em;
    font-weight: 500;
    font-size: 0.9rem;
    margin-bottom: 8px;
}

/* ---------- Active Page ---------- */
.nav-btn.active button {
    background-color: white !important;
    color: black !important;
    border: 1px solid black !important;
}

/* ---------- Hover (inactive only) ---------- */
.nav-btn:not(.active) button:hover {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
pages = [
    "Train New Model",
    "Test Model",
    "Run Simulation"
]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">AlphaPredict</div>', unsafe_allow_html=True)

    for p in pages:
        is_active = st.session_state.page == p

        st.markdown(
            f'<div class="nav-btn {"active" if is_active else ""}">',
            unsafe_allow_html=True
        )

        if st.button(p, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()  # Force rerun to update page immediately

        st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# HELPERS
# ======================================================
def phone_layout():
    return st.columns([1, 2, 1])

def save_config(section, data, msg=None):
    st.session_state.config[section] = data
    if msg:
        st.success(msg)

# ======================================================
# PAGE 1: TRAIN DATA PREPARATION
# ======================================================
if st.session_state.page == "Train New Model":

    data_files = get_files_in_dir("data")
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # --------------------------------------------------------
    # FIX: Recover DataProcessor from Session State
    # --------------------------------------------------------
    if "train_data_dp" in st.session_state.config:
        dp = st.session_state.config["train_data_dp"]
    else:
        dp = DataProcessor()

    # --- LAYOUT: 75% MAIN (Left), 25% INFO (Right) ---
    main_col, info_col = st.columns([3, 1])

    # --- RIGHT SIDEBAR (INFO PANEL) ---
    with info_col:
        st.subheader("Data Monitor")
        data_container = st.container()
        
        st.subheader("Model Monitor")
        model_container = st.container()

        st.subheader("Training Monitor")
        metrics_container = st.container()
        
        # Initialize Placeholders
        with data_container:
            data_status = st.empty()
            data_details = st.container() 
            
            if not st.session_state.config.get("data_loaded", False):
                data_status.info("⚪ Waiting for data...")
        
        with model_container:
            st.markdown("**Status**")
            model_text = st.empty()
            progress_bar = st.empty()
            model_text.info("⚪ Waiting for configuration...")

        with metrics_container:
            st.markdown("**Live Metrics**")
            metrics_text = st.empty()
            loss_chart = st.empty()
            acc_chart = st.empty()
            metrics_text.info("⚪ Waiting for training...")

        # --------------------------------------------------------
        # FIX: Repopulate Sidebar if data is already loaded
        # --------------------------------------------------------
        if st.session_state.config.get("data_loaded", False):
            data_status.success("✅ Data Active")
            with data_details:
                # Use getattr to safely access attributes in case dp is partially initialized
                st.write(f"**Train Shape:** {getattr(dp, 'train', pl.DataFrame()).shape}")
                st.write(f"**Val Shape:** {getattr(dp, 'val', pl.DataFrame()).shape}")
                st.write(f"**Stocks:** {dp.n_stocks if hasattr(dp, 'n_stocks') else 0}")
                st.write(f"**Features:** {len(getattr(dp, 'feature_cols', []))}")
                
                # Check for NaN safely
                has_nan = dp.has_nan() if hasattr(dp, 'has_nan') else "Unknown"
                st.write(f"**NaN Values:** {has_nan}")

    # --- MAIN CONTENT (LEFT COLUMN) ---
    with main_col:
        st.title("📂 Train Data Preparation")

        load_option = st.selectbox(
            "How do you want to load data?:", 
            ["Load using MLFlow run id", "Load from Local File"], 
            index=None, 
            placeholder="Choose an option..."
        )

        # --- OPTION 1: LOAD FROM MLFLOW ---
        if load_option == "Load using MLFlow run id":
            with st.form("train_data_load_form"):
                ml_flow_run_id = st.text_input("Run ID (Load train data from MLFlow)")
                submit = st.form_submit_button("Load Data")

                if submit:
                    try:
                        data_status.info("Loading data from MLFlow...")
                        run = mlflow.get_run(ml_flow_run_id)
                        
                        if run.data.tags.get("author") != author:
                            st.error("You are not the author of this run.")
                        elif run.data.tags.get("purpose") != "train_data_prep":
                            st.error("This run is not for training data preparation.")
                        else:
                            train_data_config = dp.load_config(ml_flow_run_id)
                            
                            # Save to session state
                            save_config("train_data_dp", dp, f"Data loaded successfully! Run ID: {ml_flow_run_id}")
                            st.session_state.config["data_loaded"] = True
                            st.rerun() # Force refresh to update sidebar

                    except Exception as e:
                        st.error(f"Error loading run: {e}")

        # --- OPTION 2: LOAD FROM LOCAL FILE ---
        elif load_option == "Load from Local File":
            choice = st.selectbox("Pick data file:", data_files, index=None, placeholder="Choose a file...")

            if choice and is_parquet_file(f"data/{choice}"):
                # Cache heavy operations if possible in production
                features = dp.all_features()
                stocks = dp.get_all_stocks(choice)
                
                with st.form("train_data_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        train_ratio = st.number_input("Train Ratio", 0.1, 1.0, 0.8)
                        profit_pct = st.number_input("Profit Target (%)", 0.1, 100.0, 3.0)
                        stop_pct = st.number_input("Stop Loss (%)", 0.1, 100.0, 1.0)
                    with c2:
                        seq_len = st.number_input("Lookback Period", 5, 500, 30)
                        time_horizon = st.number_input("Time Horizon (Days)", 1, 500, 5)
                        n_stocks = st.number_input("Number of Stocks (n)", 1, len(stocks), 5)
                    
                    stock_names = st.multiselect(
                        "Select Stocks (Leave empty for Random)",
                        stocks,
                        default=[]
                    )

                    indicators = st.multiselect(
                        "Indicators (Leave empty for All)",
                        features + ["None"],
                        default=[]
                    )
                    submit = st.form_submit_button("Prepare Training Data")

                    if submit:
                        train_data_config = {
                            "raw_file_name": choice,
                            "train_ratio": train_ratio,
                            "profit_pct": profit_pct,
                            "stop_pct": stop_pct,
                            "seq_len": seq_len,
                            "time_horizon": time_horizon,
                            "stock_names": stock_names,
                            "n_stocks": n_stocks,
                            "indicators": indicators
                        }

                        purpose = "train_data_prep"
                        existing_run_id = None
                        
                        if stock_names == [] and indicators == []:
                            existing_run_id = dp.exists_config(train_data_config, author, purpose)

                        if existing_run_id:
                            data_status.info("Config exists! Loading from MLFlow...")
                            config = dp.load_config(existing_run_id)
                        else:
                            data_status.info("Preparing Data...")
                            with mlflow.start_run(run_name=f"Train_Data_Prep_{run_id}"):
                                mlflow.set_tag("author_id", author)
                                mlflow.set_tag("purpose", purpose)

                                data = dp.load_data(choice, n_stocks, stock_names)
                                data, features = dp.calculate_indicators(indicators)
                                data_tg = dp.create_binary_target(data, profit_pct, stop_pct, time_horizon)
                                train, val = dp.time_based_split(seq_len, train_ratio)
                                scaler = dp.fit_scaler(train, features)
                                config = dp.save_data(run_id)

                        # Save and Reload
                        save_config("train_data_dp", dp, "Data Prepared Successfully!")
                        st.session_state.config["data_loaded"] = True
                        st.rerun()

        st.divider()
        st.title("🧠 Model Building")
        
        model_type_options = ["XGBoost", "Random Forest", "LSTM", "Transformer"]
        model_type = st.selectbox("Pick model type:", model_type_options, index=None, placeholder="Choose a model type...")

        if model_type:
            params = st.session_state.config.get("model", {}).get("params", {})
            training_params = {}

            with st.form("model_form"):
                st.subheader(f"Configure {model_type}")
                
                # --- XGBOOST ---
                if model_type == "XGBoost":
                    c1, c2 = st.columns(2)
                    with c1:
                        n_estimators = st.number_input("Number of Trees", 50, 2000, 300)
                        max_depth = st.slider("Max Depth", 2, 20, 6)
                        learning_rate = st.number_input("Learning Rate", 0.00001, 1.0, 0.001, format="%.5f")
                    with c2:
                        subsample = st.slider("Subsample", 0.0, 1.0, 0.8)
                        colsample_bytree = st.slider("Colsample Bytree", 0.0, 1.0, 0.8)
                        eval_metric = st.selectbox("Eval Metric", ["auc", "rmse"])

                    random_state = st.number_input("Random State", 0, 100, 42)
                    params = {
                        "model_type": "xgboost", "n_estimators": n_estimators, "max_depth": max_depth,
                        "subsample": subsample, "colsample_bytree": colsample_bytree,
                        "eval_metric": eval_metric, "learning_rate": learning_rate, "random_state": random_state,
                    }

                # --- RANDOM FOREST ---
                elif model_type == "Random Forest":
                    n_estimators = st.number_input("Number of Trees", 50, 2000, 300)
                    max_depth = st.slider("Max Depth", 2, 20, 6)
                    random_state = st.number_input("Random State", 0, 100, 42)
                    params = {"model_type": "rf", "n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}

                # --- LSTM ---
                elif model_type == "LSTM":
                    # Smart Default: If features exist, use that length. Otherwise 13.
                    default_dim = len(dp.feature_cols) if hasattr(dp, 'feature_cols') and dp.feature_cols else 13
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        d_in = st.number_input("Input Size", 1, 100, default_dim)
                        hidden_units = st.number_input("Hidden Units", 16, 1024, 128)
                    with c2:
                        layers = st.slider("Layers", 1, 10, 2)
                        dropout = st.slider("Dropout", 0.0, 0.9, 0.2)
                    
                    params = {"model_type": "lstm", "d_in": d_in, "hidden_units": hidden_units, "layers": layers, "dropout": dropout}

                # --- TRANSFORMER ---
                elif model_type == "Transformer":
                    default_dim = len(dp.feature_cols) if hasattr(dp, 'feature_cols') and dp.feature_cols else 13
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        d_in = st.slider("Input Size", 1, 100, default_dim)
                        hidden_units = st.number_input("Hidden Units", 16, 1024, 128)
                        nhead = st.slider("Number of Heads", 1, 10, 2)
                        ratio_val = st.selectbox("Ratio Value (Imbalance handling)", ["True", "False"], index=0)
                    with c2:
                        n_layers = st.slider("Number of Layers", 1, 10, 2)
                        dropout = st.slider("Dropout", 0.0, 0.9, 0.2)
                        max_len = st.number_input("Max Length", 1, 1000, 100)
                    
                    params = {
                        "model_type": "transformer", "d_in": d_in, "hidden_units": hidden_units,
                        "nhead": nhead, "n_layers": n_layers, "dropout": dropout,
                        "max_len": max_len, "ratio_val": ratio_val
                    }

                # --- HYPERPARAMETERS (DL ONLY) ---
                if model_type in ["LSTM", "Transformer"]:
                    st.divider()
                    st.markdown("#### Training Hyperparameters")
                    c1, c2 = st.columns(2)
                    with c1:
                        lr = st.number_input("Learning Rate", 0.00001, 1.0, 0.001, format="%.5f")
                        epochs = st.number_input("Epochs", 1, 5000, 50)
                    with c2:
                        patience = st.slider("Early Stopping Patience", 1, 100, 10)
                        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
                        pos_weight = st.selectbox("Pos Weight (Imbalance handling)", ["True", "False"], index=0)

                    training_params = {
                        "lr": lr, "epochs": epochs, "patience": patience,
                        "threshold": threshold, "pos_weight": pos_weight
                    }

                submit_train = st.form_submit_button("Start Training")

                if submit_train:
                    # 1. Update Config State
                    st.session_state.config["model"] = {"model_type": model_type, "params": params}
                    st.session_state.config["training"] = training_params
                    
                    # 2. Update Right Sidebar (Status)
                    model_text.write("🟢 Training Started...")
                    progress_bar.progress(0)
                    
                    # 3. Simulate Training Loop (Replace this with real training)
                    try:
                        total_epochs = training_params.get("epochs", 100)
                        
                        for i in range(1, 101):
                            # Update Progress
                            progress_bar.progress(i)
                            
                            # Update Metrics in Right Sidebar
                            fake_loss = 1.0 / (i + 1)
                            fake_acc = 0.5 + (i * 0.004)
                            
                            loss_chart.metric("Loss", f"{fake_loss:.4f}")
                            acc_chart.metric("Accuracy", f"{fake_acc:.2%}", f"+{(i*0.01):.2f}%")
                            
                            time.sleep(0.02) # Remove in production
                        
                        # Call your actual training function here:
                        # train_ml_model(st.session_state.config)
                        
                        model_text.write("✅ Training Complete!")
                        st.balloons()
                        
                    except Exception as e:
                        model_text.write("🔴 Error")
                        st.error(f"Training Failed: {str(e)}")

# ======================================================
# PAGE 2: TEST MODEL
# ======================================================
elif st.session_state.page == "Test Model":
    _, center, _ = phone_layout()
    with center:
        st.title("🧪 Test Model")
        cfg = st.session_state.config.get("test_data", {})

        with st.form("test_data_form"):
            test_path = st.text_input("Test Data Path", cfg.get("path", "./test_data.csv"))
            submit = st.form_submit_button("Prepare Test Data")

        if submit:
            save_config("test_data", {"path": test_path}, "Test Data Configured!")

# ======================================================
# PAGE 3: RUN SIMULATION
# ======================================================
elif st.session_state.page == "Run Simulation":
    _, center, _ = phone_layout()
    with center:
        st.title("🚀 Run Simulation")
        cfg = st.session_state.config.get("simulation", {})

        with st.form("simulation_form"):
            profit_pct = st.number_input("Profit Percentage (%)", 0.1, 100.0, cfg.get("profit_pct", 2.0))
            stop_pct = st.number_input("Stop Percentage (%)", 0.1, 100.0, cfg.get("stop_pct", 1.0))
            time_horizon = st.number_input("Time Horizon", 1, 500, cfg.get("time_horizon", 10))
            brokerage = st.number_input("Brokerage (₹)", 0.0, 1000.0, cfg.get("brokerage", 20.0))
            investment = st.number_input("Investment per Stock (₹)", 100.0, 10_000_000.0, cfg.get("investment", 10_000.0))
            submit = st.form_submit_button("Run Simulation")

        if submit:
            save_config("simulation", {
                "profit_pct": profit_pct, "stop_pct": stop_pct,
                "time_horizon": time_horizon, "brokerage": brokerage,
                "investment": investment
            }, "Simulation Config Saved!")
            
            st.subheader("📌 Current Configuration")
            st.json(st.session_state.config)