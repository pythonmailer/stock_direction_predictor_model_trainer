import streamlit as st
import plotly.graph_objects as go
import time
import uuid
from src import (
    DataProcessor, DeepLearningTrainer, Backtester, train_ml_model, 
    build_model, save_json, setup_logging, get_files_in_dir, is_parquet_file,
    set_global_seed, download_from_s3, upload_to_s3, list_s3_files
)
from datetime import datetime
import copy
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
seed = 42
set_global_seed(seed)

def create_threshold_chart(stats_df):
    """Generates the Plotly figure for a given stats dataframe"""
    plot_data = stats_df.to_pandas()
    
    fig = go.Figure()

    # Bar Chart (Wins)
    fig.add_trace(go.Bar(
        x=plot_data['Threshold'],
        y=plot_data['Win Count'],
        name='Number of Wins',
        marker_color='rgba(135, 206, 250, 0.6)'
    ))

    # Line Chart (Win Rate)
    fig.add_trace(go.Scatter(
        x=plot_data['Threshold'],
        y=plot_data['Win Rate (%)'],
        name='Win Rate %',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='firebrick', width=3)
    ))

    fig.update_layout(
        title="Trade Quantity vs. Quality",
        xaxis=dict(title="Probability Threshold"),
        yaxis=dict(title="Number of Wins", side="left"),
        yaxis2=dict(
            title="Win Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0) # Tight layout
    )
    return fig

st.set_page_config(
    page_title="AlphaPredict",
    layout="wide"
)

setup_logging()

def init_dagshub():
    if not st.session_state.get("dagshub_initialized", False):
        dagshub.init(repo_owner='pythonmailer156', repo_name='stock_direction_predictor_model_trainer', mlflow=True)
        mlflow.set_experiment("Stock_Direction_Predictor")
        st.session_state["dagshub_initialized"] = True

init_dagshub()

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

def save_config(section, data, msg=None):
    st.session_state.config[section] = data
    if msg:
        st.success(msg)

# ======================================================
# PAGE 1: TRAIN DATA PREPARATION (MODIFIED WITH RIGHT SIDEBAR)
# ======================================================

if "train_data_dp" in st.session_state.config:
    dp = st.session_state.config["train_data_dp"]
else:
    dp = DataProcessor()

main_col, info_col = st.columns([3, 1])

# --- RIGHT SIDEBAR (INFO PANEL) ---
with info_col:
    st.subheader("")
    st.subheader("Data Monitor")
    data_container = st.container()
    
    st.subheader("Model Monitor")
    model_container = st.container()

    st.subheader("Training Monitor")
    metrics_container = st.container()

    st.subheader("Test Data Monitor")
    test_data_container = st.container()

    st.subheader("Simulation Monitor")
    sim_container = st.container()
    
    # Create Placeholders
    with data_container:
        data_status = st.empty()
        data_details = st.container() 
            
        if not st.session_state.config.get("data_loaded", False):
            data_status.info("⚪ Waiting for data...")
        
    with model_container:
        model_text = st.empty()
        model_details = st.container()
        training_hp = st.container()
        if not st.session_state.config.get("model"):
            model_text.info("⚪ Waiting for configuration...")

    with metrics_container:
        st.markdown("**Live Metrics**")
        progress_bar = st.empty()
        metrics_text = st.empty()
        metrics_chart = st.container()
        metrics_text.info("⚪ Waiting for training...")

    with test_data_container:
        test_data_status = st.empty()
        test_data_details = st.container() 
            
        if not st.session_state.get("backtest", {}).get("data", False):
            test_data_status.info("⚪ Waiting for valid test data...")

    with sim_container:
        sim_status = st.empty()
        sim_details = st.container() 
            
        if not st.session_state.config.get("sim_data_loaded", False):
            sim_status.info("⚪ Waiting for data...")

    if st.session_state.config.get("data_loaded", False):
        data_status.success("✅ Data Active")
        with data_details:
            st.write(f"**Train Shape:** {dp.train_shape}")
            st.write(f"**Val Shape:** {dp.val_shape}")
            st.write(f"**Train Ratio:** {dp.train_ratio}")
            st.write(f"**Stocks:** {len(dp.stocks)}")
            st.write(f"**Features:** {len(dp.feature_cols)}")
            st.write(f"**Time Horizon:** {dp.time_horizon}")
            st.write(f"**Sequence Length:** {dp.seq_len}")
            st.write(f"**Profit Pct:** {dp.profit_pct}")
            st.write(f"**Stop Pct:** {dp.stop_pct}")
            st.write(f"**Raw Path:** {dp.full_path}")
            st.write(f"**Train Pos Count:** {dp.train_pos_count}")
            st.write(f"**Train Neg Count:** {dp.train_neg_count}")
            st.write(f"**Val Pos Count:** {dp.val_pos_count}")
            st.write(f"**Val Neg Count:** {dp.val_neg_count}")
            
        if dp.train_pos_count == 0 or dp.train_neg_count == 0:
            st.toast("Train Data has only one class!", icon="❌")
            st.session_state.valid_data = False
            
        if dp.val_pos_count == 0 or dp.val_neg_count == 0:
            st.toast("Validation Data has only one class!", icon="❌")
            st.session_state.valid_data = False
        
    if st.session_state.config.get("model"):

        model = st.session_state.config["model"]
        model_text.write(model["model_type"])
            
        for key, value in model["params"].items():
            model_details.write(f"**{key}**: {value}")

        if model.get("training"):
            training_params = model["training"]
            model_details.write("### Training Hyperparameters")

            for key, value in training_params.items():
                model_details.write(f"**{key}**: {value}")

        if st.session_state.config.get("metrics"):
            metrics = st.session_state.config["metrics"]
            metrics_text.write("✅ Training Complete")
            progress_bar.progress(1.0)

            if model.get("model_type", "").lower() in ["lstm", "transformer"]:
                with metrics_chart:
                    tab1, tab2 = st.tabs(["📉 Loss & Acc", "📊 Advanced Metrics"])
                                            
                    with tab1:
                        c1, c2 = st.columns(2)
                        train_loss_chart = c1.empty()
                        val_loss_chart = c2.empty()
                        train_acc_chart = c1.empty()
                        val_acc_chart = c2.empty()
                                            
                    with tab2:
                        c1, c2 = st.columns(2)
                        train_f1_chart = c1.empty()
                        val_f1_chart = c2.empty()
                        train_auc_chart = c1.empty()
                        val_auc_chart = c2.empty()
                        train_prec_chart = c1.empty()
                        val_prec_chart = c2.empty()
                        train_rec_chart = c1.empty()
                        val_rec_chart = c2.empty()

                train_loss_chart.metric("Train Loss", f"{metrics['train_loss']:.4f}")
                train_acc_chart.metric("Train Accuracy", f"{metrics['train_acc']:.4f}")
                train_f1_chart.metric("Train F1", f"{metrics['train_f1']:.4f}")
                train_prec_chart.metric("Train Precision", f"{metrics['train_prec']:.4f}")
                train_rec_chart.metric("Train Recall", f"{metrics['train_rec']:.4f}")
                train_auc_chart.metric("Train AUC", f"{metrics['train_auc']:.4f}")
                val_loss_chart.metric("Val Loss", f"{metrics['val_loss']:.4f}")
                val_acc_chart.metric("Val Accuracy", f"{metrics['val_acc']:.4f}")
                val_f1_chart.metric("Val F1", f"{metrics['val_f1']:.4f}")
                val_prec_chart.metric("Val Precision", f"{metrics['val_prec']:.4f}")
                val_rec_chart.metric("Val Recall", f"{metrics['val_rec']:.4f}")
                val_auc_chart.metric("Val AUC", f"{metrics['val_auc']:.4f}")

            else:
                with metrics_chart:
                    acc_chart = st.empty()
                    f1_chart = st.empty()
                    prec_chart = st.empty()
                    rec_chart = st.empty()
                    auc_chart = st.empty()
                
                acc_chart.metric("Accuracy", f"{metrics.get('acc', 0):.2%}")
                f1_chart.metric("F1", f"{metrics.get('f1', 0):.2%}")
                prec_chart.metric("Precision", f"{metrics.get('prec', 0):.2%}")
                rec_chart.metric("Recall", f"{metrics.get('rec', 0):.2%}")
                auc_chart.metric("AUC", f"{metrics.get('auc', 0):.2%}")

    if st.session_state.get("backtester", False) and st.session_state.get("backtest").data:
        bt = st.session_state.get("backtest")
        if st.session_state.get("test_data_valid"):
            test_data_status.info("🟢 Valid Test Data")
        else:
            test_data_status.info("🔴 Invalid Test Data")

        test_data_details.write("### Test Data Details")
        test_data_details.write(f"**Test Data Shape: {bt.data.shape}**")
    

if st.session_state.page == "Train New Model":

    if "run_id" not in st.session_state:
        st.session_state.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    st.session_state.valid_data = st.session_state.get("valid_data", False)

    with main_col:
        st.title("📂 Train Data Preparation")
        st.session_state.new_data = st.session_state.get("new_data", False)
        st.session_state.save_data = st.session_state.get("save_data", False)

        load_option = st.selectbox(
            "How do you want to load data?:", 
            ["Load using MLFlow run id", "Load from S3"], 
            index=None
        )

        if load_option == "Load using MLFlow run id":
            purpose = "train_data_prep"
            with st.form("mlflow_load"):
                ml_flow_run_id = st.text_input("Run ID")
                submit = st.form_submit_button("Load Data")

                if submit:
                    try:
                        run = mlflow.get_run(ml_flow_run_id)
                        if run and run.data.tags.get("author") != author:
                            data_status.info("You are not the author of this run.")
                        elif run and run.data.tags.get("purpose") != purpose:
                            data_status.info("This run is not for train data preparation!")
                        else:
                            data_status.info("Loading data from MLFlow...")
                            train_data_config = dp.load_config(ml_flow_run_id)
                            save_config("train_data_mlflow_run_id", ml_flow_run_id)
                            save_config("train_data_dp", dp)
                            st.session_state.config["data_loaded"] = True
                            st.toast('Data Loaded Successfully!', icon='✅')
                            st.session_state.valid_data = True
                            st.rerun() # Force refresh to update sidebar
                    except Exception as e:
                        st.error(f"Error: {e}")

        elif load_option == "Load from S3":
            purpose = "train_data_prep"
            data_files = list_s3_files("data")
            choice = st.selectbox("Pick data file:", data_files, index=None)
            if choice:
                download_from_s3("data/" + choice, "data/" + choice)

            if choice and not is_parquet_file("data/" + choice):
                st.toast("Selected file is not a parquet file!", icon="❌")
            
            elif choice:
                stocks = dp.get_all_stocks(choice)
                features = dp.all_features()

                with st.form("local_load"):
                    c1, c2 = st.columns(2)
                    with c1:
                        train_ratio = st.number_input("Train Ratio", 0.1, 1.0, 0.8)
                        profit_pct = (st.number_input("Profit Target (%)", 0.1, 100.0, 3.0))/100
                        stop_pct = (st.number_input("Stop Loss (%)", 0.1, 100.0, 1.0))/100
                    with c2:
                        seq_len = st.number_input("Lookback Period (How far behind the model can see to predict future direction.)", 5, 60, 30)
                        time_horizon = st.number_input("Time Horizon (How long are you willing to hold the stock, if you don't make profit or loss in this period, you will exit the trade.)", 1, 60, 5)
                        n_stocks = st.number_input("Num Stocks (Max 5 for Testing)", 1, 5, 5)
                    
                    stock_names = st.multiselect("Select Stocks (If no stock is selected, stocks will be selected randomly.)", stocks)
                    indicators = st.multiselect("Indicators", features)
                    
                    submit = st.form_submit_button("Prepare Data")

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

                        existing_run_id = None
                        
                        if stock_names == [] and indicators == []:
                            existing_run_id = dp.exists_config(train_data_config, author, purpose)

                        if existing_run_id:
                            data_status.info("Config exists! Loading from MLFlow...")
                            save_config("train_data_mlflow_run_id", existing_run_id)
                            config = dp.load_config(existing_run_id)
                            st.session_state.valid_data = True

                        else:
                            data_status.info("Loading data from file...")             
                            data = dp.load_data(choice, n_stocks, stock_names)
                            data, final_feats = dp.calculate_indicators(indicators)
                            dp.create_binary_target(data, profit_pct, stop_pct, time_horizon)
                            train, val = dp.time_based_split(seq_len, train_ratio)
                            dp.fit_scaler(train, final_feats)
                            st.session_state.new_data = True
                            st.session_state.valid_data = True       
                            
                        save_config("train_data_dp", dp)
                        st.session_state.config["data_loaded"] = True
                        st.rerun()

            if st.session_state.new_data and st.session_state.valid_data and not st.session_state.save_data and st.button("Save Data"):
                with mlflow.start_run(run_name=f"Train_Data_Prep_{st.session_state.get('run_id')}") as run:   
                    mlflow.set_tag("author", author)
                    mlflow.set_tag("purpose", purpose)
                    dp.save_data(st.session_state.get("run_id"))
                    save_config("train_data_mlflow_run_id", run.info.run_id)
                    mlflow.end_run()
                st.toast("Data saved to MLFlow successfully!", icon="✅")
                st.session_state.new_data = False
                st.session_state.save_data = True
                st.rerun() 
        
        if st.session_state.valid_data and st.session_state.save_data:

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
                            early_stopping_rounds = st.number_input("Early Stopping Rounds", 1, 100, 10)
                        
                        params = {
                            "model_type": "xgboost", "n_estimators": n_estimators, "max_depth": max_depth,
                            "subsample": subsample, "colsample_bytree": colsample_bytree,
                            "learning_rate": learning_rate, "random_state": seed,
                            "early_stopping_rounds": early_stopping_rounds
                        }

                    # --- RANDOM FOREST ---
                    elif model_type == "Random Forest":
                        n_estimators = st.number_input("Number of Trees", 50, 300, 100)
                        max_depth = st.slider("Max Depth", 2, 10, 6)
                        params = {"model_type": "rf", "n_estimators": n_estimators, "max_depth": max_depth, "random_state": seed}

                    # --- LSTM ---
                    elif model_type == "LSTM":
                        c1, c2 = st.columns(2)
                        with c1:
                            d_in = st.number_input("Input Size", 1, 100, 13)
                            hidden_units = st.number_input("Hidden Units", 16, 1024, 128)
                        with c2:
                            layers = st.slider("Layers", 1, 10, 2)
                            dropout = st.slider("Dropout", 0.0, 0.9, 0.2)
        
                        params = {"model_type": "lstm", "d_in": d_in, "hidden_units": hidden_units, "layers": layers, "dropout": dropout}

                    # --- TRANSFORMER ---
                    elif model_type == "Transformer":
                        c1, c2 = st.columns(2)
                        with c1:
                            d_in = st.slider("Input Size", 1, 100, 13)
                            hidden_units = st.number_input("Hidden Units", 16, 1024, 128)
                            nhead = st.slider("Number of Heads", 1, 10, 2)
                            ratio_val = st.selectbox("Ratio Value (Imbalance handling)", ["True", "False"], index=0, placeholder="True or False")
                        with c2:
                            n_layers = st.slider("Number of Layers", 1, 10, 2)
                            dropout = st.slider("Dropout", 0.0, 0.9, 0.2)
                            max_len = st.number_input("Max Length", 1, 1000, 100)
                        
                        params = {
                            "model_type": "transformer", 
                            "d_in": d_in, 
                            "hidden_units": hidden_units, 
                            "nhead": nhead, 
                            "n_layers": n_layers, 
                            "dropout": dropout, 
                            "max_len": max_len,
                            "ratio_val": ratio_val
                        }

                    # --- HYPERPARAMETERS (DL ONLY) ---
                    if model_type in ["LSTM", "Transformer"]:
                        st.divider()
                        st.markdown("#### Training Hyperparameters")
                        c1, c2 = st.columns(2)
                        with c1:
                            lr = st.number_input("Learning Rate", 0.00001, 1.0, 0.001, format="%.5f")
                            epochs = st.number_input("Epochs (Max 10 for Testing)", 1, 10, 5)
                            pos_weight = st.selectbox("Pos Weight (Imbalance handling)", ["True", "False"], index=0)
                        with c2:
                            patience = st.slider("Early Stopping Patience", 1, 9, 2)
                            threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

                        training_params = {
                            "lr": lr, "epochs": epochs, "patience": patience,
                            "threshold": threshold, "pos_weight": pos_weight
                        }

                    submit_train = st.form_submit_button("Start Training")

                    if submit_train:
                        st.session_state.config["model"] = {"model_type": model_type, "params": params}

                        for key, value in params.items():
                            model_details.write(f"**{key}**: {value}")

                        if training_params:
                            st.session_state.config["model"]["training"] = training_params

                            model_details.write("### Training Hyperparameters")

                            for key, value in training_params.items():
                                model_details.write(f"**{key}**: {value}")

                        model_text.write(model_type)
                        
                        # 2. Update Right Sidebar (Status)
                        metrics_text.write("🟢 Training Started...")
                        progress_bar.progress(0)

                        def ui_callback(epoch, total_epochs, metrics):

                            progress = min(float(epoch) / total_epochs, 1.0)
                            progress_bar.progress(progress)
                            metrics_text.write(f"⏳ Epoch: {epoch}/{total_epochs}")
                    
                            # Update Metrics
                            train_loss_chart.metric("Train Loss", f"{metrics['train_loss']:.4f}")
                            train_acc_chart.metric("Train Accuracy", f"{metrics['train_acc']:.4f}")
                            train_f1_chart.metric("Train F1", f"{metrics['train_f1']:.4f}")
                            train_prec_chart.metric("Train Precision", f"{metrics['train_prec']:.4f}")
                            train_rec_chart.metric("Train Recall", f"{metrics['train_rec']:.4f}")
                            train_auc_chart.metric("Train AUC", f"{metrics['train_auc']:.4f}")
                            val_loss_chart.metric("Val Loss", f"{metrics['val_loss']:.4f}")
                            val_acc_chart.metric("Val Accuracy", f"{metrics['val_acc']:.4f}")
                            val_f1_chart.metric("Val F1", f"{metrics['val_f1']:.4f}")
                            val_prec_chart.metric("Val Precision", f"{metrics['val_prec']:.4f}")
                            val_rec_chart.metric("Val Recall", f"{metrics['val_rec']:.4f}")
                            val_auc_chart.metric("Val AUC", f"{metrics['val_auc']:.4f}")
                        
                        train_scaled = dp.transform_scaler(dp.train)
                        val_scaled = dp.transform_scaler(dp.val)
                            
                        train_X, train_y = dp.create_windows(train_scaled, features, seq_len)
                        val_X, val_y = dp.create_windows(val_scaled, features, seq_len)

                        with mlflow.start_run(run_name=f"Model_Training_{model_type}_{st.session_state.run_id}") as training_run:

                            mlflow.set_tag("train_data_run_id", st.session_state.config.get("train_data_mlflow_run_id")) 
                            mlflow.set_tag("author", author)
                            mlflow.set_tag("purpose", "model_training")

                            if model_type in ["LSTM", "Transformer"]:    

                                with metrics_chart:
                                    tab1, tab2 = st.tabs(["📉 Loss & Acc", "📊 Advanced Metrics"])
                                                            
                                    with tab1:
                                        c1, c2 = st.columns(2)
                                        train_loss_chart = c1.empty()
                                        val_loss_chart = c2.empty()
                                        train_acc_chart = c1.empty()
                                        val_acc_chart = c2.empty()
                                                            
                                    with tab2:
                                        c1, c2 = st.columns(2)
                                        train_f1_chart = c1.empty()
                                        val_f1_chart = c2.empty()
                                        train_auc_chart = c1.empty()
                                        val_auc_chart = c2.empty()
                                        train_prec_chart = c1.empty()
                                        val_prec_chart = c2.empty()
                                        train_rec_chart = c1.empty()
                                        val_rec_chart = c2.empty()

                                if torch.backends.mps.is_available():
                                    device = torch.device("mps")
                                elif torch.cuda.is_available():
                                    device = torch.device("cuda")
                                else:
                                    device = torch.device("cpu")

                                # Handle Class Imbalance
                                num_pos = np.sum(train_y == 1).item()
                                num_neg = np.sum(train_y == 0).item()

                                if params["model_type"] == "transformer" and params["ratio_val"] == "True":
                                    params["ratio_val"] = num_neg / (num_pos + 1e-6)
                                elif params["model_type"] == "transformer":
                                    params["ratio_val"] = None

                                temp_params = params.copy()
                                temp_params.pop("model_type")

                                model, hp = build_model(params["model_type"], feature_cols=dp.feature_cols, **temp_params)
                                mlflow.log_params(hp)

                                if training_params["pos_weight"] == "True":
                                    training_params["pos_weight"] = torch.tensor((num_neg / (num_pos + 1e-6)), dtype=torch.float32).to(device)
                                else:
                                    training_params["pos_weight"] = None

                                criterion = nn.BCEWithLogitsLoss(pos_weight=training_params["pos_weight"])
                                optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])

                                mlflow_params = {
                                    "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                    "ratio_val": params.get("ratio_val"),
                                    "pos_weight": training_params.get("pos_weight"),
                                    "criterion": criterion.__class__.__name__,
                                    "optimizer": optimizer.__class__.__name__,
                                    "learning_rate": training_params["lr"],
                                    "threshold": training_params["threshold"],
                                }
                                mlflow.log_params(mlflow_params)

                                train_loader, val_loader = dp.to_loaders(train_X, train_y, val_X, val_y)
                                input_example = np.random.rand(1, dp.seq_len, len(dp.feature_cols)).astype(np.float32)
                                trainer = DeepLearningTrainer(model, criterion, optimizer, device, epochs, patience, input_example, model_type, threshold)
                                trained_model, result = trainer.train(train_loader, val_loader, callback=ui_callback)

                                mlflow.log_metrics(result)                           
                            else:
                                
                                with st.spinner(f"Training {model_type} model... This usually takes a few seconds."):
                                    train_X = dp.reshape_for_ml(train_X)
                                    val_X = dp.reshape_for_ml(val_X)

                                    temp_params = params.copy()
                                    temp_params.pop("model_type")

                                    model, hp = build_model(params["model_type"], feature_cols=dp.feature_cols, **temp_params)
                                    mlflow.log_params(hp)

                                    trained_model, result = train_ml_model(model, train_X, train_y, val_X, val_y, params["model_type"])
                                    mlflow.log_metrics(result)
                                
                            st.session_state.training_run_id = training_run.info.run_id 
                            mlflow.end_run()

                        st.toast("Model saved in MLFlow", icon="✅")
                        st.session_state.config["metrics"] = result
                        st.session_state.config["trained_model"] = trained_model 

                        st.rerun()

# ======================================================
# PAGE 2: TEST DATA PREPARATION (UNCHANGED)
# ======================================================
elif st.session_state.page == "Test Model":

    bt = st.session_state.config.get("backtester", Backtester())
    st.session_state.analysis_history = st.session_state.get("analysis_history", [])
    if "run_id" not in st.session_state:
        st.session_state.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with main_col:
        st.title("🧪 Test Model")

        model_choice = st.session_state.get("model_choice", None)
        if st.session_state.config.get("trained_model", None) is not None:
            model_choice = st.selectbox("Do you want to test the existing model?", 
                ["Yes", "No"], index=None) 

            if model_choice == "Yes":
                bt.dp.load_for_test(st.session_state.config["train_data_dp"])
                bt.model = st.session_state.config["trained_model"]
                bt.model_type = st.session_state.config["model"]["model_type"]
        
        if model_choice == "No" or st.session_state.config.get("trained_model", None) is None:
            selected_model = None
            selected_model = st.selectbox("Select Model", bt.get_all_models(author, "model_training"), index=None)

            if selected_model:
                st.session_state.config["train_data_dp"] = bt.load_using_mlrun_id(bt.run_dict[selected_model])
                bt.dp.load_for_test(st.session_state.config["train_data_dp"])
                st.session_state.config["data_loaded"] = True
                st.session_state.config["trained_model"] = bt.model
                model = {
                    "model_type": bt.model_type,
                    "params": bt.training_params
                }
                st.session_state.config["backtester"] = bt
                st.session_state.config["model"] = model
                st.session_state.config["metrics"] = bt.training_metrics
                
                st.session_state.model_choice = "Yes"
                st.rerun()
            
        if model_choice == "Yes":
            data_files = list_s3_files("data")
            choice = st.selectbox("Pick data file:", data_files, index=None)

            if choice:
                download_from_s3("data/" + choice, "data/" + choice)

            if choice and not is_parquet_file("data/" + choice):
                st.toast("Selected file is not a parquet file!", icon="❌")
            elif choice:
                stocks = bt.get_common_stocks(choice)

                with st.form("local_load"):
                    train_ratio = st.number_input("Train Ratio", 0.0, 0.9, 0.8)
                    submit = st.form_submit_button("Prepare Data")

                    if submit:
                        bt.run_inference(choice, train_ratio)
                        st.session_state.config["backtester"] = bt

            if bt.data is not None and bt.data.height >= bt.dp.seq_len:
                st.session_state.test_data_valid = True
            elif bt.data is not None:
                st.toast("Test data is not long enough!", icon="❌")
            
        if st.session_state.get("test_data_valid", False):

            st.subheader("Threshold Optimization")
            c1, c2 = st.columns(2)

            profit_pct = (c1.number_input("Profit Percentage (%)", 0.1, 100.0, 2.0))/100
            stop_pct = (c1.number_input("Stop Percentage (%)", 0.1, 100.0, 1.0))/100
            time_horizon = c1.number_input("Time Horizon (How long are you willing to hold the stock, if you don't make profit or loss in this period, you will exit the trade.)", 1, 60, 5)
            
            min_t = c2.slider("Min Threshold", 0.01, 0.9, 0.5, 0.01)
            max_t = c2.slider("Max Threshold", 0.1, 1.0, 0.95, 0.01)
            step = c2.slider("Step", 0.01, 0.1, 0.01)

            if st.button("Analyze & Add Graph"):
                bt.update_target(bt.results_df, profit_pct, stop_pct, time_horizon)
                stats_df = bt.analyze_thresholds(bt.results_df, min_t, max_t, step)
                new_entry = {
                    "id": str(uuid.uuid4()),
                    "data": stats_df,
                    "min": min_t,
                    "profit_pct": profit_pct,
                    "stop_pct": stop_pct,
                    "time_horizon": time_horizon,
                    "max": max_t,
                    "timestamp": time.strftime("%H:%M:%S")
                }
                st.session_state.analysis_history.append(new_entry)
                st.success(f"Graph added! Total graphs: {len(st.session_state.analysis_history)}")

            st.divider()

            for index, entry in enumerate(st.session_state.analysis_history):
                with st.container():
                    col_title, col_delete = st.columns([4, 1])
                    with col_title:
                        st.markdown(f"### 📊 Analysis #{index + 1}")
                        st.caption(f"Threshold Range: {entry['min']} - {entry['max']} | Time: {entry['timestamp']} | Profit: {entry['profit_pct']} | Stop: {entry['stop_pct']} | Horizon: {entry['time_horizon']}")
                    with col_delete:
                        if st.button("🗑️ Remove", key=f"del_{entry['id']}"):
                            st.session_state.analysis_history.pop(index)
                            st.rerun()

                    fig = create_threshold_chart(entry['data'])
                    st.plotly_chart(fig, width="stretch")
                    
                    with st.expander(f"View Data for Analysis #{index + 1}"):
                        st.dataframe(entry['data'], hide_index=True)

elif st.session_state.page == "Run Simulation":

    if st.session_state.config.get("backtester") is None or st.session_state.config["backtester"].results_df is None:
        st.toast("No predictions found. Please run a test first in the Test Model page. Redirecting...", icon="❌")
        st.session_state.page = "Test Model"
        time.sleep(2)
        st.rerun()
    
    bt = st.session_state.config.get("backtester")

    with main_col:
        st.title("🚀 Run Simulation")
        cfg = st.session_state.config.get("simulation", {})

        with st.form("simulation_form"):
            c1, c2 = st.columns(2)
            profit_pct = (c1.number_input("Profit Percentage (%)", 0.1, 100.0, 2.0))/100
            stop_pct = (c1.number_input("Stop Percentage (%)", 0.1, 100.0, 1.0))/100
            time_horizon = c1.number_input("Time Horizon (How long are you willing to hold the stock, if you don't make profit or loss in this period, you will exit the trade.)", 1, 60, 5)
            brokerage = (c2.number_input("Brokerage (%)", 0.0, 3.0, 2.0))/100
            investment = c2.number_input("Investment per Stock (₹)", 100.0, 10_000_000.0, 10_000.0)
            threshold = c2.number_input("Threshold", 0.01, 1.0, 0.5)
            submit = st.form_submit_button("Run Simulation")

        if submit:
            save_config("simulation", {
                "profit_pct": profit_pct, "stop_pct": stop_pct,
                "time_horizon": time_horizon, "brokerage": brokerage,
                "investment": investment, "threshold": threshold
            }, "Simulation Config Saved!")

            results_saved = False

            with st.spinner("Running Simulation..."):
                df_pnl, results = bt.run_simulation(
                    data=bt.results_df,
                    time_horizon=time_horizon,
                    profit_pct=profit_pct,
                    brokerage=brokerage,
                    stop_pct=stop_pct,
                    investment_per_stock=investment,
                    threshold=threshold
                )
                
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", results['total_trades'])
            c2.metric("Win Rate", f"{(results['total_wins']/results['total_trades'])*100:.1f}%" if results['total_trades'] > 0 else "0%")
            c3.metric("Net PnL", f"₹ {results['total_pnl']:,.2f}", delta_color="normal")
                
            roi = (results['total_pnl'] / (investment * results['total_trades'])) * 100 if results['total_trades'] > 0 else 0
            c4.metric("Avg ROI / Trade", f"{roi:.2f}%")

            st.divider()
                
            if results['total_trades'] > 0:
                fig = bt.plot_equity_curve(df_pnl)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No trades generated with current settings. Lower the threshold or adjust profit/stop targets.")
            
            st.subheader("📌 Current Configuration")
            st.json(st.session_state.config["simulation"])

            if not results_saved and st.button("Save Results"):
                bt.save_results()
                results_saved = True
                st.success("Results Saved!")