import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os

# === CONFIGURATION ===
PAGE_TITLE = "Stock Trend Predictor AI"
DATA_PATH = "nifty500_no_adani_kite.parquet"
MODEL_PATH = "universal_market_model.keras"  # Or your specific model path
WINDOW_SIZE = 60

# === SETUP PAGE ===
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(f"📈 {PAGE_TITLE}")
st.markdown("---")

# === CACHED FUNCTIONS (For Speed) ===
@st.cache_resource
def load_model_cached():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

@st.cache_data
def load_data_cached():
    if os.path.exists(DATA_PATH):
        # Scan parquet to get list of stocks quickly without loading everything
        q = pl.scan_parquet(DATA_PATH)
        stocks = q.select("Stock ID").unique().collect().to_series().to_list()
        return sorted(stocks)
    return []

@st.cache_data
def get_stock_data(stock_id):
    # Load data for just ONE stock
    q = pl.scan_parquet(DATA_PATH).filter(pl.col("Stock ID") == stock_id)
    df = q.collect()
    
    # Simple preprocessing for display
    # (In a real app, you'd import your actual 'data_processor.py' logic here)
    if "datetime" not in df.columns:
         df = df.with_columns(
             (pl.col("Date").cast(pl.Utf8) + " " + pl.col("Time").cast(pl.Utf8))
             .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("datetime")
         )
    return df.sort("datetime")

# === SIDEBAR INPUTS ===
st.sidebar.header("Configuration")
model = load_model_cached()
available_stocks = load_data_cached()

if not available_stocks:
    st.error("Data file not found! Please check DATA_PATH.")
    st.stop()

selected_stock = st.sidebar.selectbox("Select Stock", available_stocks, index=0)
days_to_plot = st.sidebar.slider("Zoom (Last N Days)", 30, 365, 90)

if st.sidebar.button("Run Prediction", type="primary"):
    with st.spinner(f"Analyzing {selected_stock}..."):
        # 1. Get Data
        df = get_stock_data(selected_stock)
        
        if df.height < WINDOW_SIZE:
            st.error(f"Not enough data for {selected_stock} (Need {WINDOW_SIZE} candles)")
        else:
            # 2. Prep for Model (Simplified for Demo)
            # NOTE: You should ideally import your exact 'create_features' function here
            # For visualization, we just show the Price Chart
            
            # 3. Visualization
            # Convert to Pandas for Plotly
            plot_data = df.tail(days_to_plot * 375).to_pandas() # Roughly converting days to minutes if 1-min data
            # Or if daily data:
            # plot_data = df.tail(days_to_plot).to_pandas()

            # Main Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=plot_data['datetime'],
                open=plot_data['open'], high=plot_data['high'],
                low=plot_data['low'], close=plot_data['close'],
                name=selected_stock
            ))
            
            fig.update_layout(
                title=f"{selected_stock} Price Action",
                yaxis_title="Price (INR)",
                xaxis_rangeslider_visible=False,
                height=600,
                template="plotly_dark"
            )
            
            # 4. Display Layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("AI Prediction")
                
                # Dummy Prediction Logic (Replace with real model.predict)
                # In real implementation: 
                # features = create_features(df)
                # prob = model.predict(features)[-1]
                
                # Placeholder for visual demo
                dummy_prob = np.random.uniform(0.4, 0.9) 
                
                if dummy_prob > 0.6:
                    st.success("## 🚀 BUY")
                    st.metric("Confidence", f"{dummy_prob:.1%}")
                    st.markdown("Strong uptrend detected.")
                elif dummy_prob < 0.4:
                    st.error("## 🔻 SELL")
                    st.metric("Confidence", f"{(1-dummy_prob):.1%}")
                    st.markdown("Bearish momentum.")
                else:
                    st.warning("## ⏸️ HOLD")
                    st.metric("Confidence", f"{dummy_prob:.1%}")
                    st.markdown("Market is choppy.")
                    
                st.markdown("---")
                st.write("**Latest Close:**", plot_data['close'].iloc[-1])
                st.write("**Volume:**", plot_data['volume'].iloc[-1])

else:
    st.info("Select a stock and click 'Run Prediction' to start.")


