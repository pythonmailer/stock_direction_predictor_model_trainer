📈 AlphaPredict: End-to-End MLOps Stock Direction Predictor Live App / Demo: https://bit.ly/4qAJ56D

AlphaPredict is a full-stack, production-ready Machine Learning system designed to predict short-term stock price direction. Rather than predicting absolute prices (which is heavily prone to noise), this system frames trading as a binary classification problem using the Triple-Barrier Method.

It features an automated CI/CD pipeline, experiment tracking, and a scalable UI for dynamic backtesting and threshold optimization.

🎯 Business Goal & SLOs Goal: Predict whether a stock will hit a predefined profit target (e.g., +3%) before hitting a stop-loss (e.g., -1%) within a fixed time horizon (e.g., 5 days).

Target Metric: Optimize for Precision and Win Rate over pure accuracy to minimize capital drawdown.

SLOs: * Model inference latency: < 500ms per batch.

Infrastructure cost: ~$20/month (Running on single AWS EC2 t3.micro instance + AWS S3).

🏗️ System Architecture & Data Flow

graph TD; A[Raw Market Data S3] -->|Polars Data Pipeline| B(Feature Engineering & Triple Barrier); B -->|Time-based Split & Scaling| C{Model Training}; C -->|PyTorch| D[LSTM / Transformer]; C -->|Scikit-Learn| E[XGBoost / Random Forest]; D --> F[(MLflow + DagsHub)]; E --> F; F -->|Load Best Model| G[Streamlit Serving / Backtester]; G --> H((End User Dashboard)); Infrastructure & MLOps Stack Data Processing: Polars (Multi-threaded execution for indicator calculation & windowing)

Modeling: PyTorch (LSTM, Transformers), XGBoost, Scikit-Learn

Experiment Tracking: MLflow hosted on DagsHub

App Serving: Streamlit

CI/CD & Deployment: GitHub Actions -> Docker Compose -> AWS EC2

Storage: AWS S3 (Parquet files, scalers, and temporary weights)

Latency: Data transformation + Inference p95 latency is ~120ms per 100-stock batch(30 days).

⚙️ How It Works (The Pipeline)

Feature Engineering (data_processor.py) Migrated from Pandas to Polars to handle massive sequential data processing efficiently.
Dynamically calculates technical indicators (MACD, RSI, Bollinger Bands, ATR, Stochastic) and relative volume/price shifts in fully vectorized, multi-threaded operations.

Constructs 3D sliding windows (Batch, Sequence_Length, Features) for deep learning models, while retaining the ability to flatten to 2D for tree-based models.

Triple Barrier Target Creation Implements dynamic target creation. A sample is labeled 1 (Buy) only if the upper profit barrier is hit before the lower stop-loss barrier and the time horizon expiration.

Model Training & Tracking (trainer.py & models.py) Deep Learning models (LSTM, TransformerEncoder) are trained with BCEWithLogitsLoss, dynamically applying pos_weight tensors to handle the heavy class imbalance inherent to stock market data.

Implements Custom Early Stopping.

Every model architecture, hyperparameter, and metric is automatically logged to DagsHub via MLflow.

Continuous Deployment (ci.yml) Pushes to the main branch trigger a GitHub Action that connects to the AWS EC2 instance via SSH, pulls the latest code, injects secret environment variables, and rebuilds the Docker container with zero-downtime using docker-compose.
📝 Postmortem & Key Learnings Handling Memory Bottlenecks in Windowing:

Problem: Creating overlapping sliding windows (e.g., 30-day lookbacks) for thousands of stocks caused massive RAM spikes during NumPy array concatenation.

Fix: Used numpy.lib.stride_tricks.sliding_window_view within Polars groupby loops, drastically reducing memory overhead by creating memory views instead of data copies until the final concatenation.

Class Imbalance in Financial Data:

Problem: Because the market doesn't hit a 3% profit target in 5 days very often, the dataset was heavily skewed toward 0 (No Trade).

Fix: Programmatically calculated the negative-to-positive ratio per training batch and injected it into the Transformer bias initialization init_bias = -np.log(ratio_val) and the PyTorch Loss function pos_weight, preventing the model from just predicting 0 every time.