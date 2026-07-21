# Susanoo Systems: Cryptocurrency Market Prediction Platform

This repository contains the source code for a Multi-Asset, Probabilistic, Spatial-Temporal Forecasting Engine. The platform is designed for institutional quantitative research and algorithmic trading analysis, featuring a complete live trading inference API.

## 1. Project Overview

The objective of this system is to predict the price action of multiple cryptocurrency assets simultaneously while inherently quantifying statistical risk and trading costs. Unlike simplistic directional models, this engine predicts both the categorical direction of the market (Actionable Up/Down/Neutral signals) and the entire continuous probability distribution of future price movements using a dual-head loss architecture.

The system natively calculates these forecasts for **Bitcoin (BTC), Ethereum (ETH), and Solana (SOL)** across three simultaneous time horizons: **15-minute, 1-hour, and 4-hour**.

## 2. Architecture Flow

The machine learning pipeline is built as a state-of-the-art hybrid deep learning architecture, forming a true Joint Spatial-Temporal Network:

1. **Variable Selection Network (VSN):** Utilizes Gated Linear Units (GLUs) to dynamically learn and weight the relevance of incoming high-frequency features based on the current market regime.
2. **Spatial Embedding (GNN 1):** Maps initial spatial dependencies to calculate the cross-market correlation and drag between BTC, ETH, and SOL in real-time.
3. **Temporal Processing (Mamba-3 MIMO):** A State Space Model (SSM) leveraging Rotary Position Embeddings (RoPE). It processes the temporal sequences of the structurally embedded assets, capturing cyclical momentum via complex-valued hidden states.
4. **Spatial Fusion (GNN 2):** A secondary graph layer that allows the temporally processed sequences to communicate cross-asset dependencies once more before making predictions.
5. **Dual-Head Output:** Optimizes a Combined Loss function (`loss_alpha = 0.5`). 
   - *Regression Head:* Optimizes Pinball Loss for probabilistic boundaries.
   - *Classification Head:* Optimizes Cross-Entropy Loss for Cost-Aware 3-class directional targets (Up, Down, Neutral) bounded by a 0.1% minimum trading fee threshold.

## 3. Dataset Configuration

The model requires high-dimensional Spatial-Temporal Tensors shaped `[Batch, Time, Assets, Features]`. The strictly chronological, leak-free data fusion pipeline (`data_pipeline.py`) structures historical data into a rigid 15-minute grid.

**Features per Asset Node:**
* **Price Action:** Open, High, Low, Close, Volume.
* **Microstructure Data:** Taker Buy Base Volume Ratio, Number of Trades, Average Trade Size.
* **Derived Structural:** Log Returns, 1-Day Rolling Volatility, 7-Day Rolling Volatility.
* **Momentum Oscillators:** RSI, MACD, Bollinger Band Width.
* **Global Context:** Continuous Bitcoin Perpetual Futures Funding Rates, Time-of-Day/Day-of-Week Sine/Cosine encodings.

### Data Download
Due to GitHub's file size limits, the massive historical CSV datasets required for training have been hosted externally. 
To train the architecture locally or on Colab, download the dataset directory from the following Google Drive link and place it in the `data/` folder:
[Download Susanoo Systems Datasets](https://drive.google.com/drive/folders/12zffGslgV24tN6UH4fbrC9B3XRnECgHx?usp=drive_link)

## 4. Environment Setup

### Prerequisites
* Python 3.9+
* Node.js (Optional, for frontend extension development)

### Virtual Environment Initialization
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
*(Note: Ensure PyTorch is installed with the appropriate CUDA/MPS configuration for your hardware).*

## 5. Execution Instructions

The repository features a robust 5-Fold Chronological Walk-Forward Validation training environment alongside a live production pipeline that connects directly to the Binance REST API for real-time inference.

### Model Training & Evaluation
The training pipeline outputs institutional-grade evaluation metrics (ROC AUC, PR AUC, MCC, Actionable Accuracy). You can run a hyperparameter grid search or train the final model directly:
```bash
# To run a hyperparameter sweep over multiple seeds:
python colab_hyperparameter_sweep.py

# To train the final model and export weights:
python colab_train_standalone.py
```

### Starting the Live Backend Engine
The inference engine is hosted via a FastAPI server. Start the server using Uvicorn:
```bash
.venv/bin/uvicorn api.main:app --reload
```
*The server will initialize the final Mamba-3 weights into memory and expose the `GET /predict/live` endpoint on port 8000.*

### Starting the Visualizer Dashboard
The frontend is a vanilla HTML/JS application utilizing Chart.js for real-time DOM rendering. Serve the dashboard via a lightweight Python HTTP server:
```bash
.venv/bin/python -m http.server 8080 -d frontend
```
Navigate to `http://localhost:8080` in your web browser. Initiating a forecast will trigger the backend to scrape live Binance data and execute the mathematically identical feature pipeline used in training. You can use the **Horizon Selector** dropdown to instantly toggle the charts between 15m, 1h, and 4h timeframes, and view the automated **Directional Signal Badges** (UP/DOWN/NEUTRAL) alongside the confidence boundaries.

## 6. Hardware Requirements

* **Minimum Requirements (Inference):** The pre-trained weights are highly optimized. Real-time inference can be executed efficiently on standard multi-core CPU infrastructure (e.g., Intel i5, Apple M1) or entry-level hardware accelerators. Minimum 8GB System RAM required.
* **Recommended Requirements (Training):** Training the architecture from scratch on the 118,000-row historical tensor requires a hardware accelerator with at least 16GB VRAM (e.g., Nvidia T4 on Google Colab, RTX 3080, or A100 for larger batch sizes). Minimum 16GB System RAM required.