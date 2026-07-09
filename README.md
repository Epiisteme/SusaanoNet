# Susanoo Systems: Cryptocurrency Market Prediction Platform

This repository contains the source code for a Multi-Asset, Probabilistic, Spatial-Temporal Forecasting Engine. The platform is designed for institutional quantitative research and algorithmic trading analysis.

## 1. Project Overview

The objective of this system is to predict the price action of multiple cryptocurrency assets simultaneously while inherently quantifying statistical risk. Unlike deterministic models that forecast absolute price targets, this engine predicts the entire probability distribution of future price movements using a Quantile Regression framework.

The system natively calculates the 10th percentile (Lower Confidence Bound), 50th percentile (Median Expected Value), and 90th percentile (Upper Confidence Bound) risk scenarios for Bitcoin (BTC), Ethereum (ETH), and Solana (SOL) on a 15-minute forecasting horizon.

## 2. Architecture Flow

The machine learning pipeline is built as a hybrid stack of modern deep learning architectures:

1. **Variable Selection Network (VSN):** Utilizes Gated Linear Units (GLUs) to dynamically learn and weight the relevance of incoming features based on the current market regime.
2. **Dynamic Graph Attention (GNN):** Maps spatial dependencies to calculate the cross-market correlation and drag between BTC, ETH, and SOL in real-time.
3. **Mamba-3 MIMO:** A State Space Model (SSM) leveraging Rotary Position Embeddings (RoPE). It processes the temporal sequences of all three assets in parallel streams, capturing cyclical momentum via complex-valued hidden states.
4. **Quantile Regression Head:** Optimizes Pinball Loss to output probabilistic boundaries rather than single-point estimates.

## 3. Dataset Configuration

The model requires high-dimensional Spatial-Temporal Tensors shaped `[Batch, Time, Assets, Features]`. The data fusion pipeline (`data_pipeline.py`) structures historical data into a rigid 15-minute grid.

**Features per Asset (14 Total):**
* **Price Action:** Open, High, Low, Close, Volume.
* **Derived Structural:** Log Returns, 1-Day Rolling Volatility, 7-Day Rolling Volatility.
* **Derivatives:** Continuous Bitcoin Perpetual Futures Funding Rates.
* **Macroeconomic:** S&P 500 Close, US Dollar Index (DXY), Gold Close, Bitcoin Hash Rate, Fear & Greed Index.

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

The repository features a live production pipeline that connects directly to the Binance REST API for real-time inference.

### Starting the Backend Engine
The inference engine is hosted via a FastAPI server. Start the server using Uvicorn:
```bash
.venv/bin/uvicorn api.main:app --reload
```
*The server will initialize the Mamba-3 weights into memory and expose the `GET /predict/live` endpoint on port 8000.*

### Starting the Visualizer Dashboard
The frontend is a vanilla HTML/JS application utilizing Chart.js for real-time DOM rendering. Serve the dashboard via a lightweight Python HTTP server:
```bash
.venv/bin/python -m http.server 8080 -d frontend
```
Navigate to `http://localhost:8080` in your web browser. Initiating a forecast will trigger the backend to scrape the last 15 hours of live Binance data, execute the inference pass, and overlay the quantile predictions onto the historical chart.

## 6. Hardware Requirements

* **Training (Colab Standalone):** Training the architecture on the 118,000-row historical tensor requires an Nvidia A100 GPU (40GB VRAM) or equivalent. Training duration is approximately 2-4 hours.
* **Inference (Live Production):** The trained weights are highly optimized. Real-time inference can be executed efficiently on standard CPU infrastructure or entry-level hardware accelerators (e.g., Apple M-Series Silicon or Nvidia T4). Memory overhead is less than 2GB during active inference.