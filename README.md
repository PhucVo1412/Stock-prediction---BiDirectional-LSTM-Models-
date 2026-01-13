# 📈 Stock Price Prediction using Bidirectional LSTM (BiLSTM)

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Overview
This project focuses on forecasting the stock prices of **Apple Inc. (AAPL)** using Deep Learning techniques. The primary objective is to demonstrate the superiority of **Bidirectional Long Short-Term Memory (BiLSTM)** networks over standard **LSTM** models in capturing complex, non-linear market patterns.

Key goals:
* Predict **Adjusted Closing Prices** based on historical data.
* Compare performance metrics (RMSE, MAE, R²) between BiLSTM and LSTM.
* Evaluate **Directional Accuracy (DA)** to assess the model's ability to predict market trends (Up/Down).

## 📊 Dataset
* **Source:** Historical stock data for AAPL.
* **Time Period:** Late 2020 to Late 2025.
* **Features:** Open, High, Low, Close, Volume.
* **Preprocessing:** * Data cleaning and date alignment.
    * **Feature Selection:** Pearson Correlation analysis revealed high multicollinearity among OHLC features. Thus, the input vector was optimized to use **Close Price** only.
    * **Normalization:** Min-Max Scaling (0, 1).
    * **Sliding Window:** 60 days look-back period.

## 🏗️ Model Architecture

### 1. Bidirectional LSTM (Proposed)
* Processes data in two directions (Forward & Backward).
* Captures context from both past and "future" information within the sliding window.
* **Structure:** 2 BiLSTM Layers (64 units) + Dropout (0.2) + Dense Output.

### 2. Standard LSTM (Baseline)
* Processes data in a single forward direction.
* Used as a baseline to benchmark the improvements of the BiLSTM architecture.

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/PhucVo1412/Stock-prediction---BiDirectional-LSTM-Models-.git
cd Stock-prediction---BiDirectional-LSTM-Models-
