import sys
import os

# Robust path-correction fallback script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from typing import List
from datetime import timedelta
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Internal Module Integration
from data_pipeline import build_integrated_pipeline
from quantitative import calculate_log_returns, forecast_volatility, generate_monte_carlo_envelope

# Attempt to import ML_models components, fallback to mock if not yet implemented
try:
    from ML_models import (
        prepare_sliding_windows,
        train_lstm_configuration,
        train_random_forest_baseline,
        recursive_multi_step_forecast,
        inverse_transform_predictions
    )
except ImportError:
    # Fallback mock functions if ML_models.py is still empty
    def prepare_sliding_windows(df): return None, None, None
    def train_lstm_configuration(X, y): return None
    def train_random_forest_baseline(X, y): return None
    def recursive_multi_step_forecast(model, df, horizon): return np.zeros(horizon)
    def inverse_transform_predictions(scaler, scaled_data): return scaled_data

# Initialize FastAPI app
app = FastAPI(title="Quantitative Forecasting Gateway")

# Configure CORS Security Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Request Schema
class PredictionRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol (e.g., AAPL)")
    company_name: str = Field(..., min_length=1, description="Company name for sentiment extraction")
    horizon: int = Field(default=90, ge=1, le=120, description="Forecast horizon in days")

# Response Schema
class PredictionResponse(BaseModel):
    dates: List[str]
    historical_dates: List[str]
    historical_prices: List[float]
    lstm_median_path: List[float]
    rf_median_path: List[float]
    lower_volatility_bound: List[float]
    upper_volatility_bound: List[float]

@app.post("/api/predict", response_model=PredictionResponse)
def predict_stock_endpoint(request: PredictionRequest):
    try:
        logger.info(f"Initiating prediction matrix for {request.ticker} ({request.company_name}), horizon: {request.horizon}")
        
        # 1. Data Ingestion Phase
        df = build_integrated_pipeline(ticker=request.ticker, company_name=request.company_name, period="2y")
        if df.empty or 'Close' not in df.columns:
            raise ValueError("Data ingestion failed. Empty dataframe returned.")
            
        historical_dates = pd.to_datetime(df.index).strftime('%Y-%m-%d').tolist()
        historical_prices = df['Close'].tolist()
        last_price = historical_prices[-1]
        
        # 2. Transformation & Split
        log_returns = calculate_log_returns(df['Close'])
        df['Log_Returns'] = log_returns
        df = df.dropna()
        
        # Sliding window partition
        X_train, y_train, scaler = prepare_sliding_windows(df)
        
        # 3. Model Generation Engine
        lstm_model = train_lstm_configuration(X_train, y_train)
        rf_model = train_random_forest_baseline(X_train, y_train)
        
        # 4. Forecasting Matrix
        expected_returns_lstm_scaled = recursive_multi_step_forecast(lstm_model, df, request.horizon)
        expected_returns_rf_scaled = recursive_multi_step_forecast(rf_model, df, request.horizon)
        
        # Expected returns (assuming model predicts scaled log returns)
        expected_returns_lstm = inverse_transform_predictions(scaler, expected_returns_lstm_scaled)
        expected_returns_rf = inverse_transform_predictions(scaler, expected_returns_rf_scaled)
        
        # 5. Stochastic Volatility Layers
        forecasted_vols = forecast_volatility(df['Log_Returns'], horizon=request.horizon)
        
        # Monte Carlo Simulation Engine
        mc_results = generate_monte_carlo_envelope(
            current_price=last_price,
            expected_returns=expected_returns_lstm, # using LSTM as the primary drift vector
            forecasted_vols=forecasted_vols,
            horizon=request.horizon,
            num_simulations=1000
        )
        
        # Generate LSTM Price Path
        lstm_price_path = mc_results['median_prediction']
        
        # Generate RF Price Path
        mc_results_rf = generate_monte_carlo_envelope(
            current_price=last_price,
            expected_returns=expected_returns_rf,
            forecasted_vols=forecasted_vols,
            horizon=request.horizon,
            num_simulations=100
        )
        rf_price_path = mc_results_rf['median_prediction']
        
        # 6. De-scaling and Output Structuring
        last_date = df.index[-1]
        future_dates_idx = pd.bdate_range(start=last_date + timedelta(days=1), periods=request.horizon)
        future_dates = future_dates_idx.strftime('%Y-%m-%d').tolist()
        
        return PredictionResponse(
            dates=future_dates,
            historical_dates=historical_dates,
            historical_prices=historical_prices,
            lstm_median_path=lstm_price_path,
            rf_median_path=rf_price_path,
            lower_volatility_bound=mc_results['lower_bound'],
            upper_volatility_bound=mc_results['upper_bound']
        )
        
    except Exception as e:
        logger.error(f"Internal quantitative engine computation failure: {e}")
        raise HTTPException(status_code=500, detail="Internal quantitative engine computation failure.")
