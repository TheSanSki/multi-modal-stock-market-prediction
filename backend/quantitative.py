import logging
from typing import List, Dict

import numpy as np
import pandas as pd
from arch import arch_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _clean_data(series: pd.Series) -> pd.Series:
    """
    Purges or interpolates infinite or missing values.
    """
    cleaned = series.replace([np.inf, -np.inf], np.nan)
    if cleaned.isna().any():
        logger.warning("Input series contains NaNs or infs. Interpolating.")
        cleaned = cleaned.interpolate(method='linear').ffill().bfill()
    return cleaned

def calculate_log_returns(price_series: pd.Series) -> pd.Series:
    """
    Calculates the log returns of a price series.
    
    Args:
        price_series (pd.Series): Raw Close prices.
        
    Returns:
        pd.Series: Log returns, aligned with index, first index populated with 0.0.
        
    Raises:
        ValueError: If the input series contains negative or zero values.
    """
    price_series = _clean_data(price_series)
    
    if (price_series <= 0).any():
        raise ValueError("Price series contains zero or negative values, cannot calculate log returns.")
        
    log_returns = np.log(price_series / price_series.shift(1))
    log_returns.iloc[0] = 0.0 # Keep the first index but fill the resulting NaN with 0.0
    return log_returns

def forecast_volatility(returns: pd.Series, horizon: int = 90) -> np.ndarray:
    """
    Forecasts daily volatility using a GARCH(1,1) model.
    
    Args:
        returns (pd.Series): Log returns.
        horizon (int): The number of days to forecast.
        
    Returns:
        np.ndarray: An array of length `horizon` containing daily forecasted standard deviations.
    """
    returns = _clean_data(returns)
    
    # Scale returns by 100 for GARCH numerical stability
    scaled_returns = returns * 100.0
    
    try:
        # Instantiate and fit GARCH(1,1)
        am = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
        res = am.fit(disp='off')
        
        # Forecast variance
        forecasts = res.forecast(horizon=horizon, reindex=False)
        forecasted_variance_scaled = forecasts.variance.values[-1, :]
        
        # Scale back down
        forecasted_variance = forecasted_variance_scaled / 10000.0
        
        # Convert to daily standard deviation
        forecasted_std = np.sqrt(forecasted_variance)
        
    except Exception as e:
        logger.warning(f"GARCH optimization failed: {e}. Falling back to 21-day rolling standard deviation.")
        # Fallback: calculate historical 21-day rolling standard deviation
        rolling_std = returns.rolling(window=21).std().iloc[-1]
        if pd.isna(rolling_std):
            rolling_std = returns.std() # Ultimate fallback if not enough data
        forecasted_std = np.full(horizon, rolling_std)
        
    # Ensure no forecasted standard deviation value is <= 0 by clipping
    forecasted_std = np.clip(forecasted_std, a_min=1e-6, a_max=None)
    
    return forecasted_std

def generate_monte_carlo_envelope(
    current_price: float,
    expected_returns: np.ndarray,
    forecasted_vols: np.ndarray,
    horizon: int = 90,
    num_simulations: int = 1000
) -> Dict[str, List[float]]:
    """
    Generates a Monte Carlo simulation envelope for future price paths.
    
    Args:
        current_price (float): The last observed closing price.
        expected_returns (np.ndarray): Length H array of daily expected returns forecasted by the ML models.
        forecasted_vols (np.ndarray): Length H array of GARCH forecasted daily volatilities.
        horizon (int): Number of days to simulate.
        num_simulations (int): Number of paths to simulate.
        
    Returns:
        Dict[str, List[float]]: A dictionary containing the upper_bound (95th percentile),
                                median_prediction (50th percentile), and lower_bound (5th percentile).
    """
    if len(expected_returns) != horizon or len(forecasted_vols) != horizon:
        raise ValueError("expected_returns and forecasted_vols must have length equal to horizon.")
        
    # Ensure volatilities are positive
    forecasted_vols = np.clip(forecasted_vols, a_min=1e-6, a_max=None)
    
    # Create random standard normal samples: shape (horizon, num_simulations)
    Z = np.random.standard_normal((horizon, num_simulations))
    
    # Prepare expected_returns and forecasted_vols to broadcast over simulations
    # Reshape to (horizon, 1)
    mu = expected_returns.reshape(-1, 1)
    sigma = forecasted_vols.reshape(-1, 1)
    
    # Calculate the daily drift and diffusion terms
    # (r_hat_{t+1} - 0.5 * sigma_{t+1}^2) + sigma_{t+1} * Z
    daily_returns = (mu - 0.5 * sigma**2) + sigma * Z
    
    # Add an initial row of zeros for cumulative sum to start at zero
    # So we have shape (horizon + 1, num_simulations)
    cumulative_returns = np.vstack([np.zeros((1, num_simulations)), np.cumsum(daily_returns, axis=0)])
    
    # Price paths: S_t = S_0 * exp(cumulative_returns)
    price_paths = current_price * np.exp(cumulative_returns)
    
    # Drop the 0th element (which is current_price) to match the length of `horizon`.
    price_paths = price_paths[1:, :]
    
    # Calculate percentiles across simulations for each day
    upper_bound = np.percentile(price_paths, 95, axis=1).tolist()
    median_prediction = np.percentile(price_paths, 50, axis=1).tolist()
    lower_bound = np.percentile(price_paths, 5, axis=1).tolist()
    
    return {
        "upper_bound": upper_bound,
        "median_prediction": median_prediction,
        "lower_bound": lower_bound
    }
