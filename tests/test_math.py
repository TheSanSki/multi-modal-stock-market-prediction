import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import pytz

# Add the parent directory to sys.path to allow importing from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.quantitative import calculate_log_returns, forecast_volatility, generate_monte_carlo_envelope
from backend.data_pipeline import aggregate_sentiment_to_calendar

class TestMathematicalFormulas(unittest.TestCase):

    def test_calculate_log_returns(self):
        """
        Test Case 1: Log Returns Alignment & Precision
        """
        prices = pd.Series([100.0, 110.0, 99.0, 105.0])
        log_returns = calculate_log_returns(prices)
        
        # Expected manual calculation
        expected_returns = np.array([
            0.0,
            np.log(110.0 / 100.0), # ~0.095310
            np.log(99.0 / 110.0),  # ~-0.105360
            np.log(105.0 / 99.0)   # ~0.058840
        ])
        
        # Verify first index is exactly 0.0
        self.assertEqual(log_returns.iloc[0], 0.0)
        
        # Assert array matches expected values to 5 decimal places
        np.testing.assert_array_almost_equal(log_returns.values, expected_returns, decimal=5)

    def test_vwdv_aggregation(self):
        """
        Test Case 2: Volume-Weighted Decay Vector (VWDV) Equation Verification
        """
        est = pytz.timezone('America/New_York')
        # Create a target date where market closes at 16:00 EST
        target_date = pd.Timestamp('2023-10-10')
        target_time_est = est.localize(datetime(2023, 10, 10, 16, 0))
        target_time = target_time_est.astimezone(timezone.utc)
        
        trading_dates = pd.DatetimeIndex([target_date])
        
        sentiment_data = [
            {"source": "news", "timestamp": target_time, "compound": 0.8, "engagement": 0},
            {"source": "reddit", "timestamp": target_time, "compound": -0.4, "engagement": 5}
        ]
        
        result_series = aggregate_sentiment_to_calendar(sentiment_data, trading_dates)
        
        # Manual verification math
        W_news = 0.50
        E_news = 1.0 + np.log(1.0 + 0) # 1.0
        Weight_1 = W_news * E_news # 0.50
        
        W_reddit = 0.30
        E_reddit = 1.0 + np.log(1.0 + 5) # ~2.791759
        Weight_2 = W_reddit * E_reddit # ~0.837527
        
        numerator = (0.8 * Weight_1) + (-0.4 * Weight_2)
        denominator = Weight_1 + Weight_2
        expected_score = numerator / denominator # ~0.04859
        
        actual_score = result_series.loc[target_date]
        self.assertAlmostEqual(actual_score, expected_score, places=4)

    def test_garch_exception_handling(self):
        """
        Test Case 3: GARCH Non-Convergence Fallback Handling
        """
        # Pass a series of returns containing only flat zeros
        flat_returns = pd.Series([0.0] * 100)
        horizon = 10
        
        forecasted_vols = forecast_volatility(flat_returns, horizon=horizon)
        
        # Verify length matches requested horizon
        self.assertEqual(len(forecasted_vols), horizon)
        
        # Verify contains non-null values
        self.assertFalse(np.isnan(forecasted_vols).any())
        
        # Verify it fallback properly by checking the standard deviation logic
        # Fallback value is the standard deviation of flat returns, which is 0.0, 
        # but bounded by a_min=1e-6 via np.clip
        np.testing.assert_array_almost_equal(forecasted_vols, np.full(horizon, 1e-6))

    def test_monte_carlo_envelope(self):
        """
        Test Case 4: Monte Carlo Envelope Quantile Boundaries
        """
        horizon = 90
        current_price = 100.0
        expected_returns = np.full(horizon, 0.001)
        forecasted_vols = np.full(horizon, 0.02)
        
        # Generate envelope
        envelope = generate_monte_carlo_envelope(
            current_price=current_price,
            expected_returns=expected_returns,
            forecasted_vols=forecasted_vols,
            horizon=horizon,
            num_simulations=1000
        )
        
        upper_bound = envelope["upper_bound"]
        median_pred = envelope["median_prediction"]
        lower_bound = envelope["lower_bound"]
        
        # Verify lengths
        self.assertEqual(len(upper_bound), horizon)
        self.assertEqual(len(median_pred), horizon)
        self.assertEqual(len(lower_bound), horizon)
        
        # Verify structural inequalities
        for t in range(horizon):
            self.assertTrue(lower_bound[t] <= median_pred[t] <= upper_bound[t], 
                            msg=f"Ordering violated at index {t}: {lower_bound[t]} <= {median_pred[t]} <= {upper_bound[t]}")

if __name__ == '__main__':
    unittest.main()
