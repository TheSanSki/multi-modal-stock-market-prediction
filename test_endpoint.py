import sys
import os
import pandas as pd
from backend.main import predict_stock_endpoint, PredictionRequest

request = PredictionRequest(ticker="AAPL", company_name="Apple Inc.", horizon=30)
try:
    response = predict_stock_endpoint(request)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
