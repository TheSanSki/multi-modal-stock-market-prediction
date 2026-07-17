import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta, timezone
import pytz
import urllib.parse
import re

import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lexicon Enhancement Rule
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
sia.lexicon.update({
    'bearish': -1.5, 
    'bullish': 1.5, 
    'short': -1.0, 
    'squeeze': 1.5, 
    'rally': 2.0, 
    'plummet': -2.5, 
    'bankruptcy': -3.0
})

def fetch_market_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Downloads historical market data.
    
    Args:
        ticker (str): The stock ticker symbol.
        period (str): The historical time period to fetch.
        
    Returns:
        pd.DataFrame: A cleaned DataFrame containing the 'Close' column with a datetime index.
    """
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}.")
        
        # Handle multi-index columns returned by newer versions of yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close_col = data['Close'][ticker]
        else:
            close_col = data['Close']
            
        df = pd.DataFrame({'Close': close_col})
        
        # Defensive Check: Clean NaN or infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        if df.empty:
            raise ValueError("Dataframe became empty after cleaning.")
            
        # Standardize index to be tz-naive (representing the local trading date)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Strip time part if present, keeping just the date
        df.index = pd.to_datetime(df.index.date)
        
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch market data: {e}")
        return pd.DataFrame(columns=['Close'])

def fetch_and_score_raw_sentiment(company_name: str, ticker: str) -> List[Dict[str, Any]]:
    """
    Fetches Google News RSS and mimics Social Streams, returning scored sentiments.
    
    Args:
        company_name (str): Company name for news search.
        ticker (str): Ticker for social stream simulation.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing source, timestamp, compound score, and engagement.
    """
    results = []
    
    try:
        # Google News Scraping
        query = urllib.parse.quote(company_name)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        for entry in feed.entries:
            text = f"{entry.title} {entry.get('summary', '')}"
            text_clean = re.sub(r'http\S+', '', text).lower()
            score = sia.polarity_scores(text_clean)['compound']
            
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                import time
                import calendar
                dt = datetime.fromtimestamp(calendar.timegm(entry.published_parsed), tz=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
                
            results.append({
                "source": "news",
                "timestamp": dt,
                "compound": score,
                "engagement": 1 # Default engagement
            })
    except Exception as e:
        logger.warning(f"Failed to scrape Google News: {e}")
        
    try:
        # Social Stream Generation (Mock fallback)
        now = datetime.now(timezone.utc)
        for i in range(1, 15): # Generate a couple weeks of data
            past_dt = now - timedelta(days=i * 0.75)
            
            # Reddit mimic
            text_reddit = f"${ticker} is looking super bullish right now. huge squeeze incoming!"
            score_reddit = sia.polarity_scores(text_reddit.lower())['compound']
            results.append({
                "source": "reddit",
                "timestamp": past_dt,
                "compound": score_reddit,
                "engagement": np.random.randint(50, 1000)
            })
            
            # Twitter mimic
            text_twitter = f"Just bought more $ {ticker}. Ready to rally."
            score_twitter = sia.polarity_scores(text_twitter.lower())['compound']
            results.append({
                "source": "twitter",
                "timestamp": past_dt - timedelta(hours=3),
                "compound": score_twitter,
                "engagement": np.random.randint(10, 500)
            })
    except Exception as e:
        logger.warning(f"Failed to generate social stream: {e}")
        
    return results

def aggregate_sentiment_to_calendar(sentiment_data: List[Dict[str, Any]], trading_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Aggregates sentiment data to match the trading calendar using VWDV.
    
    Args:
        sentiment_data (List[Dict[str, Any]]): Raw scored sentiment events.
        trading_dates (pd.DatetimeIndex): Trading days from the market data.
        
    Returns:
        pd.Series: Sentiment index aligned to trading dates.
    """
    est = pytz.timezone('America/New_York')
    weight_map = {'news': 0.50, 'reddit': 0.30, 'twitter': 0.20}
    processed_sentiments = []
    
    for item in sentiment_data:
        ts = item['timestamp']
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_est = ts.astimezone(est)
        
        w = weight_map.get(item['source'].lower(), 0.1)
        e = 1.0 + np.log(1.0 + item['engagement'])
        
        processed_sentiments.append({
            'ts_est': ts_est,
            'w': w,
            'e': e,
            'compound': item['compound']
        })
        
    sentiment_series = pd.Series(0.0, index=trading_dates)
    if not processed_sentiments:
        return sentiment_series
        
    lambda_val = 0.05
    
    for current_date in trading_dates:
        # Target close: Day T, 16:00 EST
        current_close = est.localize(datetime(current_date.year, current_date.month, current_date.day, 16, 0))
        
        # Identify the previous cutoff (Day T-1, 16:00 EST)
        idx_pos = trading_dates.get_loc(current_date)
        if idx_pos == 0:
            prev_date = current_date - timedelta(days=1)
            prev_close = est.localize(datetime(prev_date.year, prev_date.month, prev_date.day, 16, 0))
        else:
            prev_date = trading_dates[idx_pos - 1]
            prev_close = est.localize(datetime(prev_date.year, prev_date.month, prev_date.day, 16, 0))
            
        # Cut-off Window Filter: After Day T-1 16:00 EST up to Day T 16:00 EST
        window_items = [
            item for item in processed_sentiments 
            if prev_close < item['ts_est'] <= current_close
        ]
        
        if not window_items:
            sentiment_series[current_date] = 0.0
            continue
            
        numerator = 0.0
        denominator = 0.0
        
        for item in window_items:
            # Delta t in decimal hours relative to Target Close
            dt_hours = (current_close - item['ts_est']).total_seconds() / 3600.0
            dt_hours = max(0.0, dt_hours)
            
            decay = np.exp(-lambda_val * dt_hours)
            numerator += item['compound'] * item['w'] * item['e'] * decay
            denominator += item['w'] * item['e'] * decay
            
        sentiment_series[current_date] = (numerator / denominator) if denominator > 0 else 0.0
        
    return sentiment_series

def build_integrated_pipeline(ticker: str, company_name: str, period: str = "1y") -> pd.DataFrame:
    """
    Master coordinator coordinating ingestion, sentiment, and alignment.
    
    Args:
        ticker (str): Ticker symbol.
        company_name (str): Company name for news search.
        period (str): Data fetch period.
        
    Returns:
        pd.DataFrame: Unified DataFrame with ['Close', 'Sentiment'].
    """
    try:
        df_market = fetch_market_data(ticker, period)
        if df_market.empty:
            logger.warning("Market data is empty. Returning blank DataFrame.")
            return pd.DataFrame(columns=['Close', 'Sentiment'])
            
        trading_dates = df_market.index
        sentiment_raw = fetch_and_score_raw_sentiment(company_name, ticker)
        
        sentiment_series = aggregate_sentiment_to_calendar(sentiment_raw, trading_dates)
        df_market['Sentiment'] = sentiment_series
        
        return df_market
        
    except Exception as e:
        logger.error(f"Integrated pipeline failed: {e}")
        return pd.DataFrame(columns=['Close', 'Sentiment'])
