"""
VWAP (Volume Weighted Average Price) Indicator Module

VWAP is a trading benchmark that represents the average price a security has
traded at throughout the day, based on both volume and price.

Formula: VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
Where: Typical Price = (High + Low + Close) / 3

Usage:
- Price above VWAP: Bullish bias
- Price below VWAP: Bearish bias
- VWAP acts as dynamic support/resistance

Author: Algo Trading Bot
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def calculate_vwap(
    df: pd.DataFrame,
    reset_daily: bool = True
) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price.

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        reset_daily: If True, reset VWAP calculation at market open each day

    Returns:
        DataFrame with 'vwap' column added
    """
    try:
        df = df.copy()

        # Ensure required columns exist
        required = ['high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate Typical Price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Calculate TP * Volume
        df['tp_volume'] = df['typical_price'] * df['volume']

        if reset_daily and isinstance(df.index, pd.DatetimeIndex):
            # Group by date and calculate cumulative sums
            df['date'] = df.index.date
            df['cumulative_tp_volume'] = df.groupby('date')['tp_volume'].cumsum()
            df['cumulative_volume'] = df.groupby('date')['volume'].cumsum()
            df.drop('date', axis=1, inplace=True)
        else:
            # Calculate cumulative sums for entire period
            df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()

        # Calculate VWAP
        df['vwap'] = df['cumulative_tp_volume'] / df['cumulative_volume']

        # Handle division by zero
        df['vwap'] = df['vwap'].replace([np.inf, -np.inf], np.nan)
        df['vwap'] = df['vwap'].fillna(df['close'])

        # Clean up temporary columns
        df.drop(['typical_price', 'tp_volume', 'cumulative_tp_volume', 'cumulative_volume'],
                axis=1, inplace=True, errors='ignore')

        return df

    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
        df['vwap'] = df['close']
        return df


def calculate_vwap_bands(
    df: pd.DataFrame,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate VWAP with standard deviation bands.

    Args:
        df: DataFrame with OHLCV data
        num_std: Number of standard deviations for bands

    Returns:
        DataFrame with 'vwap', 'vwap_upper', 'vwap_lower' columns
    """
    try:
        df = calculate_vwap(df)

        # Calculate standard deviation of typical price from VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap_diff'] = df['typical_price'] - df['vwap']
        df['vwap_std'] = df['vwap_diff'].rolling(window=20).std()

        # Calculate bands
        df['vwap_upper'] = df['vwap'] + (num_std * df['vwap_std'])
        df['vwap_lower'] = df['vwap'] - (num_std * df['vwap_std'])

        # Clean up
        df.drop(['typical_price', 'vwap_diff', 'vwap_std'],
                axis=1, inplace=True, errors='ignore')

        return df

    except Exception as e:
        logger.error(f"Error calculating VWAP bands: {e}")
        return df


def get_vwap_signal(df: pd.DataFrame) -> dict:
    """
    Get VWAP-based trading signal.

    Args:
        df: DataFrame with VWAP calculated

    Returns:
        Dict with signal information
    """
    try:
        if 'vwap' not in df.columns:
            df = calculate_vwap(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = latest['close']
        vwap = latest['vwap']
        prev_close = prev['close']
        prev_vwap = prev['vwap']

        # Determine bias
        if close > vwap:
            bias = "BULLISH"
        elif close < vwap:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        # Check for crossover
        crossover = None
        if prev_close <= prev_vwap and close > vwap:
            crossover = "BULLISH_CROSSOVER"
        elif prev_close >= prev_vwap and close < vwap:
            crossover = "BEARISH_CROSSOVER"

        # Calculate distance from VWAP
        distance_pct = ((close - vwap) / vwap) * 100

        return {
            "bias": bias,
            "crossover": crossover,
            "close": close,
            "vwap": vwap,
            "distance_pct": distance_pct
        }

    except Exception as e:
        logger.error(f"Error getting VWAP signal: {e}")
        return {"bias": "NEUTRAL", "crossover": None}


if __name__ == "__main__":
    # Test the VWAP calculation
    print("VWAP Indicator Module")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range(start='2026-01-01 09:15', periods=100, freq='5min')
    np.random.seed(42)

    sample_data = {
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 101 + np.random.randn(100).cumsum(),
        'low': 99 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }

    df = pd.DataFrame(sample_data, index=dates)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Calculate VWAP
    df = calculate_vwap(df)

    print("\nSample VWAP Calculation:")
    print(df[['close', 'volume', 'vwap']].tail(10))

    # Get signal
    signal = get_vwap_signal(df)
    print(f"\nVWAP Signal: {signal}")
