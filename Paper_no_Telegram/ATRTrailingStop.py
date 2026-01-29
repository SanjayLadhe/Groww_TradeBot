"""
ATR (Average True Range) Trailing Stop Module

This module provides ATR calculation and ATR-based trailing stop functionality
using Wilder's smoothing method.

ATR measures market volatility by decomposing the entire range of an asset
for a given period.

True Range = max[(high - low), abs(high - prev_close), abs(low - prev_close)]
ATR = Wilder's Smoothed Average of True Range

Author: Algo Trading Bot
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range.

    True Range is the greatest of:
    1. Current High - Current Low
    2. |Current High - Previous Close|
    3. |Current Low - Previous Close|

    Args:
        df: DataFrame with OHLC data

    Returns:
        Series with True Range values
    """
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return true_range

    except Exception as e:
        logger.error(f"Error calculating True Range: {e}")
        return pd.Series(index=df.index)


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
    method: str = "wilder"
) -> pd.DataFrame:
    """
    Calculate Average True Range using Wilder's smoothing.

    Args:
        df: DataFrame with OHLC data
        period: ATR period (default 14)
        method: Smoothing method ('wilder' or 'sma')

    Returns:
        DataFrame with 'atr' column added
    """
    try:
        df = df.copy()

        # Calculate True Range
        df['tr'] = calculate_true_range(df)

        if method == "wilder":
            # Wilder's smoothing: ATR = ((ATR_prev * (n-1)) + TR) / n
            # Initialize with SMA
            df['atr'] = df['tr'].rolling(window=period).mean()

            # Apply Wilder's smoothing
            for i in range(period, len(df)):
                prev_atr = df['atr'].iloc[i-1]
                current_tr = df['tr'].iloc[i]
                df.loc[df.index[i], 'atr'] = ((prev_atr * (period - 1)) + current_tr) / period
        else:
            # Simple Moving Average
            df['atr'] = df['tr'].rolling(window=period).mean()

        # Clean up
        df.drop('tr', axis=1, inplace=True, errors='ignore')

        return df

    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        df['atr'] = 0
        return df


def calculate_atr_trailing_stop(
    df: pd.DataFrame,
    multiplier: float = 3.0,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate ATR-based trailing stop levels.

    Long Trailing Stop = Close - (ATR * Multiplier)
    Short Trailing Stop = Close + (ATR * Multiplier)

    Args:
        df: DataFrame with OHLC data
        multiplier: ATR multiplier (default 3.0)
        period: ATR period (default 14)

    Returns:
        DataFrame with 'atr_trailing_long' and 'atr_trailing_short' columns
    """
    try:
        df = df.copy()

        # Calculate ATR if not present
        if 'atr' not in df.columns:
            df = calculate_atr(df, period)

        # Calculate trailing stop levels
        df['atr_trailing_long'] = df['close'] - (df['atr'] * multiplier)
        df['atr_trailing_short'] = df['close'] + (df['atr'] * multiplier)

        return df

    except Exception as e:
        logger.error(f"Error calculating ATR trailing stop: {e}")
        return df


def calculate_chandelier_exit(
    df: pd.DataFrame,
    period: int = 22,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculate Chandelier Exit.

    Chandelier Exit (Long) = 22-day High - (ATR * 3)
    Chandelier Exit (Short) = 22-day Low + (ATR * 3)

    Args:
        df: DataFrame with OHLC data
        period: Lookback period for high/low
        multiplier: ATR multiplier

    Returns:
        DataFrame with chandelier exit levels
    """
    try:
        df = df.copy()

        if 'atr' not in df.columns:
            df = calculate_atr(df, 14)

        # Calculate period high and low
        df['period_high'] = df['high'].rolling(window=period).max()
        df['period_low'] = df['low'].rolling(window=period).min()

        # Calculate Chandelier exits
        df['chandelier_long'] = df['period_high'] - (df['atr'] * multiplier)
        df['chandelier_short'] = df['period_low'] + (df['atr'] * multiplier)

        # Clean up
        df.drop(['period_high', 'period_low'], axis=1, inplace=True, errors='ignore')

        return df

    except Exception as e:
        logger.error(f"Error calculating Chandelier Exit: {e}")
        return df


def get_atr_signal(
    df: pd.DataFrame,
    multiplier: float = 3.0
) -> dict:
    """
    Get ATR-based trading signal with trend direction.

    Args:
        df: DataFrame with OHLC and ATR data
        multiplier: ATR multiplier

    Returns:
        Dict with signal information
    """
    try:
        if 'atr' not in df.columns:
            df = calculate_atr(df)

        df = calculate_atr_trailing_stop(df, multiplier)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = latest['close']
        atr = latest['atr']
        long_stop = latest['atr_trailing_long']
        short_stop = latest['atr_trailing_short']

        prev_close = prev['close']
        prev_long_stop = prev['atr_trailing_long']

        # Determine trend
        if close > prev_long_stop:
            trend = "UPTREND"
            signal = "BUY"
        elif close < prev['atr_trailing_short']:
            trend = "DOWNTREND"
            signal = "SELL"
        else:
            trend = "SIDEWAYS"
            signal = "HOLD"

        # Calculate risk percentage
        risk_pct = (atr * multiplier / close) * 100

        return {
            "trend": trend,
            "signal": signal,
            "close": close,
            "atr": atr,
            "long_stop": long_stop,
            "short_stop": short_stop,
            "risk_pct": risk_pct
        }

    except Exception as e:
        logger.error(f"Error getting ATR signal: {e}")
        return {"trend": "UNKNOWN", "signal": "HOLD"}


def calculate_volatility_ratio(
    df: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20
) -> pd.DataFrame:
    """
    Calculate volatility ratio using ATR.

    Volatility Ratio = Short ATR / Long ATR
    - Ratio > 1: Increasing volatility
    - Ratio < 1: Decreasing volatility

    Args:
        df: DataFrame with OHLC data
        short_period: Short ATR period
        long_period: Long ATR period

    Returns:
        DataFrame with 'volatility_ratio' column
    """
    try:
        df = df.copy()

        # Calculate short and long ATR
        tr = calculate_true_range(df)
        df['atr_short'] = tr.rolling(window=short_period).mean()
        df['atr_long'] = tr.rolling(window=long_period).mean()

        # Calculate ratio
        df['volatility_ratio'] = df['atr_short'] / df['atr_long']

        # Clean up
        df.drop(['atr_short', 'atr_long'], axis=1, inplace=True, errors='ignore')

        return df

    except Exception as e:
        logger.error(f"Error calculating volatility ratio: {e}")
        return df


if __name__ == "__main__":
    # Test ATR calculation
    print("ATR Trailing Stop Module")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range(start='2026-01-01', periods=50, freq='D')
    np.random.seed(42)

    base_price = 100
    returns = np.random.randn(50) * 2

    sample_data = {
        'open': base_price + np.cumsum(returns),
        'high': base_price + np.cumsum(returns) + np.abs(np.random.randn(50)),
        'low': base_price + np.cumsum(returns) - np.abs(np.random.randn(50)),
        'close': base_price + np.cumsum(returns) + np.random.randn(50) * 0.5,
        'volume': np.random.randint(1000, 10000, 50)
    }

    df = pd.DataFrame(sample_data, index=dates)

    # Calculate ATR
    df = calculate_atr(df, period=14)
    df = calculate_atr_trailing_stop(df, multiplier=3)

    print("\nATR Calculation Results:")
    print(df[['close', 'atr', 'atr_trailing_long', 'atr_trailing_short']].tail(10))

    # Get signal
    signal = get_atr_signal(df)
    print(f"\nATR Signal: {signal}")
