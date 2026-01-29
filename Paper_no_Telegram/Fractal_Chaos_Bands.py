"""
Fractal Chaos Bands Indicator Module

Fractal Chaos Bands use fractal patterns to identify support and resistance levels.
Fractals are patterns where the middle bar has the highest high (fractal high)
or lowest low (fractal low) compared to surrounding bars.

Components:
- Upper Band: Based on fractal highs
- Lower Band: Based on fractal lows
- Middle Band: Average of upper and lower

Usage:
- Price above upper band: Strong bullish (breakout)
- Price below lower band: Strong bearish (breakdown)
- Price within bands: Consolidation

Author: Algo Trading Bot
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def identify_fractal_high(
    df: pd.DataFrame,
    window: int = 2
) -> pd.Series:
    """
    Identify fractal highs.

    A fractal high occurs when a bar's high is higher than the highs
    of 'window' bars on both sides.

    Args:
        df: DataFrame with OHLC data
        window: Number of bars on each side to compare

    Returns:
        Series with fractal high values (NaN where no fractal)
    """
    try:
        high = df['high']
        fractal_high = pd.Series(index=df.index, dtype=float)

        for i in range(window, len(df) - window):
            is_fractal = True
            center_high = high.iloc[i]

            # Check if center is higher than all surrounding bars
            for j in range(1, window + 1):
                if high.iloc[i - j] >= center_high or high.iloc[i + j] >= center_high:
                    is_fractal = False
                    break

            if is_fractal:
                fractal_high.iloc[i] = center_high

        return fractal_high

    except Exception as e:
        logger.error(f"Error identifying fractal highs: {e}")
        return pd.Series(index=df.index, dtype=float)


def identify_fractal_low(
    df: pd.DataFrame,
    window: int = 2
) -> pd.Series:
    """
    Identify fractal lows.

    A fractal low occurs when a bar's low is lower than the lows
    of 'window' bars on both sides.

    Args:
        df: DataFrame with OHLC data
        window: Number of bars on each side to compare

    Returns:
        Series with fractal low values (NaN where no fractal)
    """
    try:
        low = df['low']
        fractal_low = pd.Series(index=df.index, dtype=float)

        for i in range(window, len(df) - window):
            is_fractal = True
            center_low = low.iloc[i]

            # Check if center is lower than all surrounding bars
            for j in range(1, window + 1):
                if low.iloc[i - j] <= center_low or low.iloc[i + j] <= center_low:
                    is_fractal = False
                    break

            if is_fractal:
                fractal_low.iloc[i] = center_low

        return fractal_low

    except Exception as e:
        logger.error(f"Error identifying fractal lows: {e}")
        return pd.Series(index=df.index, dtype=float)


def calculate_fractal_chaos_bands(
    df: pd.DataFrame,
    window: int = 2,
    smoothing: int = 5
) -> pd.DataFrame:
    """
    Calculate Fractal Chaos Bands.

    Steps:
    1. Identify fractal highs and lows
    2. Forward fill to create continuous bands
    3. Optionally smooth the bands

    Args:
        df: DataFrame with OHLC data
        window: Fractal identification window (bars on each side)
        smoothing: Smoothing period for bands (0 for no smoothing)

    Returns:
        DataFrame with 'upper_band', 'lower_band', 'middle_band' columns
    """
    try:
        df = df.copy()

        # Identify fractals
        fractal_high = identify_fractal_high(df, window)
        fractal_low = identify_fractal_low(df, window)

        # Forward fill to create continuous bands
        df['upper_band'] = fractal_high.ffill()
        df['lower_band'] = fractal_low.ffill()

        # Fill initial NaN values with high/low
        df['upper_band'] = df['upper_band'].fillna(df['high'])
        df['lower_band'] = df['lower_band'].fillna(df['low'])

        # Apply smoothing if specified
        if smoothing > 0:
            df['upper_band'] = df['upper_band'].rolling(window=smoothing).mean()
            df['lower_band'] = df['lower_band'].rolling(window=smoothing).mean()
            df['upper_band'] = df['upper_band'].fillna(method='bfill')
            df['lower_band'] = df['lower_band'].fillna(method='bfill')

        # Calculate middle band
        df['middle_band'] = (df['upper_band'] + df['lower_band']) / 2

        # Store fractal points for visualization
        df['fractal_high'] = fractal_high
        df['fractal_low'] = fractal_low

        return df

    except Exception as e:
        logger.error(f"Error calculating Fractal Chaos Bands: {e}")
        df['upper_band'] = df['high']
        df['lower_band'] = df['low']
        df['middle_band'] = (df['high'] + df['low']) / 2
        return df


def get_fractal_signal(df: pd.DataFrame) -> dict:
    """
    Get trading signal based on Fractal Chaos Bands.

    Signal Logic:
    - Close > Upper Band: Bullish breakout
    - Close < Lower Band: Bearish breakdown
    - Close within bands: Consolidation

    Args:
        df: DataFrame with Fractal Chaos Bands calculated

    Returns:
        Dict with signal information
    """
    try:
        if 'upper_band' not in df.columns:
            df = calculate_fractal_chaos_bands(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = latest['close']
        upper = latest['upper_band']
        lower = latest['lower_band']
        middle = latest['middle_band']

        prev_close = prev['close']
        prev_upper = prev['upper_band']
        prev_lower = prev['lower_band']

        # Determine position relative to bands
        if close > upper:
            position = "ABOVE_UPPER"
            signal = "STRONG_BULLISH"
        elif close < lower:
            position = "BELOW_LOWER"
            signal = "STRONG_BEARISH"
        elif close > middle:
            position = "UPPER_HALF"
            signal = "BULLISH"
        elif close < middle:
            position = "LOWER_HALF"
            signal = "BEARISH"
        else:
            position = "MIDDLE"
            signal = "NEUTRAL"

        # Check for breakout/breakdown
        breakout = None
        if prev_close <= prev_upper and close > upper:
            breakout = "BULLISH_BREAKOUT"
        elif prev_close >= prev_lower and close < lower:
            breakout = "BEARISH_BREAKDOWN"

        # Calculate band width percentage
        band_width = upper - lower
        band_width_pct = (band_width / close) * 100

        # Check for squeeze (narrow bands)
        is_squeeze = band_width_pct < 2.0  # Arbitrary threshold

        return {
            "signal": signal,
            "position": position,
            "close": close,
            "upper_band": upper,
            "lower_band": lower,
            "middle_band": middle,
            "breakout": breakout,
            "band_width_pct": band_width_pct,
            "is_squeeze": is_squeeze
        }

    except Exception as e:
        logger.error(f"Error getting fractal signal: {e}")
        return {"signal": "NEUTRAL", "position": "UNKNOWN"}


def get_support_resistance_levels(
    df: pd.DataFrame,
    window: int = 2,
    lookback: int = 50
) -> dict:
    """
    Get support and resistance levels from fractal points.

    Args:
        df: DataFrame with OHLC data
        window: Fractal window
        lookback: Number of bars to look back for levels

    Returns:
        Dict with support and resistance levels
    """
    try:
        df = df.tail(lookback).copy()

        # Identify fractals
        fractal_high = identify_fractal_high(df, window)
        fractal_low = identify_fractal_low(df, window)

        # Get valid fractal levels
        resistance_levels = fractal_high.dropna().values.tolist()
        support_levels = fractal_low.dropna().values.tolist()

        # Sort and get unique levels
        resistance_levels = sorted(set(resistance_levels), reverse=True)
        support_levels = sorted(set(support_levels))

        # Get nearest levels
        current_price = df['close'].iloc[-1]

        nearest_resistance = None
        for level in resistance_levels:
            if level > current_price:
                nearest_resistance = level
                break

        nearest_support = None
        for level in reversed(support_levels):
            if level < current_price:
                nearest_support = level
                break

        return {
            "resistance_levels": resistance_levels[:5],  # Top 5
            "support_levels": support_levels[-5:],  # Bottom 5
            "nearest_resistance": nearest_resistance,
            "nearest_support": nearest_support,
            "current_price": current_price
        }

    except Exception as e:
        logger.error(f"Error getting support/resistance levels: {e}")
        return {"resistance_levels": [], "support_levels": []}


if __name__ == "__main__":
    # Test Fractal Chaos Bands
    print("Fractal Chaos Bands Module")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range(start='2026-01-01', periods=100, freq='D')
    np.random.seed(42)

    base_price = 100
    volatility = 2

    sample_data = {
        'open': base_price + np.random.randn(100).cumsum() * volatility,
        'high': base_price + np.random.randn(100).cumsum() * volatility + np.abs(np.random.randn(100)),
        'low': base_price + np.random.randn(100).cumsum() * volatility - np.abs(np.random.randn(100)),
        'close': base_price + np.random.randn(100).cumsum() * volatility,
        'volume': np.random.randint(1000, 10000, 100)
    }

    df = pd.DataFrame(sample_data, index=dates)

    # Ensure high > low
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 0.5
    df['low'] = df[['open', 'low', 'close']].min(axis=1) - 0.5

    # Calculate bands
    df = calculate_fractal_chaos_bands(df, window=2)

    print("\nFractal Chaos Bands:")
    print(df[['close', 'lower_band', 'middle_band', 'upper_band']].tail(10))

    # Get signal
    signal = get_fractal_signal(df)
    print(f"\nFractal Signal: {signal}")

    # Get support/resistance
    sr_levels = get_support_resistance_levels(df)
    print(f"\nSupport/Resistance: {sr_levels}")
