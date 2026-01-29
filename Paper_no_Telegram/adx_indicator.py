"""
ADX (Average Directional Index) Indicator Module

ADX measures the strength of a trend regardless of its direction.
Combined with +DI and -DI, it helps identify:
1. Whether the market is trending
2. The direction of the trend (bullish/bearish)
3. The strength of the trend

Components:
- +DI (Positive Directional Indicator): Measures upward movement
- -DI (Negative Directional Indicator): Measures downward movement
- ADX: Measures trend strength (0-100 scale)

Interpretation:
- ADX > 25: Strong trend
- ADX < 20: Weak or no trend
- +DI > -DI: Bullish trend
- -DI > +DI: Bearish trend

Author: Algo Trading Bot
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_directional_movement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Directional Movement (+DM and -DM).

    +DM = Current High - Previous High (if positive and > -DM)
    -DM = Previous Low - Current Low (if positive and > +DM)

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with '+dm' and '-dm' columns
    """
    try:
        df = df.copy()

        high = df['high']
        low = df['low']

        # Calculate directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        # +DM: positive up move that's greater than down move
        df['+dm'] = np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        )

        # -DM: positive down move that's greater than up move
        df['-dm'] = np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        )

        return df

    except Exception as e:
        logger.error(f"Error calculating directional movement: {e}")
        return df


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range.

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

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    except Exception as e:
        logger.error(f"Error calculating True Range: {e}")
        return pd.Series(index=df.index)


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Apply Wilder's smoothing to a series.

    Wilder's Smoothing: ((prev_value * (n-1)) + current_value) / n

    Args:
        series: Input series
        period: Smoothing period

    Returns:
        Smoothed series
    """
    try:
        result = series.copy()

        # Initialize with SMA
        result.iloc[:period] = series.iloc[:period].mean()

        # Apply Wilder's smoothing
        for i in range(period, len(series)):
            result.iloc[i] = ((result.iloc[i-1] * (period - 1)) + series.iloc[i]) / period

        return result

    except Exception as e:
        logger.error(f"Error in Wilder smoothing: {e}")
        return series


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14,
    adx_smoothing: int = 14
) -> pd.DataFrame:
    """
    Calculate ADX, +DI, and -DI indicators.

    Steps:
    1. Calculate +DM and -DM
    2. Calculate True Range
    3. Smooth +DM, -DM, and TR using Wilder's method
    4. Calculate +DI and -DI
    5. Calculate DX (Directional Index)
    6. Smooth DX to get ADX

    Args:
        df: DataFrame with OHLC data
        period: Smoothing period for DI calculation
        adx_smoothing: Smoothing period for ADX

    Returns:
        DataFrame with 'plus_di', 'minus_di', 'adx' columns
    """
    try:
        df = df.copy()

        # Calculate Directional Movement
        df = calculate_directional_movement(df)

        # Calculate True Range
        df['tr'] = calculate_true_range(df)

        # Apply Wilder's smoothing
        df['smoothed_tr'] = wilder_smooth(df['tr'], period)
        df['smoothed_+dm'] = wilder_smooth(df['+dm'], period)
        df['smoothed_-dm'] = wilder_smooth(df['-dm'], period)

        # Calculate +DI and -DI (as percentages)
        df['plus_di'] = (df['smoothed_+dm'] / df['smoothed_tr']) * 100
        df['minus_di'] = (df['smoothed_-dm'] / df['smoothed_tr']) * 100

        # Handle division by zero
        df['plus_di'] = df['plus_di'].replace([np.inf, -np.inf], 0)
        df['minus_di'] = df['minus_di'].replace([np.inf, -np.inf], 0)

        # Calculate DX (Directional Index)
        di_sum = df['plus_di'] + df['minus_di']
        di_diff = abs(df['plus_di'] - df['minus_di'])
        df['dx'] = (di_diff / di_sum) * 100
        df['dx'] = df['dx'].replace([np.inf, -np.inf], 0)

        # Calculate ADX (smoothed DX)
        df['adx'] = wilder_smooth(df['dx'], adx_smoothing)

        # Clean up temporary columns
        columns_to_drop = ['+dm', '-dm', 'tr', 'smoothed_tr',
                          'smoothed_+dm', 'smoothed_-dm', 'dx']
        df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

        return df

    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        df['plus_di'] = 0
        df['minus_di'] = 0
        df['adx'] = 0
        return df


def get_adx_signal(
    df: pd.DataFrame,
    adx_threshold: float = 23,
    strong_trend_threshold: float = 40
) -> dict:
    """
    Get ADX-based trading signal.

    Signal Logic:
    - ADX > threshold + +DI > -DI = Bullish (CE)
    - ADX > threshold + -DI > +DI = Bearish (PE)
    - ADX < threshold = No trend

    Args:
        df: DataFrame with ADX calculated
        adx_threshold: Minimum ADX for trend confirmation
        strong_trend_threshold: ADX level for strong trend

    Returns:
        Dict with signal information
    """
    try:
        if 'adx' not in df.columns:
            df = calculate_adx(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        adx = latest['adx']
        plus_di = latest['plus_di']
        minus_di = latest['minus_di']

        prev_adx = prev['adx']
        adx_rising = adx > prev_adx

        # Determine trend strength
        if adx >= strong_trend_threshold:
            trend_strength = "STRONG"
        elif adx >= adx_threshold:
            trend_strength = "MODERATE"
        else:
            trend_strength = "WEAK"

        # Determine direction
        if plus_di > minus_di:
            direction = "BULLISH"
            signal = "CE" if adx >= adx_threshold and adx_rising else "HOLD"
        elif minus_di > plus_di:
            direction = "BEARISH"
            signal = "PE" if adx >= adx_threshold and adx_rising else "HOLD"
        else:
            direction = "NEUTRAL"
            signal = "HOLD"

        # DI crossover detection
        prev_plus_di = prev['plus_di']
        prev_minus_di = prev['minus_di']

        crossover = None
        if prev_plus_di <= prev_minus_di and plus_di > minus_di:
            crossover = "BULLISH_DI_CROSSOVER"
        elif prev_plus_di >= prev_minus_di and plus_di < minus_di:
            crossover = "BEARISH_DI_CROSSOVER"

        return {
            "signal": signal,
            "direction": direction,
            "trend_strength": trend_strength,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx_rising": adx_rising,
            "crossover": crossover
        }

    except Exception as e:
        logger.error(f"Error getting ADX signal: {e}")
        return {
            "signal": "HOLD",
            "direction": "NEUTRAL",
            "trend_strength": "WEAK"
        }


def is_trending_market(
    df: pd.DataFrame,
    threshold: float = 23
) -> bool:
    """
    Check if market is trending based on ADX.

    Args:
        df: DataFrame with ADX data
        threshold: ADX threshold for trend confirmation

    Returns:
        True if market is trending
    """
    try:
        if 'adx' not in df.columns:
            df = calculate_adx(df)

        return df['adx'].iloc[-1] >= threshold

    except Exception as e:
        logger.error(f"Error checking trending market: {e}")
        return False


if __name__ == "__main__":
    # Test ADX calculation
    print("ADX Indicator Module")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range(start='2026-01-01', periods=50, freq='D')
    np.random.seed(42)

    base_price = 100
    trend = np.linspace(0, 10, 50)  # Upward trend

    sample_data = {
        'open': base_price + trend + np.random.randn(50),
        'high': base_price + trend + np.abs(np.random.randn(50)) + 1,
        'low': base_price + trend - np.abs(np.random.randn(50)) - 1,
        'close': base_price + trend + np.random.randn(50) * 0.5,
        'volume': np.random.randint(1000, 10000, 50)
    }

    df = pd.DataFrame(sample_data, index=dates)

    # Calculate ADX
    df = calculate_adx(df, period=14)

    print("\nADX Calculation Results:")
    print(df[['close', 'plus_di', 'minus_di', 'adx']].tail(10))

    # Get signal
    signal = get_adx_signal(df)
    print(f"\nADX Signal: {signal}")

    # Check if trending
    is_trending = is_trending_market(df)
    print(f"Is Trending: {is_trending}")
