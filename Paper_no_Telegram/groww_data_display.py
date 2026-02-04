"""
Groww Data Display Script

Fetches historical data from Groww and displays all technical indicators
used in the trading bot: VWAP, ADX, ATR Trailing Stop, and Fractal Chaos Bands.

Similar to the Dhan data display script but adapted for Groww API.

Usage:
    python groww_data_display.py

Author: Algo Trading Bot
Date: 2026
"""

import pdb
import pandas as pd
import numpy as np
import time
import traceback
from datetime import datetime, timedelta

# Groww API Wrapper
from Groww_Tradehull import Tradehull

# Technical Indicators
from VWAP import calculate_vwap, calculate_vwap_bands, get_vwap_signal
from ATRTrailingStop import (
    calculate_atr,
    calculate_atr_trailing_stop,
    calculate_chandelier_exit,
    get_atr_signal,
    calculate_volatility_ratio
)
from adx_indicator import calculate_adx, get_adx_signal, is_trending_market
from Fractal_Chaos_Bands import (
    calculate_fractal_chaos_bands,
    get_fractal_signal,
    get_support_resistance_levels
)

# Configure pandas display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Groww API Credentials (Replace with your actual credentials)
API_KEY = "YOUR_GROWW_API_KEY"
API_SECRET = "YOUR_GROWW_API_SECRET"

# Watchlist - symbols to fetch and display
watchlist = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

# Timeframe for candle data
timeframe = "5m"

# Number of candles to fetch
num_candles = 100

# ATR parameters
ATR_PERIOD = 14
ATR_MULTIPLIER = 3.0

# ADX parameters
ADX_PERIOD = 14
ADX_SMOOTHING = 14

# Fractal parameters
FRACTAL_WINDOW = 2
FRACTAL_SMOOTHING = 5

# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================


def print_separator(char="=", length=100):
    """Print a separator line."""
    print(char * length)


def print_header(title):
    """Print a section header."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")


def display_raw_ohlcv(df, symbol):
    """Display raw OHLCV data."""
    print_header(f"RAW OHLCV DATA - {symbol}")
    print(f"\nTotal candles: {len(df)}")
    print(f"Timeframe: {timeframe}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nFirst 5 candles:")
    print(df[['open', 'high', 'low', 'close', 'volume']].head(5))
    print(f"\nLast 5 candles:")
    print(df[['open', 'high', 'low', 'close', 'volume']].tail(5))


def display_vwap(df, symbol):
    """Display VWAP indicator values."""
    print_header(f"VWAP (Volume Weighted Average Price) - {symbol}")

    df_vwap = calculate_vwap(df.copy())

    print(f"\nLast 10 candles with VWAP:")
    print(df_vwap[['close', 'volume', 'vwap']].tail(10))

    # VWAP with bands
    df_bands = calculate_vwap_bands(df.copy())
    print(f"\nLast 10 candles with VWAP Bands:")
    vwap_cols = ['close', 'vwap']
    if 'vwap_upper' in df_bands.columns:
        vwap_cols.extend(['vwap_upper', 'vwap_lower'])
    print(df_bands[vwap_cols].tail(10))

    # VWAP signal
    signal = get_vwap_signal(df_vwap)
    print(f"\nVWAP Signal:")
    print(f"  Bias: {signal.get('bias', 'N/A')}")
    print(f"  Crossover: {signal.get('crossover', 'None')}")
    print(f"  Close: {signal.get('close', 'N/A'):.2f}")
    print(f"  VWAP: {signal.get('vwap', 'N/A'):.2f}")
    print(f"  Distance from VWAP: {signal.get('distance_pct', 0):.2f}%")

    return df_vwap


def display_adx(df, symbol):
    """Display ADX indicator values."""
    print_header(f"ADX (Average Directional Index) - {symbol}")

    df_adx = calculate_adx(df.copy(), period=ADX_PERIOD, adx_smoothing=ADX_SMOOTHING)

    print(f"\nLast 10 candles with ADX:")
    print(df_adx[['close', 'plus_di', 'minus_di', 'adx']].tail(10))

    # ADX signal
    signal = get_adx_signal(df_adx)
    print(f"\nADX Signal:")
    print(f"  Signal: {signal.get('signal', 'N/A')}")
    print(f"  Direction: {signal.get('direction', 'N/A')}")
    print(f"  Trend Strength: {signal.get('trend_strength', 'N/A')}")
    print(f"  ADX Value: {signal.get('adx', 0):.2f}")
    print(f"  +DI: {signal.get('plus_di', 0):.2f}")
    print(f"  -DI: {signal.get('minus_di', 0):.2f}")
    print(f"  ADX Rising: {signal.get('adx_rising', 'N/A')}")
    print(f"  DI Crossover: {signal.get('crossover', 'None')}")

    # Trending check
    trending = is_trending_market(df_adx)
    print(f"  Is Trending Market: {trending}")

    return df_adx


def display_atr(df, symbol):
    """Display ATR and Trailing Stop values."""
    print_header(f"ATR (Average True Range) & Trailing Stop - {symbol}")

    # ATR
    df_atr = calculate_atr(df.copy(), period=ATR_PERIOD)
    print(f"\nLast 10 candles with ATR (period={ATR_PERIOD}):")
    print(df_atr[['close', 'atr']].tail(10))

    # ATR Trailing Stop
    df_atr_ts = calculate_atr_trailing_stop(df_atr.copy(), multiplier=ATR_MULTIPLIER, period=ATR_PERIOD)
    print(f"\nLast 10 candles with ATR Trailing Stop (multiplier={ATR_MULTIPLIER}):")
    print(df_atr_ts[['close', 'atr', 'atr_trailing_long', 'atr_trailing_short']].tail(10))

    # Chandelier Exit
    df_chandelier = calculate_chandelier_exit(df.copy(), period=22, multiplier=ATR_MULTIPLIER)
    print(f"\nLast 10 candles with Chandelier Exit:")
    chandelier_cols = ['close']
    if 'chandelier_long' in df_chandelier.columns:
        chandelier_cols.extend(['chandelier_long', 'chandelier_short'])
    if 'atr' in df_chandelier.columns:
        chandelier_cols.append('atr')
    print(df_chandelier[chandelier_cols].tail(10))

    # Volatility Ratio
    df_vol = calculate_volatility_ratio(df.copy(), short_period=5, long_period=20)
    print(f"\nLast 10 candles with Volatility Ratio (5/20):")
    vol_cols = ['close']
    if 'volatility_ratio' in df_vol.columns:
        vol_cols.append('volatility_ratio')
    print(df_vol[vol_cols].tail(10))

    # ATR signal
    signal = get_atr_signal(df_atr_ts, multiplier=ATR_MULTIPLIER)
    print(f"\nATR Signal:")
    print(f"  Trend: {signal.get('trend', 'N/A')}")
    print(f"  Signal: {signal.get('signal', 'N/A')}")
    print(f"  Close: {signal.get('close', 0):.2f}")
    print(f"  ATR: {signal.get('atr', 0):.2f}")
    print(f"  Long Stop: {signal.get('long_stop', 0):.2f}")
    print(f"  Short Stop: {signal.get('short_stop', 0):.2f}")
    print(f"  Risk %: {signal.get('risk_pct', 0):.2f}%")

    return df_atr_ts


def display_fractal_chaos_bands(df, symbol):
    """Display Fractal Chaos Bands values."""
    print_header(f"FRACTAL CHAOS BANDS - {symbol}")

    df_fcb = calculate_fractal_chaos_bands(
        df.copy(), window=FRACTAL_WINDOW, smoothing=FRACTAL_SMOOTHING
    )

    print(f"\nLast 10 candles with Fractal Chaos Bands:")
    fcb_cols = ['close', 'lower_band', 'middle_band', 'upper_band']
    if 'fractal_high' in df_fcb.columns:
        fcb_cols.extend(['fractal_high', 'fractal_low'])
    print(df_fcb[fcb_cols].tail(10))

    # Fractal signal
    signal = get_fractal_signal(df_fcb)
    print(f"\nFractal Signal:")
    print(f"  Signal: {signal.get('signal', 'N/A')}")
    print(f"  Position: {signal.get('position', 'N/A')}")
    print(f"  Close: {signal.get('close', 0):.2f}")
    print(f"  Upper Band: {signal.get('upper_band', 0):.2f}")
    print(f"  Lower Band: {signal.get('lower_band', 0):.2f}")
    print(f"  Middle Band: {signal.get('middle_band', 0):.2f}")
    print(f"  Breakout: {signal.get('breakout', 'None')}")
    print(f"  Band Width %: {signal.get('band_width_pct', 0):.2f}%")
    print(f"  Is Squeeze: {signal.get('is_squeeze', 'N/A')}")

    # Support/Resistance levels
    sr = get_support_resistance_levels(df_fcb)
    print(f"\nSupport/Resistance Levels:")
    print(f"  Current Price: {sr.get('current_price', 'N/A')}")
    print(f"  Nearest Resistance: {sr.get('nearest_resistance', 'N/A')}")
    print(f"  Nearest Support: {sr.get('nearest_support', 'N/A')}")
    print(f"  Resistance Levels: {sr.get('resistance_levels', [])}")
    print(f"  Support Levels: {sr.get('support_levels', [])}")

    return df_fcb


def display_combined_indicators(df, symbol):
    """Display all indicators combined in a single DataFrame."""
    print_header(f"ALL INDICATORS COMBINED - {symbol}")

    # Apply all indicators
    df_all = df.copy()
    df_all = calculate_vwap(df_all)
    df_all = calculate_atr(df_all, period=ATR_PERIOD)
    df_all = calculate_atr_trailing_stop(df_all, multiplier=ATR_MULTIPLIER, period=ATR_PERIOD)
    df_all = calculate_adx(df_all, period=ADX_PERIOD, adx_smoothing=ADX_SMOOTHING)
    df_all = calculate_fractal_chaos_bands(df_all, window=FRACTAL_WINDOW, smoothing=FRACTAL_SMOOTHING)

    # Display key columns
    print(f"\nAligned Data Check (Last 10 candles):")
    print_separator()
    key_cols = ['close', 'vwap', 'atr', 'atr_trailing_long', 'atr_trailing_short',
                'plus_di', 'minus_di', 'adx', 'upper_band', 'lower_band', 'middle_band']
    available_cols = [c for c in key_cols if c in df_all.columns]
    print(df_all[available_cols].tail(10))

    # NaN check
    print(f"\nNaN Check:")
    print_separator()
    print(df_all[available_cols].isna().sum())

    # Summary of latest values
    latest = df_all.iloc[-1]
    print(f"\nLatest Values Summary:")
    print_separator()
    print(f"  Close:              {latest['close']:.2f}")
    if 'vwap' in df_all.columns:
        print(f"  VWAP:               {latest['vwap']:.2f}")
        vwap_dist = ((latest['close'] - latest['vwap']) / latest['vwap']) * 100
        print(f"  VWAP Distance:      {vwap_dist:.2f}%")
    if 'atr' in df_all.columns:
        print(f"  ATR ({ATR_PERIOD}):           {latest['atr']:.2f}")
    if 'atr_trailing_long' in df_all.columns:
        print(f"  ATR Trail Long:     {latest['atr_trailing_long']:.2f}")
    if 'atr_trailing_short' in df_all.columns:
        print(f"  ATR Trail Short:    {latest['atr_trailing_short']:.2f}")
    if 'adx' in df_all.columns:
        print(f"  ADX:                {latest['adx']:.2f}")
    if 'plus_di' in df_all.columns:
        print(f"  +DI:                {latest['plus_di']:.2f}")
    if 'minus_di' in df_all.columns:
        print(f"  -DI:                {latest['minus_di']:.2f}")
    if 'upper_band' in df_all.columns:
        print(f"  Upper Band:         {latest['upper_band']:.2f}")
    if 'lower_band' in df_all.columns:
        print(f"  Lower Band:         {latest['lower_band']:.2f}")
    if 'middle_band' in df_all.columns:
        print(f"  Middle Band:        {latest['middle_band']:.2f}")

    return df_all


def display_all_signals(df, symbol):
    """Display all trading signals from all indicators."""
    print_header(f"ALL TRADING SIGNALS - {symbol}")

    # Prepare DataFrames with indicators
    df_vwap = calculate_vwap(df.copy())
    df_atr = calculate_atr(df.copy(), period=ATR_PERIOD)
    df_atr = calculate_atr_trailing_stop(df_atr, multiplier=ATR_MULTIPLIER)
    df_adx = calculate_adx(df.copy(), period=ADX_PERIOD)
    df_fcb = calculate_fractal_chaos_bands(df.copy())

    # Get all signals
    vwap_signal = get_vwap_signal(df_vwap)
    atr_signal = get_atr_signal(df_atr, multiplier=ATR_MULTIPLIER)
    adx_signal = get_adx_signal(df_adx)
    fcb_signal = get_fractal_signal(df_fcb)

    print(f"\n  VWAP Signal:    Bias={vwap_signal.get('bias', 'N/A'):12s}  Crossover={vwap_signal.get('crossover', 'None')}")
    print(f"  ATR Signal:     Trend={atr_signal.get('trend', 'N/A'):11s}  Signal={atr_signal.get('signal', 'N/A')}")
    print(f"  ADX Signal:     Dir={adx_signal.get('direction', 'N/A'):13s}  Strength={adx_signal.get('trend_strength', 'N/A')}  Signal={adx_signal.get('signal', 'N/A')}")
    print(f"  Fractal Signal: Signal={fcb_signal.get('signal', 'N/A'):10s}  Position={fcb_signal.get('position', 'N/A')}  Breakout={fcb_signal.get('breakout', 'None')}")

    # Overall assessment
    print(f"\n  Overall Assessment:")
    print_separator("-", 60)

    bullish_count = 0
    bearish_count = 0

    if vwap_signal.get('bias') == 'BULLISH':
        bullish_count += 1
    elif vwap_signal.get('bias') == 'BEARISH':
        bearish_count += 1

    if atr_signal.get('signal') == 'BUY':
        bullish_count += 1
    elif atr_signal.get('signal') == 'SELL':
        bearish_count += 1

    if adx_signal.get('direction') == 'BULLISH':
        bullish_count += 1
    elif adx_signal.get('direction') == 'BEARISH':
        bearish_count += 1

    if fcb_signal.get('signal') in ['STRONG_BULLISH', 'BULLISH']:
        bullish_count += 1
    elif fcb_signal.get('signal') in ['STRONG_BEARISH', 'BEARISH']:
        bearish_count += 1

    print(f"  Bullish Indicators: {bullish_count}/4")
    print(f"  Bearish Indicators: {bearish_count}/4")

    if bullish_count >= 3:
        print(f"  Consensus: STRONG BULLISH (CE)")
    elif bullish_count >= 2:
        print(f"  Consensus: LEAN BULLISH")
    elif bearish_count >= 3:
        print(f"  Consensus: STRONG BEARISH (PE)")
    elif bearish_count >= 2:
        print(f"  Consensus: LEAN BEARISH")
    else:
        print(f"  Consensus: MIXED / NO CLEAR SIGNAL")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to fetch data and display all indicators."""

    print("\n" + "=" * 100)
    print("  GROWW DATA DISPLAY - Technical Indicator Viewer")
    print("=" * 100)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles: {num_candles}")
    print(f"Watchlist: {watchlist}")

    # Initialize Groww API client
    print("\n-----Connecting to Groww-----")
    try:
        tsl = Tradehull(API_KEY, API_SECRET)
        print("-----Connected to Groww-----\n")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Groww: {e}")
        print("Please check your API_KEY and API_SECRET in the configuration section.")
        return

    while True:
        for stock_name in watchlist:
            try:
                print_header(f"FETCHING DATA FOR: {stock_name}")
                print(f"  Exchange: NSE")
                print(f"  Timeframe: {timeframe}")
                print(f"  Candles requested: {num_candles}")

                # Fetch historical candle data
                df = tsl.get_candles(
                    symbol=stock_name,
                    exchange=tsl.NSE,
                    interval=timeframe,
                    num_candles=num_candles
                )

                if df is None or df.empty:
                    print(f"[ERROR] Failed to fetch data for {stock_name}. Returned empty.")
                    print(f"Skipping {stock_name}...")
                    continue

                if len(df) < 50:
                    print(f"[WARNING] Insufficient data for {stock_name} ({len(df)} candles). Need at least 50.")
                    continue

                print(f"  Fetched {len(df)} candles successfully.")

                # Display each indicator separately
                display_raw_ohlcv(df, stock_name)
                display_vwap(df, stock_name)
                display_adx(df, stock_name)
                display_atr(df, stock_name)
                display_fractal_chaos_bands(df, stock_name)

                # Display all indicators combined
                df_combined = display_combined_indicators(df, stock_name)

                # Display all signals
                display_all_signals(df, stock_name)

                # Debug breakpoint (uncomment to inspect data interactively)
                # pdb.set_trace()

            except Exception as e:
                print(f"\n[ERROR] Error processing {stock_name}: {e}")
                traceback.print_exc()
                continue

            print(f"\n{'#' * 100}")
            print(f"  END OF {stock_name}")
            print(f"{'#' * 100}\n")

        # Wait before next refresh cycle
        print(f"\n{'=' * 100}")
        print(f"  Cycle complete. Waiting 60 seconds before next refresh...")
        print(f"  Press Ctrl+C to stop.")
        print(f"{'=' * 100}\n")

        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break


if __name__ == "__main__":
    main()
