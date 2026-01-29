"""
CE (Call Option) Entry Logic Module

This module handles the entry logic for Call options (bullish trades).
It validates technical conditions, calculates position sizing, and executes orders.

Groww API Documentation: https://groww.in/trade-api/docs/python-sdk
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def check_ce_entry_conditions(
    df: pd.DataFrame,
    symbol: str,
    tsl: Any
) -> Dict[str, Any]:
    """
    Check if CE entry conditions are met.

    Entry Conditions for CE (Bullish):
    1. Price above VWAP
    2. +DI > -DI (bullish directional movement)
    3. ADX > 23 (trending market)
    4. ADX rising (momentum building)
    5. Price above upper Fractal Chaos Band (breakout confirmation)
    6. ATR conditions met

    Args:
        df: DataFrame with OHLCV and indicator data
        symbol: Stock symbol
        tsl: Trading instance

    Returns:
        Dict with 'valid' boolean and signal details
    """
    try:
        if len(df) < 50:
            return {"valid": False, "reason": "Insufficient data"}

        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Check required columns exist
        required_cols = ['close', 'vwap', 'adx', 'plus_di', 'minus_di', 'atr']
        for col in required_cols:
            if col not in df.columns:
                return {"valid": False, "reason": f"Missing indicator: {col}"}

        # Condition 1: Price above VWAP (bullish bias)
        if latest['close'] <= latest['vwap']:
            return {"valid": False, "reason": "Price below VWAP"}

        # Condition 2: +DI > -DI (bullish directional movement)
        if latest['plus_di'] <= latest['minus_di']:
            return {"valid": False, "reason": "Bearish DI crossover"}

        # Condition 3: ADX > 23 (trending market)
        adx_threshold = 23
        if latest['adx'] < adx_threshold:
            return {"valid": False, "reason": f"ADX below {adx_threshold}"}

        # Condition 4: ADX rising (momentum building)
        if latest['adx'] <= prev['adx']:
            return {"valid": False, "reason": "ADX not rising"}

        # Condition 5: Fractal Chaos Band confirmation (optional)
        if 'upper_band' in df.columns:
            if latest['close'] < latest['upper_band']:
                return {"valid": False, "reason": "Below upper band"}

        # All conditions met
        signal = {
            "valid": True,
            "symbol": symbol,
            "signal_type": "CE",
            "close": latest['close'],
            "vwap": latest['vwap'],
            "adx": latest['adx'],
            "plus_di": latest['plus_di'],
            "minus_di": latest['minus_di'],
            "atr": latest['atr'],
            "timestamp": datetime.now()
        }

        logger.info(f"CE Entry Signal: {symbol}")
        logger.info(f"  Close: {latest['close']:.2f}, VWAP: {latest['vwap']:.2f}")
        logger.info(f"  ADX: {latest['adx']:.2f}, +DI: {latest['plus_di']:.2f}, -DI: {latest['minus_di']:.2f}")

        return signal

    except Exception as e:
        logger.error(f"Error checking CE entry for {symbol}: {e}")
        return {"valid": False, "reason": str(e)}


def validate_bid_ask_spread(
    tsl: Any,
    option_symbol: str,
    max_spread_pct: float = 1.0
) -> Dict[str, Any]:
    """
    Validate bid-ask spread is within acceptable range.

    Args:
        tsl: Trading instance
        option_symbol: Option trading symbol
        max_spread_pct: Maximum acceptable spread percentage

    Returns:
        Dict with validation result and prices
    """
    try:
        quote = tsl.get_quote(option_symbol, tsl.NFO)

        if not quote:
            return {"valid": False, "reason": "Could not fetch quote"}

        bid = float(quote.get('bid_price', 0))
        ask = float(quote.get('ask_price', 0))
        ltp = float(quote.get('ltp', 0))

        if bid <= 0 or ask <= 0:
            return {"valid": False, "reason": "Invalid bid/ask prices"}

        spread = ask - bid
        spread_pct = (spread / ask) * 100

        if spread_pct > max_spread_pct:
            return {
                "valid": False,
                "reason": f"Spread too wide: {spread_pct:.2f}%",
                "bid": bid,
                "ask": ask,
                "spread_pct": spread_pct
            }

        return {
            "valid": True,
            "bid": bid,
            "ask": ask,
            "ltp": ltp,
            "spread_pct": spread_pct
        }

    except Exception as e:
        logger.error(f"Error validating spread: {e}")
        return {"valid": False, "reason": str(e)}


def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    risk_per_trade: float,
    lot_size: int
) -> int:
    """
    Calculate position size based on risk parameters.

    Args:
        entry_price: Expected entry price
        stop_loss: Stop loss price
        risk_per_trade: Maximum risk amount per trade
        lot_size: F&O lot size

    Returns:
        Number of lots to trade
    """
    try:
        risk_per_unit = entry_price - stop_loss

        if risk_per_unit <= 0:
            logger.warning("Invalid risk per unit, using 1 lot")
            return 1

        risk_per_lot = risk_per_unit * lot_size
        max_lots = int(risk_per_trade / risk_per_lot)

        # Minimum 1 lot, maximum 10 lots
        lots = max(1, min(max_lots, 10))

        logger.info(f"Position sizing: Risk={risk_per_trade:.2f}, "
                   f"Risk/Lot={risk_per_lot:.2f}, Lots={lots}")

        return lots

    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 1


def get_option_details(
    tsl: Any,
    symbol: str,
    spot_price: float,
    option_type: str = "CE"
) -> Dict[str, Any]:
    """
    Get ATM option details for trading.

    Args:
        tsl: Trading instance
        symbol: Underlying symbol
        spot_price: Current spot price
        option_type: "CE" or "PE"

    Returns:
        Dict with option details
    """
    try:
        # Determine strike interval based on symbol
        if "NIFTY" in symbol.upper():
            strike_interval = 50
        elif "BANK" in symbol.upper():
            strike_interval = 100
        else:
            strike_interval = 50  # Default

        # Get ATM strike
        atm_strike = tsl.get_atm_strike(symbol, spot_price, strike_interval)

        # Get nearest expiry
        expiry = tsl.get_nearest_expiry(symbol)

        if not expiry:
            return {"valid": False, "reason": "Could not get expiry"}

        # Generate option symbol
        option_symbol = tsl.get_option_symbol(symbol, expiry, atm_strike, option_type)

        # Get lot size
        lot_size = tsl.get_lot_size(symbol)

        # Get option LTP
        option_ltp = tsl.get_ltp(option_symbol, tsl.NFO)

        return {
            "valid": True,
            "option_symbol": option_symbol,
            "strike": atm_strike,
            "expiry": expiry,
            "lot_size": lot_size,
            "ltp": option_ltp
        }

    except Exception as e:
        logger.error(f"Error getting option details: {e}")
        return {"valid": False, "reason": str(e)}


def execute_ce_entry(
    tsl: Any,
    symbol: str,
    df: pd.DataFrame,
    signal: Dict,
    risk_per_trade: float,
    atr_multiplier: float = 3,
    option_sl_pct: float = 0.15
) -> Dict[str, Any]:
    """
    Execute CE (Call) entry trade.

    Args:
        tsl: Trading instance
        symbol: Underlying symbol
        df: DataFrame with indicator data
        signal: Entry signal from check_ce_entry_conditions
        risk_per_trade: Risk amount per trade
        atr_multiplier: ATR multiplier for stop loss
        option_sl_pct: Option stop loss percentage

    Returns:
        Dict with execution result
    """
    try:
        logger.info(f"Executing CE Entry for {symbol}")

        # Get spot price
        spot_price = signal['close']

        # Get option details
        option_info = get_option_details(tsl, symbol, spot_price, "CE")

        if not option_info["valid"]:
            return {
                "success": False,
                "message": option_info.get("reason", "Failed to get option details")
            }

        option_symbol = option_info["option_symbol"]
        entry_price = option_info["ltp"]
        lot_size = option_info["lot_size"]

        if entry_price <= 0:
            return {"success": False, "message": "Invalid option price"}

        # Validate bid-ask spread
        spread_check = validate_bid_ask_spread(tsl, option_symbol)
        if not spread_check["valid"]:
            return {"success": False, "message": spread_check["reason"]}

        # Calculate stop loss
        stop_loss = entry_price * (1 - option_sl_pct)

        # Calculate target (Risk:Reward ratio)
        risk = entry_price - stop_loss
        target = entry_price + (risk * 3)  # 1:3 risk-reward

        # Calculate position size
        lots = calculate_position_size(entry_price, stop_loss, risk_per_trade, lot_size)
        quantity = lots * lot_size

        logger.info(f"CE Order Details:")
        logger.info(f"  Symbol: {option_symbol}")
        logger.info(f"  Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, Target: {target:.2f}")
        logger.info(f"  Lots: {lots}, Quantity: {quantity}")

        # Place the order
        order_result = tsl.place_order(
            symbol=option_symbol,
            exchange=tsl.NFO,
            transaction_type=tsl.BUY,
            quantity=quantity,
            order_type=tsl.ORDER_TYPE_MARKET,
            product_type=tsl.PRODUCT_INTRADAY
        )

        if order_result["status"] == "success":
            # Get executed price
            order_id = order_result["order_id"]
            order_status = tsl.get_order_status(order_id)
            executed_price = order_status.get("average_price", entry_price)

            logger.info(f"CE Order Executed: {order_id} @ {executed_price}")

            return {
                "success": True,
                "order_id": order_id,
                "option_symbol": option_symbol,
                "entry_price": executed_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "target": target,
                "lots": lots
            }
        else:
            return {
                "success": False,
                "message": order_result.get("message", "Order placement failed")
            }

    except Exception as e:
        logger.error(f"CE entry execution error: {e}")
        return {"success": False, "message": str(e)}
