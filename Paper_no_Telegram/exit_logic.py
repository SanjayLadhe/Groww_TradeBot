"""
Exit Logic Module

This module handles the exit logic for all positions including:
- Stop loss monitoring
- Target hit detection
- Trailing stop loss updates
- Time-based exits
- Position closing

Groww API Documentation: https://groww.in/trade-api/docs/python-sdk
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Exit time configuration (IST)
SQUARE_OFF_HOUR = 15
SQUARE_OFF_MINUTE = 15


def check_exit_conditions(
    position: Dict[str, Any],
    current_price: float,
    option_type: str = "CE"
) -> Dict[str, Any]:
    """
    Check if exit conditions are met for a position.

    Exit Conditions:
    1. Stop loss hit
    2. Trailing stop loss hit
    3. Target hit
    4. Time-based exit (end of day)

    Args:
        position: Position dictionary with entry details
        current_price: Current market price
        option_type: "CE" or "PE"

    Returns:
        Dict with 'should_exit' boolean and reason
    """
    try:
        entry_price = position["entry_price"]
        stop_loss = position["stop_loss"]
        trailing_stop = position["trailing_stop"]
        target = position["target"]
        entry_time = position["entry_time"]

        # Check time-based exit
        now = datetime.now()
        square_off_time = now.replace(hour=SQUARE_OFF_HOUR, minute=SQUARE_OFF_MINUTE, second=0)

        if now >= square_off_time:
            return {
                "should_exit": True,
                "reason": "Time-based square off",
                "exit_type": "TIME"
            }

        # Check stop loss hit
        if current_price <= stop_loss:
            return {
                "should_exit": True,
                "reason": f"Stop loss hit @ {current_price:.2f}",
                "exit_type": "SL"
            }

        # Check trailing stop loss hit
        if current_price <= trailing_stop and trailing_stop > stop_loss:
            return {
                "should_exit": True,
                "reason": f"Trailing SL hit @ {current_price:.2f}",
                "exit_type": "TSL"
            }

        # Check target hit
        if current_price >= target:
            return {
                "should_exit": True,
                "reason": f"Target hit @ {current_price:.2f}",
                "exit_type": "TARGET"
            }

        # Calculate current P&L percentage
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Check for significant loss (circuit breaker)
        if pnl_pct < -30:
            return {
                "should_exit": True,
                "reason": f"Circuit breaker: {pnl_pct:.1f}% loss",
                "exit_type": "CIRCUIT"
            }

        # No exit condition met
        return {
            "should_exit": False,
            "reason": None,
            "current_pnl_pct": pnl_pct
        }

    except Exception as e:
        logger.error(f"Error checking exit conditions: {e}")
        return {"should_exit": False, "reason": None}


def update_trailing_stop(
    position: Dict[str, Any],
    current_price: float,
    atr_multiplier: float = 3,
    trail_trigger_pct: float = 0.02
) -> float:
    """
    Calculate and update trailing stop loss.

    The trailing stop is updated when:
    1. Price moves above entry by trigger percentage
    2. New TSL would be higher than current TSL

    Trailing Stop Logic:
    - Once in profit by trigger%, start trailing
    - TSL = Highest Price - (ATR * Multiplier)
    - TSL can only move up, never down

    Args:
        position: Position dictionary
        current_price: Current market price
        atr_multiplier: Multiplier for ATR-based stop
        trail_trigger_pct: Minimum profit % to start trailing

    Returns:
        New trailing stop price
    """
    try:
        entry_price = position["entry_price"]
        current_tsl = position["trailing_stop"]
        highest_price = position["highest_price"]
        original_sl = position["stop_loss"]

        # Update highest price
        new_highest = max(highest_price, current_price)
        position["highest_price"] = new_highest

        # Check if we're in enough profit to trail
        profit_pct = (current_price - entry_price) / entry_price

        if profit_pct < trail_trigger_pct:
            # Not enough profit to start trailing
            return current_tsl

        # Calculate new trailing stop
        # Use percentage-based trailing for options
        trail_distance_pct = 0.10  # 10% trailing distance

        new_tsl = new_highest * (1 - trail_distance_pct)

        # TSL can only move up
        if new_tsl > current_tsl:
            logger.info(f"TSL Update: {current_tsl:.2f} -> {new_tsl:.2f}")
            return new_tsl

        return current_tsl

    except Exception as e:
        logger.error(f"Error updating trailing stop: {e}")
        return position.get("trailing_stop", position.get("stop_loss", 0))


def calculate_dynamic_tsl(
    df: pd.DataFrame,
    entry_price: float,
    current_price: float,
    position_type: str = "long"
) -> float:
    """
    Calculate dynamic trailing stop based on ATR.

    Args:
        df: DataFrame with OHLCV and ATR data
        entry_price: Position entry price
        current_price: Current market price
        position_type: "long" or "short"

    Returns:
        Dynamic trailing stop price
    """
    try:
        if 'atr' not in df.columns or df.empty:
            # Fallback to percentage-based TSL
            return current_price * 0.90 if position_type == "long" else current_price * 1.10

        atr = df['atr'].iloc[-1]

        if position_type == "long":
            tsl = current_price - (atr * 2)
        else:
            tsl = current_price + (atr * 2)

        return tsl

    except Exception as e:
        logger.error(f"Error calculating dynamic TSL: {e}")
        return entry_price * 0.85


def execute_exit(
    tsl: Any,
    symbol: str,
    quantity: int,
    exit_price: float = None
) -> Dict[str, Any]:
    """
    Execute position exit order.

    Args:
        tsl: Trading instance
        symbol: Option symbol to exit
        quantity: Quantity to sell
        exit_price: Expected exit price (for logging)

    Returns:
        Dict with execution result
    """
    try:
        logger.info(f"Executing exit for {symbol}, Qty: {quantity}")

        # Place market sell order
        order_result = tsl.place_order(
            symbol=symbol,
            exchange=tsl.NFO,
            transaction_type=tsl.SELL,
            quantity=quantity,
            order_type=tsl.ORDER_TYPE_MARKET,
            product_type=tsl.PRODUCT_INTRADAY
        )

        if order_result["status"] == "success":
            order_id = order_result["order_id"]

            # Get executed price
            order_status = tsl.get_order_status(order_id)
            executed_price = order_status.get("average_price", exit_price or 0)

            logger.info(f"Exit executed: {order_id} @ {executed_price}")

            return {
                "success": True,
                "order_id": order_id,
                "executed_price": executed_price
            }
        else:
            logger.error(f"Exit order failed: {order_result.get('message')}")
            return {
                "success": False,
                "message": order_result.get("message", "Exit failed")
            }

    except Exception as e:
        logger.error(f"Exit execution error: {e}")
        return {"success": False, "message": str(e)}


def check_sl_order_status(
    tsl: Any,
    sl_order_id: str
) -> Dict[str, Any]:
    """
    Check if SL order has been triggered/executed.

    Args:
        tsl: Trading instance
        sl_order_id: Stop loss order ID

    Returns:
        Dict with SL status
    """
    try:
        order_status = tsl.get_order_status(sl_order_id)

        status = order_status.get("order_status", "")
        executed_qty = order_status.get("filled_quantity", 0)
        avg_price = order_status.get("average_price", 0)

        if status == tsl.STATUS_COMPLETE:
            return {
                "triggered": True,
                "executed_price": avg_price,
                "filled_quantity": executed_qty
            }
        elif status == tsl.STATUS_CANCELLED:
            return {
                "triggered": False,
                "cancelled": True,
                "reason": "SL order cancelled"
            }
        elif status == tsl.STATUS_REJECTED:
            return {
                "triggered": False,
                "rejected": True,
                "reason": order_status.get("rejection_reason", "Unknown")
            }
        else:
            return {
                "triggered": False,
                "status": status,
                "pending": status in [tsl.STATUS_PENDING, tsl.STATUS_TRIGGER_PENDING]
            }

    except Exception as e:
        logger.error(f"Error checking SL order status: {e}")
        return {"triggered": False, "error": str(e)}


def place_sl_order(
    tsl: Any,
    symbol: str,
    quantity: int,
    trigger_price: float,
    limit_price: float = None
) -> Dict[str, Any]:
    """
    Place a stop loss order.

    Args:
        tsl: Trading instance
        symbol: Option symbol
        quantity: Quantity
        trigger_price: SL trigger price
        limit_price: Limit price (optional, for SL-L orders)

    Returns:
        Dict with order result
    """
    try:
        order_type = tsl.ORDER_TYPE_SL_MARKET
        price = 0

        if limit_price:
            order_type = tsl.ORDER_TYPE_SL
            price = limit_price

        order_result = tsl.place_order(
            symbol=symbol,
            exchange=tsl.NFO,
            transaction_type=tsl.SELL,
            quantity=quantity,
            order_type=order_type,
            trigger_price=trigger_price,
            price=price,
            product_type=tsl.PRODUCT_INTRADAY
        )

        if order_result["status"] == "success":
            logger.info(f"SL order placed: {order_result['order_id']} @ {trigger_price}")

        return order_result

    except Exception as e:
        logger.error(f"Error placing SL order: {e}")
        return {"status": "error", "message": str(e)}


def modify_sl_order(
    tsl: Any,
    order_id: str,
    new_trigger_price: float,
    new_limit_price: float = None
) -> Dict[str, Any]:
    """
    Modify an existing stop loss order.

    Args:
        tsl: Trading instance
        order_id: SL order ID to modify
        new_trigger_price: New trigger price
        new_limit_price: New limit price (optional)

    Returns:
        Dict with modification result
    """
    try:
        modify_params = {
            "trigger_price": new_trigger_price
        }

        if new_limit_price:
            modify_params["price"] = new_limit_price

        result = tsl.modify_order(order_id, **modify_params)

        if result["status"] == "success":
            logger.info(f"SL order modified: {order_id} -> {new_trigger_price}")

        return result

    except Exception as e:
        logger.error(f"Error modifying SL order: {e}")
        return {"status": "error", "message": str(e)}


def cancel_sl_order(tsl: Any, order_id: str) -> Dict[str, Any]:
    """
    Cancel an existing stop loss order.

    Args:
        tsl: Trading instance
        order_id: SL order ID to cancel

    Returns:
        Dict with cancellation result
    """
    try:
        result = tsl.cancel_order(order_id)

        if result["status"] == "success":
            logger.info(f"SL order cancelled: {order_id}")

        return result

    except Exception as e:
        logger.error(f"Error cancelling SL order: {e}")
        return {"status": "error", "message": str(e)}


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: int,
    transaction_type: str = "BUY"
) -> Dict[str, float]:
    """
    Calculate profit/loss for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Traded quantity
        transaction_type: "BUY" or "SELL"

    Returns:
        Dict with P&L details
    """
    try:
        if transaction_type == "BUY":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        return {
            "pnl": pnl,
            "pnl_percentage": pnl_pct,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity
        }

    except Exception as e:
        logger.error(f"Error calculating P&L: {e}")
        return {"pnl": 0, "pnl_percentage": 0}
