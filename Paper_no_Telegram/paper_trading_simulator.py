"""
Paper Trading Simulator Module

This module provides a simulated trading environment that mimics the real
Groww API without placing actual orders.

Features:
- Simulated order execution with realistic slippage
- Virtual balance and P&L tracking
- Trade logging
- Position management
- Always returns valid executed prices (never "None")

Usage:
    from paper_trading_simulator import PaperTradingSimulator
    simulator = PaperTradingSimulator(api_key)
    order = simulator.place_order(...)

Author: Algo Trading Bot
"""

import logging
import time
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import threading

from paper_trading_config import (
    PAPER_TRADING_BALANCE,
    PAPER_TRADING_MARGIN,
    SLIPPAGE_PERCENTAGE,
    SLIPPAGE_POINTS,
    ORDER_EXECUTION_DELAY,
    ORDER_EXECUTION_PROBABILITY,
    SIMULATE_ORDER_FAILURES,
    FAILURE_RATE,
    PAPER_TRADING_LOG_FILE,
    DETAILED_LOGGING
)

logger = logging.getLogger(__name__)


class PaperTradingSimulator:
    """
    Simulated trading environment for paper trading.

    This class provides the same interface as Groww_Tradehull.Tradehull
    but executes all trades in simulation mode.
    """

    # Exchange Constants (mirror Tradehull)
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    NFO = "NFO"
    BFO = "BFO"

    # Segment Constants
    SEGMENT_CASH = "CASH"
    SEGMENT_FNO = "FNO"

    # Order Type Constants
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_SL = "SL"
    ORDER_TYPE_SL_MARKET = "SL-M"

    # Transaction Type Constants
    BUY = "BUY"
    SELL = "SELL"

    # Product Type Constants
    PRODUCT_CNC = "CNC"
    PRODUCT_INTRADAY = "INTRADAY"
    PRODUCT_MARGIN = "MARGIN"

    # Order Status Constants
    STATUS_PENDING = "PENDING"
    STATUS_OPEN = "OPEN"
    STATUS_COMPLETE = "COMPLETE"
    STATUS_CANCELLED = "CANCELLED"
    STATUS_REJECTED = "REJECTED"

    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize the paper trading simulator.

        Args:
            api_key: Groww API key (stored but not used)
            api_secret: Groww API secret (stored but not used)
        """
        self.api_key = api_key
        self.api_secret = api_secret

        # Account state
        self._balance = PAPER_TRADING_BALANCE
        self._margin = PAPER_TRADING_MARGIN
        self._initial_balance = PAPER_TRADING_BALANCE

        # Order tracking
        self._orders: Dict[str, Dict] = {}
        self._trades: List[Dict] = []
        self._positions: Dict[str, Dict] = {}
        self._holdings: Dict[str, Dict] = {}

        # Statistics
        self._total_pnl = 0.0
        self._winning_trades = 0
        self._losing_trades = 0
        self._order_counter = 0

        # Simulated price cache
        self._price_cache: Dict[str, float] = {}

        # Threading lock
        self._lock = threading.Lock()

        # Initialize log file
        self._init_log_file()

        logger.info("Paper Trading Simulator initialized")
        logger.info(f"Starting Balance: {self._balance:,.2f}")

    def _init_log_file(self):
        """Initialize paper trading log file."""
        try:
            with open(PAPER_TRADING_LOG_FILE, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Paper Trading Session Started: {datetime.now()}\n")
                f.write(f"Starting Balance: {self._balance:,.2f}\n")
                f.write(f"{'='*60}\n\n")
        except Exception as e:
            logger.error(f"Error initializing log file: {e}")

    def _log_activity(self, activity: str):
        """Log trading activity to file."""
        try:
            with open(PAPER_TRADING_LOG_FILE, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {activity}\n")
        except Exception as e:
            logger.error(f"Error logging activity: {e}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"PAPER_{self._order_counter}_{uuid.uuid4().hex[:8].upper()}"

    def _simulate_slippage(self, price: float, transaction_type: str) -> float:
        """
        Simulate price slippage.

        Args:
            price: Base price
            transaction_type: BUY or SELL

        Returns:
            Price with slippage applied
        """
        # Percentage slippage
        pct_slip = price * (SLIPPAGE_PERCENTAGE / 100)

        # Point slippage
        point_slip = SLIPPAGE_POINTS

        # Total slippage
        total_slip = pct_slip + point_slip

        # Apply direction
        if transaction_type == self.BUY:
            return round(price + total_slip, 2)
        else:
            return round(price - total_slip, 2)

    def _simulate_price(self, symbol: str, base_price: float = None) -> float:
        """
        Get simulated price for a symbol.

        Args:
            symbol: Trading symbol
            base_price: Base price to vary from

        Returns:
            Simulated current price
        """
        if base_price:
            # Small random variation
            variation = random.uniform(-0.5, 0.5) / 100
            return round(base_price * (1 + variation), 2)

        if symbol in self._price_cache:
            # Vary from cached price
            cached = self._price_cache[symbol]
            variation = random.uniform(-0.5, 0.5) / 100
            new_price = round(cached * (1 + variation), 2)
            self._price_cache[symbol] = new_price
            return new_price

        # Generate new price
        new_price = round(100 + random.random() * 200, 2)
        self._price_cache[symbol] = new_price
        return new_price

    def place_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str = ORDER_TYPE_MARKET,
        price: float = 0,
        trigger_price: float = 0,
        product_type: str = PRODUCT_INTRADAY,
        validity: str = "DAY",
        disclosed_quantity: int = 0,
        tag: str = ""
    ) -> Dict[str, Any]:
        """
        Simulate order placement.

        Args:
            symbol: Trading symbol
            exchange: Exchange
            transaction_type: BUY or SELL
            quantity: Order quantity
            order_type: Order type
            price: Limit price
            trigger_price: Trigger price for SL orders
            product_type: Product type
            validity: Order validity
            disclosed_quantity: Disclosed quantity
            tag: Order tag

        Returns:
            Dict with order_id and status
        """
        with self._lock:
            try:
                # Simulate execution delay
                time.sleep(ORDER_EXECUTION_DELAY)

                # Check for simulated failures
                if SIMULATE_ORDER_FAILURES:
                    if random.random() < FAILURE_RATE:
                        return {
                            "status": "error",
                            "order_id": None,
                            "message": "Simulated order rejection"
                        }

                # Check order execution probability
                if random.random() > ORDER_EXECUTION_PROBABILITY:
                    return {
                        "status": "error",
                        "order_id": None,
                        "message": "Order not executed"
                    }

                # Get execution price
                if order_type == self.ORDER_TYPE_MARKET:
                    if price > 0:
                        base_price = price
                    else:
                        base_price = self._simulate_price(symbol)
                    exec_price = self._simulate_slippage(base_price, transaction_type)
                else:
                    exec_price = price if price > 0 else self._simulate_price(symbol)

                # Generate order ID
                order_id = self._generate_order_id()

                # Calculate order value
                order_value = exec_price * quantity

                # Check margin for buy orders
                if transaction_type == self.BUY:
                    if order_value > self._margin:
                        return {
                            "status": "error",
                            "order_id": None,
                            "message": "Insufficient margin"
                        }
                    self._margin -= order_value

                # Create order record
                order = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "exchange": exchange,
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "order_type": order_type,
                    "price": price,
                    "trigger_price": trigger_price,
                    "product_type": product_type,
                    "status": self.STATUS_COMPLETE,
                    "filled_quantity": quantity,
                    "average_price": exec_price,
                    "order_time": datetime.now(),
                    "tag": tag
                }

                self._orders[order_id] = order

                # Update positions
                self._update_position(symbol, transaction_type, quantity, exec_price)

                # Log activity
                activity = (f"ORDER {transaction_type}: {symbol} x {quantity} @ "
                           f"{exec_price:.2f} ({order_type}) - {order_id}")
                self._log_activity(activity)

                if DETAILED_LOGGING:
                    logger.info(f"Paper Trade: {activity}")

                return {
                    "status": "success",
                    "order_id": order_id,
                    "message": "Order placed successfully"
                }

            except Exception as e:
                logger.error(f"Paper order error: {e}")
                return {
                    "status": "error",
                    "order_id": None,
                    "message": str(e)
                }

    def _update_position(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        price: float
    ):
        """Update position after trade."""
        if symbol not in self._positions:
            self._positions[symbol] = {
                "symbol": symbol,
                "quantity": 0,
                "average_price": 0,
                "pnl": 0
            }

        pos = self._positions[symbol]

        if transaction_type == self.BUY:
            # Opening/adding to long position
            total_cost = (pos["quantity"] * pos["average_price"]) + (quantity * price)
            new_qty = pos["quantity"] + quantity
            pos["quantity"] = new_qty
            pos["average_price"] = total_cost / new_qty if new_qty > 0 else 0

        else:  # SELL
            # Closing/reducing position
            if pos["quantity"] > 0:
                pnl = (price - pos["average_price"]) * min(quantity, pos["quantity"])
                self._total_pnl += pnl
                self._balance += pnl

                if pnl >= 0:
                    self._winning_trades += 1
                else:
                    self._losing_trades += 1

                pos["pnl"] += pnl
                pos["quantity"] -= quantity

                # Release margin
                self._margin += price * quantity

    def modify_order(
        self,
        order_id: str,
        quantity: int = None,
        price: float = None,
        trigger_price: float = None,
        order_type: str = None
    ) -> Dict[str, Any]:
        """Simulate order modification."""
        with self._lock:
            if order_id not in self._orders:
                return {"status": "error", "message": "Order not found"}

            order = self._orders[order_id]

            if order["status"] != self.STATUS_PENDING:
                return {"status": "error", "message": "Cannot modify completed order"}

            if quantity:
                order["quantity"] = quantity
            if price:
                order["price"] = price
            if trigger_price:
                order["trigger_price"] = trigger_price
            if order_type:
                order["order_type"] = order_type

            self._log_activity(f"ORDER MODIFIED: {order_id}")

            return {"status": "success", "order_id": order_id}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Simulate order cancellation."""
        with self._lock:
            if order_id not in self._orders:
                return {"status": "error", "message": "Order not found"}

            order = self._orders[order_id]

            if order["status"] == self.STATUS_COMPLETE:
                return {"status": "error", "message": "Cannot cancel completed order"}

            order["status"] = self.STATUS_CANCELLED
            self._log_activity(f"ORDER CANCELLED: {order_id}")

            return {"status": "success", "order_id": order_id}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        with self._lock:
            if order_id not in self._orders:
                return {"status": "error", "message": "Order not found"}

            order = self._orders[order_id]

            return {
                "status": "success",
                "order_id": order_id,
                "order_status": order["status"],
                "filled_quantity": order.get("filled_quantity", 0),
                "average_price": order.get("average_price", 0),
                "details": order
            }

    def get_order_list(self, timeout: int = 5) -> List[Dict]:
        """Get list of all orders."""
        return list(self._orders.values())

    def get_trade_list(self) -> List[Dict]:
        """Get list of trades."""
        return self._trades

    def get_positions(self) -> Dict[str, List[Dict]]:
        """Get open positions."""
        positions = [p for p in self._positions.values() if p["quantity"] != 0]
        return {"day": positions, "net": positions}

    def get_holdings(self) -> List[Dict]:
        """Get holdings."""
        return list(self._holdings.values())

    def get_funds(self) -> Dict[str, Any]:
        """Get funds/balance information."""
        return {
            "available_balance": self._balance,
            "available_margin": self._margin,
            "used_margin": PAPER_TRADING_MARGIN - self._margin,
            "total_pnl": self._total_pnl
        }

    def get_available_balance(self) -> float:
        """Get available balance."""
        return self._balance

    def get_ltp(self, symbol: str, exchange: str = NSE, segment: str = SEGMENT_CASH) -> float:
        """Get simulated LTP."""
        return self._simulate_price(symbol)

    def get_quote(self, symbol: str, exchange: str = NSE) -> Dict[str, Any]:
        """Get simulated quote."""
        ltp = self._simulate_price(symbol)
        spread = ltp * 0.001

        return {
            "ltp": ltp,
            "bid_price": round(ltp - spread, 2),
            "ask_price": round(ltp + spread, 2),
            "volume": random.randint(10000, 100000),
            "open": round(ltp * 0.99, 2),
            "high": round(ltp * 1.02, 2),
            "low": round(ltp * 0.98, 2),
            "close": ltp
        }

    def get_candles(
        self,
        symbol: str,
        exchange: str,
        interval: str = "5m",
        num_candles: int = 100
    ) -> pd.DataFrame:
        """Generate simulated candle data."""
        import numpy as np

        dates = pd.date_range(end=datetime.now(), periods=num_candles, freq='5min')
        base_price = self._simulate_price(symbol)

        returns = np.random.randn(num_candles) * 0.005
        prices = base_price * (1 + np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices + np.random.randn(num_candles) * 0.5,
            'high': prices + np.abs(np.random.randn(num_candles)) * 1,
            'low': prices - np.abs(np.random.randn(num_candles)) * 1,
            'close': prices,
            'volume': np.random.randint(1000, 10000, num_candles)
        }, index=dates)

        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        return df

    def get_atm_strike(self, symbol: str, spot_price: float = None, strike_interval: int = 50) -> float:
        """Get ATM strike price."""
        if spot_price is None:
            spot_price = self._simulate_price(symbol)
        return round(spot_price / strike_interval) * strike_interval

    def get_nearest_expiry(self, symbol: str) -> str:
        """Get nearest expiry date."""
        from datetime import timedelta
        today = datetime.now()
        # Find next Thursday
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_thursday = today + timedelta(days=days_ahead)
        return next_thursday.strftime("%d%b%Y").upper()

    def get_option_symbol(self, underlying: str, expiry: str, strike: float, option_type: str) -> str:
        """Generate option symbol."""
        return f"{underlying} {expiry} {int(strike)} {option_type}"

    def get_lot_size(self, symbol: str) -> int:
        """Get lot size."""
        lot_sizes = {
            "NIFTY": 25,
            "BANKNIFTY": 15,
            "FINNIFTY": 25,
            "MIDCPNIFTY": 50
        }
        for key, lot in lot_sizes.items():
            if key in symbol.upper():
                return lot
        return 1

    def get_expiry_dates(self, symbol: str) -> List[str]:
        """Get expiry dates."""
        from datetime import timedelta
        today = datetime.now()
        expiries = []
        for i in range(4):
            days_ahead = 3 - today.weekday() + (i * 7)
            if days_ahead <= 0:
                days_ahead += 7
            exp_date = today + timedelta(days=days_ahead)
            expiries.append(exp_date.strftime("%d%b%Y").upper())
        return expiries

    def get_stats(self) -> Dict[str, Any]:
        """Get paper trading statistics."""
        return {
            "initial_balance": self._initial_balance,
            "current_balance": self._balance,
            "total_pnl": self._total_pnl,
            "pnl_percentage": (self._total_pnl / self._initial_balance) * 100,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "total_trades": self._winning_trades + self._losing_trades,
            "win_rate": (self._winning_trades / max(1, self._winning_trades + self._losing_trades)) * 100,
            "open_positions": len([p for p in self._positions.values() if p["quantity"] != 0])
        }

    def print_summary(self):
        """Print paper trading summary."""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("PAPER TRADING SUMMARY")
        print("=" * 50)
        print(f"Initial Balance:   {stats['initial_balance']:>15,.2f}")
        print(f"Current Balance:   {stats['current_balance']:>15,.2f}")
        print(f"Total P&L:         {stats['total_pnl']:>15,.2f}")
        print(f"P&L %:             {stats['pnl_percentage']:>14.2f}%")
        print("-" * 50)
        print(f"Winning Trades:    {stats['winning_trades']:>15}")
        print(f"Losing Trades:     {stats['losing_trades']:>15}")
        print(f"Win Rate:          {stats['win_rate']:>14.2f}%")
        print(f"Open Positions:    {stats['open_positions']:>15}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    print("Paper Trading Simulator")
    print("=" * 50)

    # Test simulator
    sim = PaperTradingSimulator()

    # Place a test order
    result = sim.place_order(
        symbol="NIFTY 30JAN2026 24000 CE",
        exchange="NFO",
        transaction_type="BUY",
        quantity=25,
        order_type="MARKET"
    )
    print(f"Order Result: {result}")

    # Get order status
    if result["order_id"]:
        status = sim.get_order_status(result["order_id"])
        print(f"Order Status: {status}")

    # Print summary
    sim.print_summary()
