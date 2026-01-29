"""
Groww_Tradehull.py - Groww Trading Platform API Wrapper

This module provides a unified interface for interacting with the Groww Trading API,
similar to Dhan_Tradehull_V3.py for Dhan broker.

Features:
- Order placement (BUY/SELL, MARKET/LIMIT)
- Order status tracking and cancellation
- Balance and portfolio queries
- Historical candle data retrieval
- Option chain fetching
- Live price (LTP) data
- Account and position management

Groww API Documentation: https://groww.in/trade-api/docs/python-sdk
Package: pip install growwapi (version 1.5.0+)

Author: Algo Trading Bot
Date: 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, List, Tuple, Union, Any
import requests
import json

try:
    from growwapi import GrowwAPI, GrowwFeed
except ImportError:
    raise ImportError("Please install growwapi: pip install growwapi>=1.5.0")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Tradehull:
    """
    Groww Trading Platform Wrapper Class

    Provides unified interface for trading operations compatible with
    the existing Dhan trading bot architecture.
    """

    # Exchange Constants
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    NFO = "NFO"  # NSE F&O
    BFO = "BFO"  # BSE F&O

    # Segment Constants
    SEGMENT_CASH = "CASH"
    SEGMENT_FNO = "FNO"
    SEGMENT_COMMODITY = "COMMODITY"

    # Order Type Constants
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_SL = "SL"
    ORDER_TYPE_SL_MARKET = "SL-M"

    # Transaction Type Constants
    BUY = "BUY"
    SELL = "SELL"

    # Product Type Constants
    PRODUCT_CNC = "CNC"  # Cash and Carry (Delivery)
    PRODUCT_INTRADAY = "INTRADAY"
    PRODUCT_MARGIN = "MARGIN"
    PRODUCT_MTF = "MTF"

    # Order Status Constants
    STATUS_PENDING = "PENDING"
    STATUS_OPEN = "OPEN"
    STATUS_COMPLETE = "COMPLETE"
    STATUS_CANCELLED = "CANCELLED"
    STATUS_REJECTED = "REJECTED"
    STATUS_TRIGGER_PENDING = "TRIGGER_PENDING"

    # Validity Constants
    VALIDITY_DAY = "DAY"
    VALIDITY_IOC = "IOC"  # Immediate or Cancel

    def __init__(self, api_key: str, api_secret: str = None):
        """
        Initialize the Groww Tradehull wrapper.

        Args:
            api_key: Groww API Key
            api_secret: Groww API Secret (optional, for token generation)
        """
        self.api_key = api_key
        self.api_secret = api_secret

        # Initialize Groww API client
        try:
            self.groww = GrowwAPI(api_key)
            self.feed = None  # Initialized on demand
            logger.info("Groww API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groww API: {e}")
            raise

        # Cache for instrument data
        self._instrument_cache = {}
        self._option_chain_cache = {}
        self._ltp_cache = {}

        # Load instrument master
        self._load_instrument_master()

    def _load_instrument_master(self):
        """Load instrument master data for symbol lookups."""
        try:
            # Try to load from local file if exists
            instrument_file = "Dependencies/all_instruments.csv"
            try:
                self.instruments_df = pd.read_csv(instrument_file)
                logger.info(f"Loaded instruments from {instrument_file}")
            except FileNotFoundError:
                self.instruments_df = pd.DataFrame()
                logger.warning("Instrument file not found, will fetch from API")
        except Exception as e:
            logger.error(f"Error loading instrument master: {e}")
            self.instruments_df = pd.DataFrame()

    # ============================================
    # ORDER MANAGEMENT METHODS
    # ============================================

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
        validity: str = VALIDITY_DAY,
        disclosed_quantity: int = 0,
        tag: str = ""
    ) -> Dict[str, Any]:
        """
        Place an order on Groww.

        Args:
            symbol: Trading symbol (e.g., 'RELIANCE', 'NIFTY 25JAN 24000 CE')
            exchange: Exchange (NSE, BSE, NFO, MCX)
            transaction_type: BUY or SELL
            quantity: Order quantity
            order_type: MARKET, LIMIT, SL, SL-M
            price: Limit price (for LIMIT/SL orders)
            trigger_price: Trigger price (for SL/SL-M orders)
            product_type: CNC, INTRADAY, MARGIN
            validity: DAY or IOC
            disclosed_quantity: Disclosed quantity
            tag: Order tag for identification

        Returns:
            Dict with order_id and status
        """
        try:
            logger.info(f"Placing {transaction_type} order: {symbol} x {quantity} @ {order_type}")

            order_params = {
                "symbol": symbol,
                "exchange": exchange,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "product_type": product_type,
                "validity": validity,
            }

            if order_type in [self.ORDER_TYPE_LIMIT, self.ORDER_TYPE_SL]:
                order_params["price"] = price

            if order_type in [self.ORDER_TYPE_SL, self.ORDER_TYPE_SL_MARKET]:
                order_params["trigger_price"] = trigger_price

            if disclosed_quantity > 0:
                order_params["disclosed_quantity"] = disclosed_quantity

            if tag:
                order_params["tag"] = tag

            # Place order via Groww API
            response = self.groww.place_order(**order_params)

            if response and "order_id" in response:
                logger.info(f"Order placed successfully: {response['order_id']}")
                return {
                    "status": "success",
                    "order_id": response["order_id"],
                    "message": "Order placed successfully"
                }
            else:
                logger.error(f"Order placement failed: {response}")
                return {
                    "status": "error",
                    "order_id": None,
                    "message": str(response)
                }

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {
                "status": "error",
                "order_id": None,
                "message": str(e)
            }

    def modify_order(
        self,
        order_id: str,
        quantity: int = None,
        price: float = None,
        trigger_price: float = None,
        order_type: str = None
    ) -> Dict[str, Any]:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            quantity: New quantity (optional)
            price: New price (optional)
            trigger_price: New trigger price (optional)
            order_type: New order type (optional)

        Returns:
            Dict with modification status
        """
        try:
            modify_params = {"order_id": order_id}

            if quantity is not None:
                modify_params["quantity"] = quantity
            if price is not None:
                modify_params["price"] = price
            if trigger_price is not None:
                modify_params["trigger_price"] = trigger_price
            if order_type is not None:
                modify_params["order_type"] = order_type

            response = self.groww.modify_order(**modify_params)

            logger.info(f"Order {order_id} modified: {response}")
            return {
                "status": "success",
                "order_id": order_id,
                "message": "Order modified successfully"
            }

        except Exception as e:
            logger.error(f"Order modification error: {e}")
            return {
                "status": "error",
                "order_id": order_id,
                "message": str(e)
            }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Dict with cancellation status
        """
        try:
            response = self.groww.cancel_order(order_id=order_id)

            logger.info(f"Order {order_id} cancelled: {response}")
            return {
                "status": "success",
                "order_id": order_id,
                "message": "Order cancelled successfully"
            }

        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return {
                "status": "error",
                "order_id": order_id,
                "message": str(e)
            }

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of a specific order.

        Args:
            order_id: Order ID to check

        Returns:
            Dict with order details and status
        """
        try:
            response = self.groww.get_order_details(order_id=order_id)

            return {
                "status": "success",
                "order_id": order_id,
                "order_status": response.get("status"),
                "filled_quantity": response.get("filled_quantity", 0),
                "average_price": response.get("average_price", 0),
                "details": response
            }

        except Exception as e:
            logger.error(f"Get order status error: {e}")
            return {
                "status": "error",
                "order_id": order_id,
                "message": str(e)
            }

    def get_order_list(self, timeout: int = 5) -> List[Dict]:
        """
        Get list of all orders for the day.

        Args:
            timeout: Request timeout in seconds

        Returns:
            List of order dictionaries
        """
        try:
            orders = self.groww.get_order_list(timeout=timeout)
            return orders if orders else []
        except Exception as e:
            logger.error(f"Get order list error: {e}")
            return []

    def get_trade_list(self) -> List[Dict]:
        """
        Get list of all executed trades for the day.

        Returns:
            List of trade dictionaries
        """
        try:
            trades = self.groww.get_trade_list()
            return trades if trades else []
        except Exception as e:
            logger.error(f"Get trade list error: {e}")
            return []

    # ============================================
    # PORTFOLIO MANAGEMENT METHODS
    # ============================================

    def get_holdings(self) -> List[Dict]:
        """
        Get all holdings in the portfolio.

        Returns:
            List of holding dictionaries
        """
        try:
            holdings = self.groww.get_holdings()
            return holdings if holdings else []
        except Exception as e:
            logger.error(f"Get holdings error: {e}")
            return []

    def get_positions(self) -> Dict[str, List[Dict]]:
        """
        Get all open positions.

        Returns:
            Dict with 'day' and 'net' positions
        """
        try:
            positions = self.groww.get_positions()
            return positions if positions else {"day": [], "net": []}
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return {"day": [], "net": []}

    def get_funds(self) -> Dict[str, Any]:
        """
        Get available funds/balance.

        Returns:
            Dict with fund details
        """
        try:
            funds = self.groww.get_funds()
            return funds if funds else {}
        except Exception as e:
            logger.error(f"Get funds error: {e}")
            return {}

    def get_available_balance(self) -> float:
        """
        Get available trading balance.

        Returns:
            Available balance as float
        """
        try:
            funds = self.get_funds()
            return float(funds.get("available_balance", 0))
        except Exception as e:
            logger.error(f"Get available balance error: {e}")
            return 0.0

    # ============================================
    # MARKET DATA METHODS
    # ============================================

    def get_ltp(
        self,
        symbol: str,
        exchange: str = NSE,
        segment: str = SEGMENT_CASH
    ) -> float:
        """
        Get Last Traded Price for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, NFO)
            segment: Market segment

        Returns:
            LTP as float
        """
        try:
            # Initialize feed if not already done
            if self.feed is None:
                self.feed = GrowwFeed(self.api_key)

            ltp = self.feed.get_stocks_ltp(symbol, timeout=3)
            return float(ltp) if ltp else 0.0

        except Exception as e:
            logger.error(f"Get LTP error for {symbol}: {e}")
            return 0.0

    def get_quote(self, symbol: str, exchange: str = NSE) -> Dict[str, Any]:
        """
        Get detailed quote for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange

        Returns:
            Dict with quote details (LTP, bid, ask, volume, etc.)
        """
        try:
            quote = self.groww.get_quote(symbol=symbol, exchange=exchange)
            return quote if quote else {}
        except Exception as e:
            logger.error(f"Get quote error for {symbol}: {e}")
            return {}

    def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: datetime,
        to_date: datetime = None
    ) -> pd.DataFrame:
        """
        Get historical candle data.

        Args:
            symbol: Trading symbol
            exchange: Exchange
            interval: Candle interval ('1m', '3m', '5m', '15m', '30m', '1h', '1d')
            from_date: Start date
            to_date: End date (default: now)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if to_date is None:
                to_date = datetime.now()

            historical = self.groww.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d")
            )

            if historical and len(historical) > 0:
                df = pd.DataFrame(historical)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Get historical data error: {e}")
            return pd.DataFrame()

    def get_candles(
        self,
        symbol: str,
        exchange: str,
        interval: str = "5m",
        num_candles: int = 100
    ) -> pd.DataFrame:
        """
        Get recent candles for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange
            interval: Candle interval
            num_candles: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate from_date based on interval and num_candles
            interval_minutes = {
                "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "1d": 1440
            }

            minutes = interval_minutes.get(interval, 5)
            from_date = datetime.now() - timedelta(minutes=minutes * num_candles * 2)

            df = self.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                from_date=from_date
            )

            if not df.empty:
                return df.tail(num_candles)

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Get candles error: {e}")
            return pd.DataFrame()

    # ============================================
    # OPTION CHAIN METHODS
    # ============================================

    def get_option_chain(
        self,
        symbol: str,
        expiry_date: str = None
    ) -> pd.DataFrame:
        """
        Get option chain for a symbol.

        Args:
            symbol: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')
            expiry_date: Expiry date in YYYY-MM-DD format

        Returns:
            DataFrame with option chain data
        """
        try:
            option_chain = self.groww.get_option_chain(
                symbol=symbol,
                expiry_date=expiry_date
            )

            if option_chain:
                return pd.DataFrame(option_chain)

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Get option chain error: {e}")
            return pd.DataFrame()

    def get_atm_strike(
        self,
        symbol: str,
        spot_price: float = None,
        strike_interval: int = 50
    ) -> float:
        """
        Get ATM (At The Money) strike price.

        Args:
            symbol: Underlying symbol
            spot_price: Current spot price (fetched if not provided)
            strike_interval: Strike price interval

        Returns:
            ATM strike price
        """
        try:
            if spot_price is None:
                spot_price = self.get_ltp(symbol)

            atm_strike = round(spot_price / strike_interval) * strike_interval
            return atm_strike

        except Exception as e:
            logger.error(f"Get ATM strike error: {e}")
            return 0.0

    def get_option_symbol(
        self,
        underlying: str,
        expiry_date: str,
        strike: float,
        option_type: str
    ) -> str:
        """
        Generate option trading symbol.

        Args:
            underlying: Underlying symbol (NIFTY, BANKNIFTY, etc.)
            expiry_date: Expiry date (DDMMMYYYY format, e.g., '30JAN2026')
            strike: Strike price
            option_type: 'CE' or 'PE'

        Returns:
            Option trading symbol
        """
        try:
            # Format: NIFTY 30JAN2026 24000 CE
            symbol = f"{underlying} {expiry_date} {int(strike)} {option_type}"
            return symbol
        except Exception as e:
            logger.error(f"Get option symbol error: {e}")
            return ""

    # ============================================
    # LIVE DATA STREAMING METHODS
    # ============================================

    def init_feed(self):
        """Initialize the feed connection for live data."""
        try:
            if self.feed is None:
                self.feed = GrowwFeed(self.api_key)
            logger.info("Feed initialized successfully")
        except Exception as e:
            logger.error(f"Feed initialization error: {e}")

    def subscribe_live_data(
        self,
        symbols: List[str],
        segment: str = SEGMENT_CASH,
        callback: callable = None
    ):
        """
        Subscribe to live market data.

        Args:
            symbols: List of symbols to subscribe
            segment: Market segment
            callback: Callback function for data updates
        """
        try:
            self.init_feed()

            for symbol in symbols:
                self.feed.subscribe_live_data(
                    segment=segment,
                    symbol=symbol,
                    on_data_received=callback
                )

            logger.info(f"Subscribed to live data for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Subscribe live data error: {e}")

    def unsubscribe_live_data(self, symbols: List[str], segment: str = SEGMENT_CASH):
        """
        Unsubscribe from live market data.

        Args:
            symbols: List of symbols to unsubscribe
            segment: Market segment
        """
        try:
            if self.feed:
                for symbol in symbols:
                    self.feed.unsubscribe_live_data(segment=segment, symbol=symbol)

                logger.info(f"Unsubscribed from {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Unsubscribe live data error: {e}")

    # ============================================
    # UTILITY METHODS
    # ============================================

    def get_lot_size(self, symbol: str) -> int:
        """
        Get lot size for F&O symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Lot size as int
        """
        try:
            # Try to get from instruments data
            if not self.instruments_df.empty:
                mask = self.instruments_df['symbol'] == symbol
                if mask.any():
                    return int(self.instruments_df.loc[mask, 'lot_size'].values[0])

            # Default lot sizes
            default_lots = {
                "NIFTY": 25,
                "BANKNIFTY": 15,
                "FINNIFTY": 25,
                "MIDCPNIFTY": 50,
            }

            for key, lot in default_lots.items():
                if key in symbol.upper():
                    return lot

            return 1

        except Exception as e:
            logger.error(f"Get lot size error: {e}")
            return 1

    def get_expiry_dates(self, symbol: str) -> List[str]:
        """
        Get available expiry dates for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            List of expiry dates
        """
        try:
            expiries = self.groww.get_expiry_dates(symbol=symbol)
            return expiries if expiries else []
        except Exception as e:
            logger.error(f"Get expiry dates error: {e}")
            return []

    def get_nearest_expiry(self, symbol: str) -> str:
        """
        Get nearest expiry date for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            Nearest expiry date string
        """
        try:
            expiries = self.get_expiry_dates(symbol)
            if expiries:
                return expiries[0]
            return ""
        except Exception as e:
            logger.error(f"Get nearest expiry error: {e}")
            return ""

    def round_to_tick(self, price: float, tick_size: float = 0.05) -> float:
        """
        Round price to nearest tick size.

        Args:
            price: Price to round
            tick_size: Tick size

        Returns:
            Rounded price
        """
        return round(price / tick_size) * tick_size

    def calculate_margin_required(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float
    ) -> float:
        """
        Calculate margin required for a trade.

        Args:
            symbol: Trading symbol
            exchange: Exchange
            transaction_type: BUY or SELL
            quantity: Order quantity
            price: Order price

        Returns:
            Margin required as float
        """
        try:
            margin = self.groww.get_margin_required(
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                price=price
            )
            return float(margin) if margin else 0.0
        except Exception as e:
            logger.error(f"Calculate margin error: {e}")
            return 0.0


# Alias for compatibility
GrowwTradehull = Tradehull


if __name__ == "__main__":
    # Test the wrapper
    print("Groww Tradehull Wrapper")
    print("=" * 50)
    print("This module provides a unified interface for Groww API")
    print("\nUsage:")
    print("  from Groww_Tradehull import Tradehull")
    print("  tsl = Tradehull(api_key='YOUR_API_KEY')")
    print("  orders = tsl.get_order_list()")
    print("  ltp = tsl.get_ltp('RELIANCE')")
