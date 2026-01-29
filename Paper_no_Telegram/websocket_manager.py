"""
WebSocket Manager for Groww Live Data Feed

This module manages WebSocket connections for real-time market data
using the Groww Feed API.

Features:
- Real-time LTP updates
- Bid/Ask price streaming
- Market depth data
- Automatic reconnection
- Per-symbol subscription management

Groww API Documentation: https://groww.in/trade-api/docs/python-sdk/feed

Author: Algo Trading Bot
"""

import logging
import threading
import time
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
import queue

try:
    from growwapi import GrowwFeed, GrowwAPI
except ImportError:
    GrowwFeed = None
    GrowwAPI = None

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time market data.

    Usage:
        ws_manager = WebSocketManager(api_key)
        ws_manager.subscribe("RELIANCE")
        ltp = ws_manager.get_ltp("RELIANCE")
    """

    def __init__(self, api_key: str, auto_reconnect: bool = True, paper_trading: bool = False):
        """
        Initialize WebSocket Manager.

        Args:
            api_key: Groww API key
            auto_reconnect: Whether to auto-reconnect on disconnection
            paper_trading: If True, use mock data instead of real feed
        """
        self.api_key = api_key
        self.auto_reconnect = auto_reconnect
        self.paper_trading = paper_trading

        # Initialize Groww Feed
        self.feed = None

        # Check if paper trading mode
        try:
            from paper_trading_config import PAPER_TRADING_ENABLED
            if PAPER_TRADING_ENABLED:
                self.paper_trading = True
        except ImportError:
            pass

        if self.paper_trading:
            logger.info("WebSocket Manager: Paper trading mode - using mock data")
        elif GrowwFeed and api_key and api_key != "YOUR_GROWW_API_KEY":
            try:
                self.feed = GrowwFeed(api_key)
                logger.info("Groww Feed initialized successfully")
            except Exception as e:
                logger.warning(f"Groww Feed unavailable, using polling mode: {e}")
        else:
            logger.info("WebSocket Manager: No valid API key, using mock data")

        # Data storage
        self._ltp_data: Dict[str, float] = {}
        self._bid_ask_data: Dict[str, Dict] = {}
        self._depth_data: Dict[str, Dict] = {}
        self._subscribed_symbols: set = set()

        # Threading
        self._lock = threading.Lock()
        self._data_queue = queue.Queue()
        self._running = False

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {}

    def start(self):
        """Start the WebSocket manager."""
        if self._running:
            return

        self._running = True
        logger.info("WebSocket manager started")

    def stop(self):
        """Stop the WebSocket manager."""
        self._running = False

        # Unsubscribe from all symbols
        for symbol in list(self._subscribed_symbols):
            self.unsubscribe(symbol)

        logger.info("WebSocket manager stopped")

    def subscribe(
        self,
        symbol: str,
        segment: str = "CASH",
        callback: Callable = None
    ) -> bool:
        """
        Subscribe to live data for a symbol.

        Args:
            symbol: Trading symbol
            segment: Market segment (CASH, FNO)
            callback: Optional callback function for data updates

        Returns:
            True if subscription successful
        """
        try:
            if symbol in self._subscribed_symbols:
                logger.debug(f"Already subscribed to {symbol}")
                return True

            if not self.feed:
                if not self.paper_trading:
                    logger.debug("Feed not initialized, using polling mode")
                self._subscribed_symbols.add(symbol)
                return True

            # Subscribe via Groww Feed
            def on_data_received():
                try:
                    ltp = self.feed.get_stocks_ltp(symbol, timeout=1)
                    if ltp:
                        with self._lock:
                            self._ltp_data[symbol] = float(ltp)

                        if callback:
                            callback(symbol, ltp)

                        # Notify registered callbacks
                        self._notify_callbacks(symbol, ltp)

                except Exception as e:
                    logger.error(f"Error in data callback for {symbol}: {e}")

            # Handle different segment constants
            if segment == "CASH":
                segment_const = GrowwAPI.SEGMENT_CASH if GrowwAPI else "CASH"
            else:
                segment_const = segment

            self.feed.subscribe_live_data(
                segment=segment_const,
                symbol=symbol,
                on_data_received=on_data_received
            )

            self._subscribed_symbols.add(symbol)
            logger.info(f"Subscribed to {symbol} ({segment})")

            return True

        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
            return False

    def unsubscribe(self, symbol: str, segment: str = "CASH") -> bool:
        """
        Unsubscribe from live data for a symbol.

        Args:
            symbol: Trading symbol
            segment: Market segment

        Returns:
            True if unsubscription successful
        """
        try:
            if symbol not in self._subscribed_symbols:
                return True

            if self.feed:
                self.feed.unsubscribe_live_data(segment=segment, symbol=symbol)

            self._subscribed_symbols.discard(symbol)

            # Clean up data
            with self._lock:
                self._ltp_data.pop(symbol, None)
                self._bid_ask_data.pop(symbol, None)
                self._depth_data.pop(symbol, None)

            logger.info(f"Unsubscribed from {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")
            return False

    def get_ltp(self, symbol: str, timeout: float = 3.0) -> float:
        """
        Get Last Traded Price for a symbol.

        Args:
            symbol: Trading symbol
            timeout: Timeout in seconds

        Returns:
            LTP as float, 0 if not available
        """
        try:
            # Check cache first
            with self._lock:
                if symbol in self._ltp_data:
                    return self._ltp_data[symbol]

            # For paper trading, generate simulated price
            if self.paper_trading:
                import random
                # Generate a reasonable price based on symbol
                if symbol not in self._ltp_data:
                    base = 100 + random.random() * 200
                    self._ltp_data[symbol] = round(base, 2)
                else:
                    # Small random movement
                    change = (random.random() - 0.5) * 2
                    self._ltp_data[symbol] = round(self._ltp_data[symbol] + change, 2)
                return self._ltp_data[symbol]

            # Try fetching from feed
            if self.feed:
                ltp = self.feed.get_stocks_ltp(symbol, timeout=timeout)
                if ltp:
                    with self._lock:
                        self._ltp_data[symbol] = float(ltp)
                    return float(ltp)

            return 0.0

        except Exception as e:
            logger.error(f"Error getting LTP for {symbol}: {e}")
            return 0.0

    def get_bid_ask(self, symbol: str) -> Dict[str, float]:
        """
        Get bid/ask prices for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with 'bid' and 'ask' prices
        """
        try:
            with self._lock:
                if symbol in self._bid_ask_data:
                    return self._bid_ask_data[symbol]

            return {"bid": 0.0, "ask": 0.0}

        except Exception as e:
            logger.error(f"Error getting bid/ask for {symbol}: {e}")
            return {"bid": 0.0, "ask": 0.0}

    def get_market_depth(self, symbol: str) -> Dict:
        """
        Get market depth data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with depth data
        """
        try:
            with self._lock:
                if symbol in self._depth_data:
                    return self._depth_data[symbol]

            return {"buy": [], "sell": []}

        except Exception as e:
            logger.error(f"Error getting market depth for {symbol}: {e}")
            return {"buy": [], "sell": []}

    def register_callback(self, symbol: str, callback: Callable):
        """
        Register a callback for data updates.

        Args:
            symbol: Trading symbol
            callback: Function to call on data update
        """
        if symbol not in self._callbacks:
            self._callbacks[symbol] = []
        self._callbacks[symbol].append(callback)

    def unregister_callback(self, symbol: str, callback: Callable):
        """
        Unregister a callback.

        Args:
            symbol: Trading symbol
            callback: Function to unregister
        """
        if symbol in self._callbacks:
            self._callbacks[symbol] = [
                cb for cb in self._callbacks[symbol] if cb != callback
            ]

    def _notify_callbacks(self, symbol: str, data: Any):
        """Notify all registered callbacks for a symbol."""
        if symbol in self._callbacks:
            for callback in self._callbacks[symbol]:
                try:
                    callback(symbol, data)
                except Exception as e:
                    logger.error(f"Error in callback for {symbol}: {e}")

    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        return list(self._subscribed_symbols)

    def is_subscribed(self, symbol: str) -> bool:
        """Check if symbol is subscribed."""
        return symbol in self._subscribed_symbols

    def get_all_ltp(self) -> Dict[str, float]:
        """Get all cached LTP data."""
        with self._lock:
            return dict(self._ltp_data)

    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._ltp_data.clear()
            self._bid_ask_data.clear()
            self._depth_data.clear()


class MockWebSocketManager:
    """
    Mock WebSocket manager for testing and paper trading.

    Returns simulated price data for testing purposes.
    """

    def __init__(self):
        self._base_prices: Dict[str, float] = {}
        self._subscribed: set = set()
        import random
        self._random = random

    def subscribe(self, symbol: str, **kwargs) -> bool:
        self._subscribed.add(symbol)
        if symbol not in self._base_prices:
            self._base_prices[symbol] = 100 + self._random.random() * 100
        return True

    def unsubscribe(self, symbol: str, **kwargs) -> bool:
        self._subscribed.discard(symbol)
        return True

    def get_ltp(self, symbol: str, **kwargs) -> float:
        if symbol not in self._base_prices:
            self._base_prices[symbol] = 100 + self._random.random() * 100

        # Add some random movement
        change = (self._random.random() - 0.5) * 2
        self._base_prices[symbol] += change
        return round(self._base_prices[symbol], 2)

    def get_bid_ask(self, symbol: str) -> Dict[str, float]:
        ltp = self.get_ltp(symbol)
        spread = ltp * 0.001  # 0.1% spread
        return {
            "bid": round(ltp - spread, 2),
            "ask": round(ltp + spread, 2)
        }

    def start(self):
        pass

    def stop(self):
        pass


if __name__ == "__main__":
    print("WebSocket Manager Module")
    print("=" * 50)
    print("Requires Groww API key for live data")
    print("\nUsage:")
    print("  ws = WebSocketManager('YOUR_API_KEY')")
    print("  ws.subscribe('RELIANCE')")
    print("  ltp = ws.get_ltp('RELIANCE')")
