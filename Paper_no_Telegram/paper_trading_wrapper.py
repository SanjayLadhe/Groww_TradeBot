"""
Paper Trading Wrapper Module

Auto-switches between paper trading and live trading based on configuration.

Usage:
    from paper_trading_wrapper import get_trading_instance
    tsl = get_trading_instance(api_key, api_secret)
"""

from paper_trading_config import PAPER_TRADING_ENABLED

def get_trading_instance(api_key: str, api_secret: str = None):
    """Get appropriate trading instance based on config."""
    if PAPER_TRADING_ENABLED:
        from paper_trading_simulator import PaperTradingSimulator
        return PaperTradingSimulator(api_key, api_secret)
    else:
        from Groww_Tradehull import Tradehull
        return Tradehull(api_key, api_secret)
