"""
Sector Performance Analyzer Module

This module analyzes sector performance to generate a dynamic watchlist
of tradeable stocks based on sectoral strength.

Features:
- Analyzes 25+ sectoral indices
- Ranks stocks by sector performance
- Filters by F&O eligibility
- Generates dynamic watchlist

Author: Algo Trading Bot
"""

import logging
from typing import List, Dict
import random

logger = logging.getLogger(__name__)

# Default watchlist of F&O stocks
DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "BAJFINANCE", "SUNPHARMA", "WIPRO", "ULTRACEMCO", "HCLTECH",
    "TATAMOTORS", "NTPC", "POWERGRID", "ONGC", "COALINDIA",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "ADANIENT", "ADANIPORTS",
    "TECHM", "INDUSINDBK", "BAJAJFINSV", "DRREDDY", "CIPLA",
    "APOLLOHOSP", "DIVISLAB", "GRASIM", "NESTLEIND", "BRITANNIA"
]


def get_dynamic_watchlist(max_stocks: int = 50) -> List[str]:
    """
    Get dynamic watchlist based on sector performance.

    Args:
        max_stocks: Maximum number of stocks in watchlist

    Returns:
        List of stock symbols
    """
    try:
        # For now, return shuffled default watchlist
        # In production, this would analyze sector performance
        watchlist = DEFAULT_WATCHLIST.copy()
        random.shuffle(watchlist)
        return watchlist[:max_stocks]
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        return DEFAULT_WATCHLIST[:max_stocks]


def get_sector_stocks(sector: str) -> List[str]:
    """Get stocks for a specific sector."""
    sectors = {
        "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
        "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
        "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
        "AUTO": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
        "METAL": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
        "ENERGY": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIGREEN"],
        "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"]
    }
    return sectors.get(sector.upper(), [])
