"""
RL State Builder Module

Converts market data, indicator values, and position state into a normalized
state vector for the RL agent.

The state representation captures:
1. Technical indicator values (from your existing strategy)
2. Price action features
3. Time-based features
4. Current position state
5. Session statistics
6. Market regime indicators

Author: Algo Trading Bot
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

from rl_config import (
    STATE_FEATURES,
    NORMALIZE_STATE,
    STATE_CLIP_RANGE,
    get_state_dimension
)

logger = logging.getLogger(__name__)


class StateBuilder:
    """
    Builds state vectors for RL agent from market data and position info.
    
    The state is a normalized vector that captures all relevant information
    for making trading decisions.
    """
    
    def __init__(self):
        """Initialize state builder with normalization parameters."""
        self.state_dim = get_state_dimension()
        
        # Running statistics for normalization
        self._running_mean = np.zeros(self.state_dim)
        self._running_var = np.ones(self.state_dim)
        self._count = 0
        
        # Feature extractors
        self._feature_extractors = {
            "vwap_distance": self._extract_vwap_distance,
            "adx_value": self._extract_adx_value,
            "adx_trend": self._extract_adx_trend,
            "plus_di": self._extract_plus_di,
            "minus_di": self._extract_minus_di,
            "di_crossover": self._extract_di_crossover,
            "atr_value": self._extract_atr_value,
            "atr_percentile": self._extract_atr_percentile,
            "fractal_position": self._extract_fractal_position,
            "price_momentum": self._extract_price_momentum,
            "price_volatility": self._extract_price_volatility,
            "candle_patterns": self._extract_candle_patterns,
            "time_of_day": self._extract_time_of_day,
            "time_to_close": self._extract_time_to_close,
            "day_of_week": self._extract_day_of_week,
            "has_position": self._extract_has_position,
            "position_pnl": self._extract_position_pnl,
            "position_duration": self._extract_position_duration,
            "distance_to_sl": self._extract_distance_to_sl,
            "distance_to_target": self._extract_distance_to_target,
            "daily_pnl": self._extract_daily_pnl,
            "win_rate_today": self._extract_win_rate_today,
            "trades_remaining": self._extract_trades_remaining,
            "trend_strength": self._extract_trend_strength,
            "volatility_regime": self._extract_volatility_regime,
        }
        
        logger.info(f"StateBuilder initialized with {self.state_dim} features")
    
    def build_state(
        self,
        df: pd.DataFrame,
        position: Optional[Dict] = None,
        session_stats: Optional[Dict] = None,
        current_time: datetime = None
    ) -> np.ndarray:
        """
        Build complete state vector from market data and context.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            position: Current position info (None if no position)
            session_stats: Today's trading statistics
            current_time: Current datetime
            
        Returns:
            Normalized state vector as numpy array
        """
        if current_time is None:
            current_time = datetime.now()
        
        if session_stats is None:
            session_stats = {
                "daily_pnl": 0,
                "win_rate": 0,
                "trades_today": 0,
                "max_trades": 2
            }
        
        # Extract all enabled features
        features = []
        
        for feature_name, enabled in STATE_FEATURES.items():
            if enabled and feature_name in self._feature_extractors:
                extractor = self._feature_extractors[feature_name]
                feature_values = extractor(df, position, session_stats, current_time)
                
                if isinstance(feature_values, (list, np.ndarray)):
                    features.extend(feature_values)
                else:
                    features.append(feature_values)
        
        state = np.array(features, dtype=np.float32)
        
        # Normalize state
        if NORMALIZE_STATE:
            state = self._normalize(state)
        
        return state
    
    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics."""
        # Update running statistics
        self._count += 1
        delta = state - self._running_mean
        self._running_mean += delta / self._count
        delta2 = state - self._running_mean
        self._running_var += delta * delta2
        
        # Normalize
        std = np.sqrt(self._running_var / max(1, self._count - 1) + 1e-8)
        normalized = (state - self._running_mean) / std
        
        # Clip to prevent extreme values
        normalized = np.clip(normalized, -STATE_CLIP_RANGE, STATE_CLIP_RANGE)
        
        return normalized
    
    # =========================================================================
    # FEATURE EXTRACTORS - Technical Indicators
    # =========================================================================
    
    def _extract_vwap_distance(self, df, position, stats, time) -> float:
        """Extract price distance from VWAP as percentage."""
        if df.empty or 'vwap' not in df.columns:
            return 0.0
        
        close = df['close'].iloc[-1]
        vwap = df['vwap'].iloc[-1]
        
        if vwap == 0:
            return 0.0
        
        return ((close - vwap) / vwap) * 100
    
    def _extract_adx_value(self, df, position, stats, time) -> float:
        """Extract ADX value normalized to 0-1 range."""
        if df.empty or 'adx' not in df.columns:
            return 0.0
        
        adx = df['adx'].iloc[-1]
        return min(adx / 100, 1.0)  # Normalize to 0-1
    
    def _extract_adx_trend(self, df, position, stats, time) -> float:
        """Extract ADX trend (rising/falling)."""
        if df.empty or 'adx' not in df.columns or len(df) < 2:
            return 0.0
        
        current_adx = df['adx'].iloc[-1]
        prev_adx = df['adx'].iloc[-2]
        
        # Return normalized change
        if prev_adx == 0:
            return 0.0
        
        change = (current_adx - prev_adx) / prev_adx
        return np.tanh(change * 10)  # Squash to -1 to 1
    
    def _extract_plus_di(self, df, position, stats, time) -> float:
        """Extract +DI value normalized."""
        if df.empty or 'plus_di' not in df.columns:
            return 0.0
        
        return min(df['plus_di'].iloc[-1] / 100, 1.0)
    
    def _extract_minus_di(self, df, position, stats, time) -> float:
        """Extract -DI value normalized."""
        if df.empty or 'minus_di' not in df.columns:
            return 0.0
        
        return min(df['minus_di'].iloc[-1] / 100, 1.0)
    
    def _extract_di_crossover(self, df, position, stats, time) -> float:
        """Extract DI crossover signal (-1 bearish, 0 none, 1 bullish)."""
        if df.empty or 'plus_di' not in df.columns or len(df) < 2:
            return 0.0
        
        plus_di = df['plus_di'].iloc[-1]
        minus_di = df['minus_di'].iloc[-1]
        prev_plus = df['plus_di'].iloc[-2]
        prev_minus = df['minus_di'].iloc[-2]
        
        # Bullish crossover
        if prev_plus <= prev_minus and plus_di > minus_di:
            return 1.0
        # Bearish crossover
        elif prev_plus >= prev_minus and plus_di < minus_di:
            return -1.0
        
        return 0.0
    
    def _extract_atr_value(self, df, position, stats, time) -> float:
        """Extract ATR as percentage of price."""
        if df.empty or 'atr' not in df.columns:
            return 0.0
        
        atr = df['atr'].iloc[-1]
        close = df['close'].iloc[-1]
        
        if close == 0:
            return 0.0
        
        return (atr / close) * 100
    
    def _extract_atr_percentile(self, df, position, stats, time) -> float:
        """Extract ATR percentile (volatility regime)."""
        if df.empty or 'atr' not in df.columns or len(df) < 20:
            return 0.5
        
        current_atr = df['atr'].iloc[-1]
        atr_values = df['atr'].iloc[-50:].values
        
        percentile = np.sum(atr_values <= current_atr) / len(atr_values)
        return percentile
    
    def _extract_fractal_position(self, df, position, stats, time) -> float:
        """Extract price position within fractal bands (-1 to 1)."""
        if df.empty:
            return 0.0
        
        close = df['close'].iloc[-1]
        
        if 'upper_band' in df.columns and 'lower_band' in df.columns:
            upper = df['upper_band'].iloc[-1]
            lower = df['lower_band'].iloc[-1]
            
            if upper == lower:
                return 0.0
            
            # Normalize position: -1 at lower, 0 at middle, 1 at upper
            middle = (upper + lower) / 2
            range_size = (upper - lower) / 2
            
            return (close - middle) / range_size
        
        return 0.0
    
    # =========================================================================
    # FEATURE EXTRACTORS - Price Action
    # =========================================================================
    
    def _extract_price_momentum(self, df, position, stats, time) -> float:
        """Extract short-term price momentum."""
        if df.empty or len(df) < 5:
            return 0.0
        
        close = df['close'].iloc[-1]
        close_5 = df['close'].iloc[-5]
        
        if close_5 == 0:
            return 0.0
        
        momentum = ((close - close_5) / close_5) * 100
        return np.tanh(momentum)  # Squash to -1 to 1
    
    def _extract_price_volatility(self, df, position, stats, time) -> float:
        """Extract recent price volatility."""
        if df.empty or len(df) < 20:
            return 0.0
        
        returns = df['close'].pct_change().iloc[-20:]
        volatility = returns.std() * 100
        
        return min(volatility, 5.0) / 5.0  # Normalize
    
    def _extract_candle_patterns(self, df, position, stats, time) -> List[float]:
        """Extract candle pattern signals."""
        if df.empty:
            return [0.0, 0.0, 0.0]
        
        latest = df.iloc[-1]
        open_p = latest.get('open', 0)
        high = latest.get('high', 0)
        low = latest.get('low', 0)
        close = latest.get('close', 0)
        
        if open_p == 0:
            return [0.0, 0.0, 0.0]
        
        body = close - open_p
        range_size = high - low
        
        # Feature 1: Body size relative to range
        body_ratio = abs(body) / range_size if range_size > 0 else 0
        
        # Feature 2: Upper wick ratio
        upper_wick = high - max(open_p, close)
        upper_ratio = upper_wick / range_size if range_size > 0 else 0
        
        # Feature 3: Direction (bullish/bearish)
        direction = 1.0 if body > 0 else -1.0 if body < 0 else 0.0
        
        return [body_ratio, upper_ratio, direction]
    
    # =========================================================================
    # FEATURE EXTRACTORS - Time Features
    # =========================================================================
    
    def _extract_time_of_day(self, df, position, stats, time) -> float:
        """Extract normalized time of day (0 at open, 1 at close)."""
        market_open = time.replace(hour=9, minute=15, second=0)
        market_close = time.replace(hour=15, minute=30, second=0)
        
        total_minutes = (market_close - market_open).total_seconds() / 60
        elapsed_minutes = (time - market_open).total_seconds() / 60
        
        return max(0, min(1, elapsed_minutes / total_minutes))
    
    def _extract_time_to_close(self, df, position, stats, time) -> float:
        """Extract minutes to market close (normalized)."""
        market_close = time.replace(hour=15, minute=30, second=0)
        minutes_to_close = (market_close - time).total_seconds() / 60
        
        # Normalize: 0 at close, 1 at 6+ hours before
        return min(minutes_to_close / 360, 1.0)
    
    def _extract_day_of_week(self, df, position, stats, time) -> List[float]:
        """Extract day of week as one-hot encoding."""
        day = time.weekday()  # 0=Monday, 4=Friday
        one_hot = [0.0] * 5
        if 0 <= day < 5:
            one_hot[day] = 1.0
        return one_hot
    
    # =========================================================================
    # FEATURE EXTRACTORS - Position State
    # =========================================================================
    
    def _extract_has_position(self, df, position, stats, time) -> float:
        """Extract whether we have an open position."""
        return 1.0 if position is not None else 0.0
    
    def _extract_position_pnl(self, df, position, stats, time) -> float:
        """Extract current position P&L percentage."""
        if position is None:
            return 0.0
        
        entry = position.get('entry_price', 0)
        current = position.get('current_price', entry)
        
        if entry == 0:
            return 0.0
        
        pnl_pct = ((current - entry) / entry) * 100
        return np.tanh(pnl_pct / 10)  # Squash to -1 to 1
    
    def _extract_position_duration(self, df, position, stats, time) -> float:
        """Extract position duration in normalized form."""
        if position is None:
            return 0.0
        
        entry_time = position.get('entry_time')
        if entry_time is None:
            return 0.0
        
        duration_minutes = (time - entry_time).total_seconds() / 60
        
        # Normalize: 0 at entry, 1 at 6+ hours
        return min(duration_minutes / 360, 1.0)
    
    def _extract_distance_to_sl(self, df, position, stats, time) -> float:
        """Extract distance to stop loss as percentage."""
        if position is None:
            return 0.0
        
        current = position.get('current_price', 0)
        sl = position.get('trailing_stop', position.get('stop_loss', 0))
        
        if current == 0:
            return 0.0
        
        distance = ((current - sl) / current) * 100
        return np.tanh(distance / 10)
    
    def _extract_distance_to_target(self, df, position, stats, time) -> float:
        """Extract distance to target as percentage."""
        if position is None:
            return 0.0
        
        current = position.get('current_price', 0)
        target = position.get('target', 0)
        
        if current == 0:
            return 0.0
        
        distance = ((target - current) / current) * 100
        return np.tanh(distance / 20)
    
    # =========================================================================
    # FEATURE EXTRACTORS - Session Stats
    # =========================================================================
    
    def _extract_daily_pnl(self, df, position, stats, time) -> float:
        """Extract today's P&L normalized."""
        daily_pnl = stats.get('daily_pnl', 0)
        # Normalize assuming max daily P&L of ±50000
        return np.tanh(daily_pnl / 50000)
    
    def _extract_win_rate_today(self, df, position, stats, time) -> float:
        """Extract today's win rate."""
        return stats.get('win_rate', 0) / 100
    
    def _extract_trades_remaining(self, df, position, stats, time) -> float:
        """Extract trades remaining today (normalized)."""
        trades_today = stats.get('trades_today', 0)
        max_trades = stats.get('max_trades', 2)
        
        remaining = max_trades - trades_today
        return remaining / max_trades if max_trades > 0 else 0.0
    
    # =========================================================================
    # FEATURE EXTRACTORS - Market Regime
    # =========================================================================
    
    def _extract_trend_strength(self, df, position, stats, time) -> float:
        """Extract overall trend strength."""
        if df.empty or 'adx' not in df.columns:
            return 0.0
        
        adx = df['adx'].iloc[-1]
        
        # Determine direction from DI
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            plus_di = df['plus_di'].iloc[-1]
            minus_di = df['minus_di'].iloc[-1]
            direction = 1 if plus_di > minus_di else -1
        else:
            direction = 1
        
        # ADX indicates strength, direction indicates trend
        strength = (adx / 100) * direction
        return strength
    
    def _extract_volatility_regime(self, df, position, stats, time) -> List[float]:
        """Extract volatility regime as one-hot (low/medium/high)."""
        percentile = self._extract_atr_percentile(df, position, stats, time)
        
        if percentile < 0.33:
            return [1.0, 0.0, 0.0]  # Low
        elif percentile < 0.67:
            return [0.0, 1.0, 0.0]  # Medium
        else:
            return [0.0, 0.0, 1.0]  # High
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        names = []
        
        feature_dims = {
            "candle_patterns": ["candle_body_ratio", "candle_upper_wick", "candle_direction"],
            "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
            "volatility_regime": ["vol_low", "vol_med", "vol_high"],
        }
        
        for feature_name, enabled in STATE_FEATURES.items():
            if enabled:
                if feature_name in feature_dims:
                    names.extend(feature_dims[feature_name])
                else:
                    names.append(feature_name)
        
        return names
    
    def reset_normalization(self):
        """Reset normalization statistics."""
        self._running_mean = np.zeros(self.state_dim)
        self._running_var = np.ones(self.state_dim)
        self._count = 0
    
    def save_normalization(self, path: str):
        """Save normalization parameters."""
        np.savez(path,
                 mean=self._running_mean,
                 var=self._running_var,
                 count=self._count)
    
    def load_normalization(self, path: str):
        """Load normalization parameters."""
        data = np.load(path)
        self._running_mean = data['mean']
        self._running_var = data['var']
        self._count = data['count']


if __name__ == "__main__":
    print("State Builder Module")
    print("=" * 50)
    
    builder = StateBuilder()
    print(f"State dimension: {builder.state_dim}")
    print(f"Feature names: {builder.get_feature_names()[:10]}...")
