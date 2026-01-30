"""
RL Reward Function Module

This module defines various reward functions for training the RL agent.
The reward function is crucial for shaping agent behavior.

Reward types:
1. Simple P&L - Raw profit/loss
2. Risk-Adjusted - Sharpe-like reward
3. Shaped - Multiple reward components

Author: Algo Trading Bot
"""

import numpy as np
from typing import Dict, Optional, Any
import logging

from rl_config import (
    REWARD_TYPE,
    REWARD_WEIGHTS,
    REWARD_SCALE,
    REWARD_CLIP
)

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Calculates rewards for RL agent based on trading outcomes.
    
    The reward function shapes how the agent learns. We use a multi-component
    reward that encourages profitable trading while penalizing excessive risk.
    """
    
    def __init__(self, reward_type: str = None):
        """
        Initialize reward calculator.
        
        Args:
            reward_type: Type of reward function to use
        """
        self.reward_type = reward_type or REWARD_TYPE
        
        # Running statistics for Sharpe calculation
        self._returns = []
        self._max_returns_tracked = 100
        
        # Trade-level tracking
        self._current_trade_high = 0
        self._current_trade_low = float('inf')
        
        logger.info(f"Reward Calculator initialized with type: {self.reward_type}")
    
    def calculate_reward(
        self,
        action_taken: bool,
        position_opened: bool,
        position_closed: bool,
        entry_price: float = 0,
        exit_price: float = 0,
        current_price: float = 0,
        stop_loss: float = 0,
        target: float = 0,
        quantity: int = 0,
        exit_reason: str = None,
        signal_was_valid: bool = True,
        time_in_trade: float = 0
    ) -> float:
        """
        Calculate reward for current step.
        
        Args:
            action_taken: Whether agent took an action
            position_opened: Whether a new position was opened
            position_closed: Whether position was closed
            entry_price: Entry price (if position opened)
            exit_price: Exit price (if position closed)
            current_price: Current market price
            stop_loss: Stop loss price
            target: Target price
            quantity: Position quantity
            exit_reason: Why position was closed (SL, TARGET, TSL, etc.)
            signal_was_valid: Whether the underlying signal was valid
            time_in_trade: Time in position (minutes)
            
        Returns:
            Calculated reward value
        """
        if self.reward_type == "simple_pnl":
            reward = self._simple_pnl_reward(
                position_closed, entry_price, exit_price, quantity
            )
        elif self.reward_type == "risk_adjusted":
            reward = self._risk_adjusted_reward(
                action_taken, position_opened, position_closed,
                entry_price, exit_price, current_price,
                stop_loss, target, quantity, exit_reason,
                signal_was_valid, time_in_trade
            )
        elif self.reward_type == "sharpe":
            reward = self._sharpe_reward(
                position_closed, entry_price, exit_price, quantity
            )
        else:
            reward = self._risk_adjusted_reward(
                action_taken, position_opened, position_closed,
                entry_price, exit_price, current_price,
                stop_loss, target, quantity, exit_reason,
                signal_was_valid, time_in_trade
            )
        
        # Scale and clip reward
        reward = reward * REWARD_SCALE
        reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)
        
        return reward
    
    def _simple_pnl_reward(
        self,
        position_closed: bool,
        entry_price: float,
        exit_price: float,
        quantity: int
    ) -> float:
        """
        Simple P&L based reward.
        
        Returns raw profit/loss when trade closes, 0 otherwise.
        """
        if position_closed and entry_price > 0:
            pnl = (exit_price - entry_price) * quantity
            # Normalize by typical trade size
            normalized_pnl = pnl / 10000  # Assuming ~10k typical trade
            return normalized_pnl
        
        return 0.0
    
    def _risk_adjusted_reward(
        self,
        action_taken: bool,
        position_opened: bool,
        position_closed: bool,
        entry_price: float,
        exit_price: float,
        current_price: float,
        stop_loss: float,
        target: float,
        quantity: int,
        exit_reason: str,
        signal_was_valid: bool,
        time_in_trade: float
    ) -> float:
        """
        Risk-adjusted reward with multiple components.
        
        Components:
        1. Realized P&L (when trade closes)
        2. Unrealized P&L (while in trade)
        3. Risk penalty (for taking excessive risk)
        4. Win/loss bonuses
        5. Exit type bonuses/penalties
        """
        reward = 0.0
        
        # Component 1: Realized P&L
        if position_closed and entry_price > 0:
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = ((exit_price / entry_price) - 1) * 100
            
            # Normalized P&L reward
            realized_reward = pnl_pct / 10  # Scale by 10%
            reward += realized_reward * REWARD_WEIGHTS["realized_pnl"]
            
            # Component 4: Win/Loss bonus
            if pnl > 0:
                reward += REWARD_WEIGHTS["win_bonus"]
            else:
                reward += REWARD_WEIGHTS["loss_penalty"]
            
            # Component 5: Exit type bonuses
            if exit_reason == "TARGET":
                reward += REWARD_WEIGHTS["target_hit_bonus"]
            elif exit_reason == "SL":
                reward += REWARD_WEIGHTS["sl_hit_penalty"]
            elif exit_reason == "TSL" and pnl > 0:
                # Trailing stop with profit is good
                reward += REWARD_WEIGHTS["target_hit_bonus"] * 0.5
            elif exit_reason == "MANUAL" or exit_reason == "EARLY":
                # Early exit penalty unless profitable
                if pnl < 0:
                    reward += REWARD_WEIGHTS["early_exit_penalty"]
        
        # Component 2: Unrealized P&L (small reward while holding)
        elif entry_price > 0 and current_price > 0 and not position_closed:
            unrealized_pct = ((current_price / entry_price) - 1) * 100
            reward += (unrealized_pct / 100) * REWARD_WEIGHTS["unrealized_pnl"]
            
            # Small holding cost
            if time_in_trade > 0:
                reward += REWARD_WEIGHTS["holding_cost"] * (time_in_trade / 60)  # Per hour
        
        # Component 3: Risk penalty
        if entry_price > 0 and stop_loss > 0:
            risk_pct = abs((entry_price - stop_loss) / entry_price) * 100
            if risk_pct > 20:  # Excessive risk
                reward += REWARD_WEIGHTS["risk_penalty"] * (risk_pct - 20) / 10
        
        # Penalty for skipping valid signals (opportunity cost)
        if not action_taken and signal_was_valid:
            reward -= 0.01  # Small penalty for missed opportunity
        
        # Penalty for taking invalid signals
        if action_taken and not signal_was_valid:
            reward -= 0.05  # Penalty for false signal
        
        return reward
    
    def _sharpe_reward(
        self,
        position_closed: bool,
        entry_price: float,
        exit_price: float,
        quantity: int
    ) -> float:
        """
        Sharpe ratio inspired reward.
        
        Rewards consistent returns over high variance.
        """
        if position_closed and entry_price > 0:
            returns = (exit_price - entry_price) / entry_price
            
            # Track returns for Sharpe calculation
            self._returns.append(returns)
            if len(self._returns) > self._max_returns_tracked:
                self._returns.pop(0)
            
            # Calculate rolling Sharpe-like ratio
            if len(self._returns) >= 5:
                mean_return = np.mean(self._returns)
                std_return = np.std(self._returns) + 1e-8
                sharpe = mean_return / std_return
                return sharpe
            else:
                return returns * 10  # Early trades just use scaled returns
        
        return 0.0
    
    def update_trade_tracking(self, current_price: float, entry_price: float):
        """Update tracking for maximum favorable/adverse excursion."""
        if entry_price > 0:
            pnl_pct = ((current_price / entry_price) - 1) * 100
            self._current_trade_high = max(self._current_trade_high, pnl_pct)
            self._current_trade_low = min(self._current_trade_low, pnl_pct)
    
    def reset_trade_tracking(self):
        """Reset trade-level tracking."""
        self._current_trade_high = 0
        self._current_trade_low = float('inf')
    
    def get_trade_stats(self) -> Dict[str, float]:
        """Get current trade statistics."""
        return {
            "max_favorable_excursion": self._current_trade_high,
            "max_adverse_excursion": self._current_trade_low,
            "recent_returns_mean": np.mean(self._returns) if self._returns else 0,
            "recent_returns_std": np.std(self._returns) if self._returns else 0
        }


class RewardShaper:
    """
    Shapes rewards using potential-based reward shaping.
    
    This helps accelerate learning by providing intermediate rewards
    based on state improvement.
    """
    
    def __init__(self, gamma: float = 0.99):
        """Initialize reward shaper."""
        self.gamma = gamma
        self._last_potential = None
    
    def calculate_potential(
        self,
        position_pnl: float,
        distance_to_target: float,
        distance_to_sl: float,
        time_to_close: float
    ) -> float:
        """
        Calculate state potential.
        
        Higher potential for states closer to target,
        lower for states near SL or close to market end.
        """
        # Potential increases with unrealized profit
        pnl_potential = np.tanh(position_pnl / 10) * 0.5
        
        # Potential increases as we approach target
        if distance_to_target > 0:
            target_potential = (1 - distance_to_target) * 0.3
        else:
            target_potential = 0
        
        # Potential decreases near stop loss
        if distance_to_sl < 0.1:  # Within 10% of SL
            sl_potential = -0.5
        else:
            sl_potential = 0
        
        # Potential decreases near market close (time pressure)
        time_potential = time_to_close * 0.2  # More time = better
        
        return pnl_potential + target_potential + sl_potential + time_potential
    
    def shape_reward(
        self,
        base_reward: float,
        position_pnl: float,
        distance_to_target: float,
        distance_to_sl: float,
        time_to_close: float
    ) -> float:
        """
        Apply potential-based reward shaping.
        
        Shaped reward = base_reward + gamma * new_potential - old_potential
        """
        new_potential = self.calculate_potential(
            position_pnl, distance_to_target, distance_to_sl, time_to_close
        )
        
        if self._last_potential is not None:
            shaping_reward = self.gamma * new_potential - self._last_potential
            shaped_reward = base_reward + shaping_reward
        else:
            shaped_reward = base_reward
        
        self._last_potential = new_potential
        
        return shaped_reward
    
    def reset(self):
        """Reset shaper for new episode."""
        self._last_potential = None


def calculate_trade_reward(
    trade_result: Dict[str, Any],
    signal_info: Dict[str, Any] = None
) -> float:
    """
    Convenience function to calculate reward from trade result.
    
    Args:
        trade_result: Dict with trade outcome details
        signal_info: Optional dict with signal information
        
    Returns:
        Calculated reward
    """
    calculator = RewardCalculator()
    
    return calculator.calculate_reward(
        action_taken=trade_result.get("action_taken", True),
        position_opened=trade_result.get("position_opened", False),
        position_closed=trade_result.get("position_closed", False),
        entry_price=trade_result.get("entry_price", 0),
        exit_price=trade_result.get("exit_price", 0),
        current_price=trade_result.get("current_price", 0),
        stop_loss=trade_result.get("stop_loss", 0),
        target=trade_result.get("target", 0),
        quantity=trade_result.get("quantity", 0),
        exit_reason=trade_result.get("exit_reason"),
        signal_was_valid=signal_info.get("valid", True) if signal_info else True,
        time_in_trade=trade_result.get("time_in_trade", 0)
    )


if __name__ == "__main__":
    print("RL Reward Module")
    print("=" * 50)
    
    calculator = RewardCalculator()
    
    # Test winning trade
    reward_win = calculator.calculate_reward(
        action_taken=True,
        position_opened=False,
        position_closed=True,
        entry_price=100,
        exit_price=115,
        quantity=25,
        exit_reason="TARGET"
    )
    print(f"Winning trade reward: {reward_win:.4f}")
    
    # Test losing trade
    reward_loss = calculator.calculate_reward(
        action_taken=True,
        position_opened=False,
        position_closed=True,
        entry_price=100,
        exit_price=85,
        quantity=25,
        exit_reason="SL"
    )
    print(f"Losing trade reward: {reward_loss:.4f}")
    
    # Test holding position
    reward_hold = calculator.calculate_reward(
        action_taken=True,
        position_opened=False,
        position_closed=False,
        entry_price=100,
        current_price=105,
        quantity=25,
        time_in_trade=30
    )
    print(f"Holding reward: {reward_hold:.4f}")
