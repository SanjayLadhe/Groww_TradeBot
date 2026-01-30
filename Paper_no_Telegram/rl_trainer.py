"""
RL Trainer Module - Offline and Online Training

This module provides training utilities for the RL trading agent:
1. Offline training on historical data (backtesting)
2. Online training during paper trading
3. Hyperparameter tuning
4. Model evaluation

Usage:
    # Offline training
    python rl_trainer.py --mode offline --episodes 1000
    
    # Evaluate model
    python rl_trainer.py --mode evaluate --model models/rl_trading_model.pt

Author: Algo Trading Bot
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import json

from rl_config import (
    RL_ALGORITHM,
    EPISODES_PER_TRAINING,
    MAX_STEPS_PER_EPISODE,
    BATCH_SIZE,
    MODEL_SAVE_DIR,
    SAVE_FREQUENCY,
    CHECKPOINT_FREQUENCY,
    HARD_LIMITS,
    get_state_dimension
)
from rl_agent import PPOAgent, RLDecisionMaker
from rl_state_builder import StateBuilder
from rl_reward import RewardCalculator

# Import trading components for simulation
from VWAP import calculate_vwap
from ATRTrailingStop import calculate_atr
from adx_indicator import calculate_adx
from Fractal_Chaos_Bands import calculate_fractal_chaos_bands
from ce_entry_logic import check_ce_entry_conditions
from pe_entry_logic import check_pe_entry_conditions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    Simulated trading environment for RL training.
    
    Simulates the trading process using historical data,
    allowing the RL agent to learn without risking real money.
    """
    
    def __init__(
        self,
        initial_balance: float = 100000,
        max_positions: int = 2,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001
    ):
        """Initialize trading environment."""
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        self.state_builder = StateBuilder()
        self.reward_calculator = RewardCalculator()
        
        # Episode state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.balance = self.initial_balance
        self.positions: Dict[str, Dict] = {}
        self.trades_today = 0
        self.daily_pnl = 0
        self.wins = 0
        self.losses = 0
        self.current_step = 0
        
        # Generate random market data for training
        self.market_data = self._generate_market_data()
        self.current_idx = 50  # Start after enough data for indicators
        
        return self._get_state()
    
    def _generate_market_data(self, n_candles: int = 500) -> pd.DataFrame:
        """Generate synthetic market data for training."""
        np.random.seed(int(time.time() * 1000) % (2**32))
        
        # Generate random walk price series
        base_price = 100 + np.random.rand() * 100
        returns = np.random.randn(n_candles) * 0.02  # 2% daily volatility
        
        # Add some trend
        trend = np.linspace(0, np.random.randn() * 0.5, n_candles)
        prices = base_price * np.exp(np.cumsum(returns) + trend)
        
        # Generate OHLCV
        df = pd.DataFrame({
            'open': prices + np.random.randn(n_candles) * prices * 0.005,
            'high': prices + np.abs(np.random.randn(n_candles)) * prices * 0.01,
            'low': prices - np.abs(np.random.randn(n_candles)) * prices * 0.01,
            'close': prices,
            'volume': np.random.randint(10000, 100000, n_candles)
        })
        
        # Ensure high > low
        df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
        df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999
        
        # Add timestamps
        df.index = pd.date_range(
            start=datetime.now() - timedelta(minutes=5 * n_candles),
            periods=n_candles,
            freq='5min'
        )
        
        # Calculate indicators
        df = calculate_vwap(df)
        df = calculate_atr(df)
        df = calculate_adx(df)
        df = calculate_fractal_chaos_bands(df)
        
        return df
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        if self.current_idx >= len(self.market_data):
            return np.zeros(self.state_builder.state_dim)
        
        df = self.market_data.iloc[:self.current_idx + 1]
        
        # Get position if any
        position = None
        if self.positions:
            pos_symbol = list(self.positions.keys())[0]
            position = self.positions[pos_symbol]
            position["current_price"] = df['close'].iloc[-1]
        
        session_stats = {
            "daily_pnl": self.daily_pnl,
            "win_rate": (self.wins / max(1, self.wins + self.losses)) * 100,
            "trades_today": self.trades_today,
            "max_trades": self.max_positions
        }
        
        return self.state_builder.build_state(df, position, session_stats)
    
    def _check_signal(self) -> Tuple[bool, str]:
        """Check if there's a valid entry signal."""
        df = self.market_data.iloc[:self.current_idx + 1]
        
        # Check CE signal
        ce_signal = check_ce_entry_conditions(df, "SIM", None)
        if ce_signal.get("valid", False):
            return True, "CE"
        
        # Check PE signal
        pe_signal = check_pe_entry_conditions(df, "SIM", None)
        if pe_signal.get("valid", False):
            return True, "PE"
        
        return False, ""
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action dict from RL agent
            
        Returns:
            next_state, reward, done, info
        """
        self.current_step += 1
        self.current_idx += 1
        
        if self.current_idx >= len(self.market_data) - 1:
            # Episode done
            return self._get_state(), 0, True, {"reason": "end_of_data"}
        
        current_price = self.market_data['close'].iloc[self.current_idx]
        reward = 0
        info = {}
        
        # Check if there's a signal
        has_signal, signal_type = self._check_signal()
        
        # Handle position management
        if self.positions:
            # Monitor existing position
            pos_symbol = list(self.positions.keys())[0]
            position = self.positions[pos_symbol]
            entry_price = position["entry_price"]
            
            # Update current price
            position["current_price"] = current_price
            position["highest_price"] = max(position["highest_price"], current_price)
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Stop loss hit
            if current_price <= position["stop_loss"]:
                should_exit = True
                exit_reason = "SL"
            # Target hit
            elif current_price >= position["target"]:
                should_exit = True
                exit_reason = "TARGET"
            # Trailing stop hit
            elif current_price <= position["trailing_stop"]:
                should_exit = True
                exit_reason = "TSL"
            # Max time (60 steps)
            elif self.current_step - position.get("entry_step", 0) > 60:
                should_exit = True
                exit_reason = "TIME"
            
            if should_exit:
                # Close position
                pnl = (current_price - entry_price) * position["quantity"]
                pnl_pct = ((current_price / entry_price) - 1) * 100
                
                # Apply slippage
                pnl -= abs(pnl) * self.slippage_pct
                
                self.balance += pnl
                self.daily_pnl += pnl
                
                if pnl >= 0:
                    self.wins += 1
                else:
                    self.losses += 1
                
                # Calculate reward
                reward = self.reward_calculator.calculate_reward(
                    action_taken=True,
                    position_closed=True,
                    entry_price=entry_price,
                    exit_price=current_price,
                    quantity=position["quantity"],
                    stop_loss=position["stop_loss"],
                    target=position["target"],
                    exit_reason=exit_reason,
                    time_in_trade=self.current_step - position.get("entry_step", 0)
                )
                
                del self.positions[pos_symbol]
                info["trade_closed"] = True
                info["pnl"] = pnl
                info["exit_reason"] = exit_reason
            else:
                # Update trailing stop
                if current_price > entry_price * 1.02:  # In 2% profit
                    new_tsl = current_price * 0.95  # 5% trailing
                    if new_tsl > position["trailing_stop"]:
                        position["trailing_stop"] = new_tsl
                
                # Small holding reward/penalty
                unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100
                reward = self.reward_calculator.calculate_reward(
                    action_taken=True,
                    position_closed=False,
                    entry_price=entry_price,
                    current_price=current_price,
                    quantity=position["quantity"],
                    stop_loss=position["stop_loss"],
                    target=position["target"],
                    time_in_trade=self.current_step - position.get("entry_step", 0)
                )
        
        elif has_signal and len(self.positions) < self.max_positions:
            # Check if RL says to take signal
            if action.get("take_signal", False):
                # Open position
                entry_price = current_price * (1 + self.slippage_pct)  # Slippage
                
                # Apply RL adjustments
                sl_pct = 0.15 * action.get("sl_adjustment", 1.0)
                sl_pct = np.clip(sl_pct, 0.05, 0.30)
                
                stop_loss = entry_price * (1 - sl_pct)
                risk = entry_price - stop_loss
                
                target_mult = 3 * action.get("target_adjustment", 1.0)
                target = entry_price + (risk * target_mult)
                
                # Position size
                quantity = int(25 * action.get("position_size_multiplier", 1.0))
                quantity = max(1, min(quantity, 50))
                
                self.positions["SIM_POS"] = {
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "target": target,
                    "trailing_stop": stop_loss,
                    "highest_price": entry_price,
                    "signal_type": signal_type,
                    "entry_step": self.current_step
                }
                
                self.trades_today += 1
                info["trade_opened"] = True
                
                # Small positive reward for taking valid signal
                reward = 0.01
            else:
                # Skipped a valid signal
                reward = -0.005  # Small penalty for missed opportunity
                info["signal_skipped"] = True
        
        # Check if episode should end
        done = False
        if self.balance < self.initial_balance * 0.8:  # 20% drawdown
            done = True
            info["reason"] = "max_drawdown"
            reward -= 1.0  # Large penalty
        elif self.current_step >= MAX_STEPS_PER_EPISODE:
            done = True
            info["reason"] = "max_steps"
        
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def get_metrics(self) -> Dict:
        """Get episode metrics."""
        return {
            "final_balance": self.balance,
            "pnl": self.balance - self.initial_balance,
            "pnl_pct": ((self.balance / self.initial_balance) - 1) * 100,
            "trades": self.trades_today,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": (self.wins / max(1, self.wins + self.losses)) * 100
        }


class RLTrainer:
    """
    Trainer for RL trading agent.
    
    Handles training loop, evaluation, and checkpointing.
    """
    
    def __init__(
        self,
        agent: PPOAgent = None,
        env: TradingEnvironment = None
    ):
        """Initialize trainer."""
        self.agent = agent or PPOAgent()
        self.env = env or TradingEnvironment()
        
        # Training history
        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_win_rates = []
        
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def train(
        self,
        n_episodes: int = None,
        save_freq: int = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train agent for specified episodes.
        
        Args:
            n_episodes: Number of episodes to train
            save_freq: How often to save checkpoints
            verbose: Print training progress
            
        Returns:
            Training metrics
        """
        if n_episodes is None:
            n_episodes = EPISODES_PER_TRAINING
        if save_freq is None:
            save_freq = SAVE_FREQUENCY
        
        logger.info(f"Starting training for {n_episodes} episodes")
        
        best_reward = float('-inf')
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action from agent
                action = self.agent.select_action(state)
                
                # Environment step
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                discrete_action = 1 if action["take_signal"] else 0
                continuous_action = np.array([
                    action["position_size_multiplier"],
                    action["sl_adjustment"],
                    action["target_adjustment"],
                    action["trailing_stop_factor"]
                ])
                
                self.agent.store_experience(
                    state, discrete_action, continuous_action,
                    reward, next_state, done,
                    action["log_prob"], action["value"]
                )
                
                state = next_state
                episode_reward += reward
            
            # Update agent
            if len(self.agent.buffer) >= BATCH_SIZE:
                update_metrics = self.agent.update()
            
            # Track metrics
            metrics = self.env.get_metrics()
            self.episode_rewards.append(episode_reward)
            self.episode_pnls.append(metrics["pnl"])
            self.episode_win_rates.append(metrics["win_rate"])
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save(os.path.join(MODEL_SAVE_DIR, "best_model.pt"))
            
            # Periodic checkpoint
            if (episode + 1) % save_freq == 0:
                self.agent.save(os.path.join(MODEL_SAVE_DIR, f"checkpoint_{episode+1}.pt"))
            
            # Logging
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_pnl = np.mean(self.episode_pnls[-10:])
                avg_wr = np.mean(self.episode_win_rates[-10:])
                
                logger.info(
                    f"Episode {episode+1}/{n_episodes} | "
                    f"Reward: {avg_reward:.2f} | "
                    f"P&L: ₹{avg_pnl:.2f} | "
                    f"Win Rate: {avg_wr:.1f}%"
                )
        
        # Final save
        self.agent.save(os.path.join(MODEL_SAVE_DIR, "final_model.pt"))
        
        return {
            "total_episodes": n_episodes,
            "best_reward": best_reward,
            "final_avg_reward": np.mean(self.episode_rewards[-100:]),
            "final_avg_pnl": np.mean(self.episode_pnls[-100:]),
            "final_win_rate": np.mean(self.episode_win_rates[-100:])
        }
    
    def evaluate(
        self,
        n_episodes: int = 100,
        model_path: str = None
    ) -> Dict:
        """
        Evaluate trained agent.
        
        Args:
            n_episodes: Number of episodes to evaluate
            model_path: Path to model to evaluate
            
        Returns:
            Evaluation metrics
        """
        if model_path:
            self.agent.load(model_path)
        
        eval_rewards = []
        eval_pnls = []
        eval_win_rates = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Use deterministic policy for evaluation
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                episode_reward += reward
            
            metrics = self.env.get_metrics()
            eval_rewards.append(episode_reward)
            eval_pnls.append(metrics["pnl"])
            eval_win_rates.append(metrics["win_rate"])
        
        return {
            "episodes": n_episodes,
            "avg_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "avg_pnl": np.mean(eval_pnls),
            "std_pnl": np.std(eval_pnls),
            "avg_win_rate": np.mean(eval_win_rates),
            "sharpe": np.mean(eval_pnls) / (np.std(eval_pnls) + 1e-8) * np.sqrt(252)
        }
    
    def save_training_history(self, path: str = "logs/training_history.json"):
        """Save training history to file."""
        history = {
            "rewards": self.episode_rewards,
            "pnls": self.episode_pnls,
            "win_rates": self.episode_win_rates
        }
        
        with open(path, 'w') as f:
            json.dump(history, f)
        
        logger.info(f"Training history saved to {path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="RL Trading Agent Trainer")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--save-freq", type=int, default=50)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🧠 RL TRADING AGENT TRAINER")
    print("=" * 60)
    
    trainer = RLTrainer()
    
    if args.mode == "train":
        print(f"\n📊 Training for {args.episodes} episodes...")
        metrics = trainer.train(
            n_episodes=args.episodes,
            save_freq=args.save_freq,
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("📈 TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total Episodes: {metrics['total_episodes']}")
        print(f"Best Reward: {metrics['best_reward']:.2f}")
        print(f"Final Avg Reward: {metrics['final_avg_reward']:.2f}")
        print(f"Final Avg P&L: ₹{metrics['final_avg_pnl']:.2f}")
        print(f"Final Win Rate: {metrics['final_win_rate']:.1f}%")
        
        trainer.save_training_history()
        
    elif args.mode == "evaluate":
        model_path = args.model or os.path.join(MODEL_SAVE_DIR, "best_model.pt")
        print(f"\n📊 Evaluating model: {model_path}")
        
        metrics = trainer.evaluate(n_episodes=args.episodes, model_path=model_path)
        
        print("\n" + "=" * 60)
        print("📈 EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes: {metrics['episodes']}")
        print(f"Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Avg P&L: ₹{metrics['avg_pnl']:.2f} ± ₹{metrics['std_pnl']:.2f}")
        print(f"Avg Win Rate: {metrics['avg_win_rate']:.1f}%")
        print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
