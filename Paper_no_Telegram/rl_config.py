"""
Reinforcement Learning Configuration Module

This module contains all configuration settings for the RL-enhanced trading system.
The RL acts as a meta-layer on top of the existing strategy to optimize:
1. Entry decisions (take signal or skip)
2. Position sizing (lot multiplier)
3. Risk parameters (SL/Target adjustments)
4. Exit timing optimization

Author: Algo Trading Bot
"""

# =============================================================================
# RL MAIN SWITCH
# =============================================================================

# Enable/Disable RL optimization
RL_ENABLED = True

# Use pre-trained model or train from scratch
USE_PRETRAINED_MODEL = False
PRETRAINED_MODEL_PATH = "models/rl_trading_model.pt"

# =============================================================================
# RL ALGORITHM SETTINGS
# =============================================================================

# Algorithm choice: "DQN", "PPO", "A2C", "SAC"
RL_ALGORITHM = "PPO"

# Neural network architecture
HIDDEN_LAYERS = [256, 128, 64]
ACTIVATION = "relu"  # relu, tanh, leaky_relu

# Learning parameters
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE parameter for PPO

# Training parameters
BATCH_SIZE = 64
N_EPOCHS = 10  # PPO epochs per update
CLIP_RANGE = 0.2  # PPO clip range
ENTROPY_COEF = 0.01  # Exploration bonus

# Experience replay (for DQN)
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 1000

# =============================================================================
# STATE SPACE CONFIGURATION
# =============================================================================

# Which features to include in state
STATE_FEATURES = {
    # Market indicators (from your strategy)
    "vwap_distance": True,      # Price distance from VWAP (%)
    "adx_value": True,          # ADX strength (0-100)
    "adx_trend": True,          # ADX rising/falling
    "plus_di": True,            # +DI value
    "minus_di": True,           # -DI value
    "di_crossover": True,       # DI crossover signal
    "atr_value": True,          # ATR normalized
    "atr_percentile": True,     # ATR percentile (volatility regime)
    "fractal_position": True,   # Price position in fractal bands
    
    # Price action
    "price_momentum": True,     # Short-term momentum
    "price_volatility": True,   # Recent price volatility
    "candle_patterns": True,    # Bullish/bearish patterns
    
    # Time features
    "time_of_day": True,        # Normalized time (0-1)
    "time_to_close": True,      # Minutes to market close
    "day_of_week": True,        # Encoded day
    
    # Position state
    "has_position": True,       # Currently in trade
    "position_pnl": True,       # Unrealized P&L %
    "position_duration": True,  # How long in trade
    "distance_to_sl": True,     # Distance to stop loss
    "distance_to_target": True, # Distance to target
    
    # Session stats
    "daily_pnl": True,          # Today's realized P&L
    "win_rate_today": True,     # Today's win rate
    "trades_remaining": True,   # Trades left today
    
    # Market regime
    "trend_strength": True,     # Overall trend strength
    "volatility_regime": True,  # Low/Medium/High vol
}

# State normalization
NORMALIZE_STATE = True
STATE_CLIP_RANGE = 5.0  # Clip normalized values

# =============================================================================
# ACTION SPACE CONFIGURATION
# =============================================================================

# Action type: "discrete" or "continuous"
ACTION_TYPE = "hybrid"  # Mix of discrete and continuous

# Discrete actions
DISCRETE_ACTIONS = {
    "entry_decision": ["SKIP", "TAKE_SIGNAL"],
    "exit_decision": ["HOLD", "EXIT_NOW", "PARTIAL_EXIT"],
}

# Continuous actions (ranges)
CONTINUOUS_ACTIONS = {
    "position_size_multiplier": (0.5, 2.0),   # Scale lots by this
    "sl_adjustment": (0.8, 1.2),              # Adjust SL by this factor
    "target_adjustment": (0.8, 1.5),          # Adjust target by this factor
    "trailing_stop_factor": (0.5, 1.5),       # Adjust TSL distance
}

# =============================================================================
# REWARD CONFIGURATION
# =============================================================================

# Reward function type
REWARD_TYPE = "risk_adjusted"  # "simple_pnl", "risk_adjusted", "sharpe"

# Reward components and weights
REWARD_WEIGHTS = {
    "realized_pnl": 1.0,        # Weight for closed trade P&L
    "unrealized_pnl": 0.1,      # Weight for open position P&L
    "risk_penalty": -0.5,       # Penalty for excessive risk
    "holding_cost": -0.001,     # Small penalty for holding
    "win_bonus": 0.2,           # Bonus for winning trade
    "loss_penalty": -0.1,       # Additional penalty for losing
    "target_hit_bonus": 0.5,    # Bonus for hitting target
    "sl_hit_penalty": -0.3,     # Penalty for hitting SL
    "early_exit_penalty": -0.1, # Penalty for premature exit
}

# Reward scaling
REWARD_SCALE = 100.0  # Scale rewards for stability
REWARD_CLIP = 10.0    # Clip rewards

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training mode
TRAINING_MODE = "online"  # "online" (live paper), "offline" (historical)

# Online training settings
ONLINE_UPDATE_FREQUENCY = 10  # Update model every N steps
EXPLORATION_RATE_START = 1.0
EXPLORATION_RATE_END = 0.1
EXPLORATION_DECAY_STEPS = 10000

# Offline training settings (historical backtesting)
HISTORICAL_DATA_PATH = "data/historical/"
BACKTEST_START_DATE = "2025-01-01"
BACKTEST_END_DATE = "2026-01-01"

# Episode settings
MAX_STEPS_PER_EPISODE = 500
EPISODES_PER_TRAINING = 100

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

# Save/Load settings
MODEL_SAVE_DIR = "models/"
SAVE_FREQUENCY = 50  # Save every N episodes
CHECKPOINT_FREQUENCY = 10  # Checkpoint every N episodes

# Model versioning
MODEL_VERSION = "v1.0"

# =============================================================================
# SAFETY CONSTRAINTS
# =============================================================================

# RL cannot override these safety limits
HARD_LIMITS = {
    "max_position_size": 10,    # Max lots regardless of RL
    "min_stop_loss_pct": 0.05,  # Min 5% SL
    "max_stop_loss_pct": 0.30,  # Max 30% SL
    "min_target_pct": 0.05,     # Min 5% target
    "max_target_pct": 1.0,      # Max 100% target
    "max_trades_per_day": 5,    # Hard limit on daily trades
}

# Confidence threshold for RL actions
MIN_CONFIDENCE_TO_ACT = 0.6  # Only act if RL confidence > 60%

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

# RL-specific logging
RL_LOG_FILE = "logs/rl_training.log"
LOG_ACTIONS = True
LOG_REWARDS = True
LOG_STATE_STATS = True

# TensorBoard logging
USE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = "logs/tensorboard/"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_state_dimension() -> int:
    """Calculate state dimension based on enabled features."""
    dim = 0
    feature_dims = {
        "vwap_distance": 1,
        "adx_value": 1,
        "adx_trend": 1,
        "plus_di": 1,
        "minus_di": 1,
        "di_crossover": 1,
        "atr_value": 1,
        "atr_percentile": 1,
        "fractal_position": 1,
        "price_momentum": 1,
        "price_volatility": 1,
        "candle_patterns": 3,  # Multiple patterns
        "time_of_day": 1,
        "time_to_close": 1,
        "day_of_week": 5,  # One-hot encoded
        "has_position": 1,
        "position_pnl": 1,
        "position_duration": 1,
        "distance_to_sl": 1,
        "distance_to_target": 1,
        "daily_pnl": 1,
        "win_rate_today": 1,
        "trades_remaining": 1,
        "trend_strength": 1,
        "volatility_regime": 3,  # One-hot encoded
    }
    
    for feature, enabled in STATE_FEATURES.items():
        if enabled:
            dim += feature_dims.get(feature, 1)
    
    return dim


def get_action_dimension() -> int:
    """Calculate action dimension based on configuration."""
    if ACTION_TYPE == "discrete":
        # Total discrete actions
        total = 1
        for actions in DISCRETE_ACTIONS.values():
            total *= len(actions)
        return total
    elif ACTION_TYPE == "continuous":
        return len(CONTINUOUS_ACTIONS)
    else:  # hybrid
        return len(DISCRETE_ACTIONS) + len(CONTINUOUS_ACTIONS)


def validate_config() -> bool:
    """Validate RL configuration."""
    errors = []
    
    if LEARNING_RATE <= 0:
        errors.append("LEARNING_RATE must be positive")
    
    if GAMMA < 0 or GAMMA > 1:
        errors.append("GAMMA must be between 0 and 1")
    
    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    
    if errors:
        for error in errors:
            print(f"RL Config Error: {error}")
        return False
    
    return True


if __name__ == "__main__":
    print("RL Configuration")
    print("=" * 50)
    print(f"Algorithm: {RL_ALGORITHM}")
    print(f"State Dimension: {get_state_dimension()}")
    print(f"Action Dimension: {get_action_dimension()}")
    print(f"Hidden Layers: {HIDDEN_LAYERS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Config Valid: {validate_config()}")
