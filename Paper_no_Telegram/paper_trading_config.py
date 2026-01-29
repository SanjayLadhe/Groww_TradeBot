"""
Paper Trading Configuration Module

This module contains all configuration settings for paper trading mode.
Edit these settings to customize the paper trading behavior.

Paper trading allows testing strategies without risking real money.
All trades are simulated with realistic execution.

Author: Algo Trading Bot
"""

# =============================================================================
# MAIN SWITCH
# =============================================================================

# Set to True to enable paper trading (simulated trades)
# Set to False to use live trading with real money
PAPER_TRADING_ENABLED = True

# =============================================================================
# PAPER TRADING ACCOUNT SETTINGS
# =============================================================================

# Starting balance for paper trading account
PAPER_TRADING_BALANCE = 100000.0

# Initial margin available
PAPER_TRADING_MARGIN = 500000.0

# =============================================================================
# EXECUTION SIMULATION SETTINGS
# =============================================================================

# Slippage simulation - percentage of price
# Simulates market impact and execution variance
SLIPPAGE_PERCENTAGE = 0.1  # 0.1% slippage

# Slippage in absolute points (used for low-priced options)
SLIPPAGE_POINTS = 0.5

# Order execution delay in seconds
# Simulates network and exchange latency
ORDER_EXECUTION_DELAY = 0.5

# Probability of order execution (1.0 = always executes)
# Set lower to simulate order rejections
ORDER_EXECUTION_PROBABILITY = 1.0

# =============================================================================
# FAILURE SIMULATION (FOR TESTING)
# =============================================================================

# Enable to simulate random order failures
SIMULATE_ORDER_FAILURES = False

# Failure rate when simulation is enabled (0.0 to 1.0)
FAILURE_RATE = 0.05  # 5% failure rate

# Simulate margin shortfall
SIMULATE_MARGIN_SHORTFALL = False

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Log file for paper trading activity
PAPER_TRADING_LOG_FILE = "paper_trading_log.txt"

# Enable detailed logging
DETAILED_LOGGING = True

# Log trade history to Excel
LOG_TO_EXCEL = True
EXCEL_LOG_FILE = "paper_trading_history.xlsx"

# =============================================================================
# MARKET SIMULATION SETTINGS
# =============================================================================

# Price update interval (seconds) for simulated prices
PRICE_UPDATE_INTERVAL = 1.0

# Maximum price movement per interval (percentage)
MAX_PRICE_MOVEMENT = 0.5

# Bid-ask spread simulation (percentage of price)
SIMULATED_SPREAD = 0.1  # 0.1% spread

# =============================================================================
# RISK MANAGEMENT (Paper Trading Specific)
# =============================================================================

# Maximum loss allowed before stopping (percentage of balance)
MAX_DRAWDOWN_PERCENT = 20.0

# Daily loss limit (percentage of balance)
DAILY_LOSS_LIMIT = 5.0

# Alert when drawdown reaches this level
DRAWDOWN_ALERT_THRESHOLD = 10.0

# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================

# Track and display performance metrics
TRACK_PERFORMANCE = True

# Display P&L summary interval (in trades)
PNL_SUMMARY_INTERVAL = 5

# Save daily performance summary
SAVE_DAILY_SUMMARY = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_paper_trading_status() -> dict:
    """Get current paper trading configuration status."""
    return {
        "enabled": PAPER_TRADING_ENABLED,
        "balance": PAPER_TRADING_BALANCE,
        "margin": PAPER_TRADING_MARGIN,
        "slippage_pct": SLIPPAGE_PERCENTAGE,
        "slippage_points": SLIPPAGE_POINTS,
        "execution_delay": ORDER_EXECUTION_DELAY,
        "simulate_failures": SIMULATE_ORDER_FAILURES,
        "log_file": PAPER_TRADING_LOG_FILE
    }


def validate_config() -> bool:
    """Validate configuration settings."""
    errors = []

    if PAPER_TRADING_BALANCE <= 0:
        errors.append("PAPER_TRADING_BALANCE must be positive")

    if SLIPPAGE_PERCENTAGE < 0 or SLIPPAGE_PERCENTAGE > 10:
        errors.append("SLIPPAGE_PERCENTAGE should be between 0 and 10")

    if ORDER_EXECUTION_PROBABILITY < 0 or ORDER_EXECUTION_PROBABILITY > 1:
        errors.append("ORDER_EXECUTION_PROBABILITY must be between 0 and 1")

    if errors:
        for error in errors:
            print(f"Config Error: {error}")
        return False

    return True


if __name__ == "__main__":
    print("Paper Trading Configuration")
    print("=" * 50)
    print(f"\nPaper Trading Enabled: {PAPER_TRADING_ENABLED}")
    print(f"Starting Balance: {PAPER_TRADING_BALANCE:,.2f}")
    print(f"Slippage: {SLIPPAGE_PERCENTAGE}% + {SLIPPAGE_POINTS} points")
    print(f"Execution Delay: {ORDER_EXECUTION_DELAY}s")
    print(f"Simulate Failures: {SIMULATE_ORDER_FAILURES}")
    print(f"\nConfiguration Valid: {validate_config()}")
