"""
Trade Logger Module - Comprehensive Logging and Excel Export

This module provides:
1. Detailed logging of all trading activities with emoji indicators
2. LIVE Excel export using xlwings (real-time updates)
3. Real-time trade tracking matching Dhan_Algo format

Author: Algo Trading Bot
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

# Try to import xlwings for live Excel updates
try:
    import xlwings as xw
    XLWINGS_AVAILABLE = True
except ImportError:
    XLWINGS_AVAILABLE = False
    print("⚠️ xlwings not available, using openpyxl for Excel export")

# Configure detailed logger
logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Comprehensive trade logging and Excel export.

    Tracks all orders, executions, and P&L with export to Excel.
    """

    def __init__(self, log_dir: str = "logs", excel_file: str = "Live Trade Data.xlsx"):
        """
        Initialize Trade Logger.

        Args:
            log_dir: Directory for log files
            excel_file: Excel file name for trade export
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.excel_file = self.log_dir / excel_file

        # Initialize data storage
        self.orders: List[Dict] = []
        self.trades: List[Dict] = []
        self.positions: List[Dict] = []
        self.pnl_history: List[Dict] = []

        # Orderbook for live tracking (like Dhan)
        self.orderbook: Dict[str, Dict] = {}
        self.completed_orders: List[Dict] = []

        # Session info
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

        # Excel workbook (xlwings for live updates)
        self.wb = None
        self.live_sheet = None
        self.completed_sheet = None

        # Setup file logging
        self._setup_file_logging()

        # Setup Excel with xlwings
        self._setup_excel()

        self._print_banner()
        logger.info(f"Trade Logger initialized - Session: {self.session_id}")
        logger.info(f"Excel export file: {self.excel_file}")

    def _print_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 70)
        print("🚀 GROWW ALGO TRADING BOT - TRADE LOGGER INITIALIZED")
        print("=" * 70)
        print(f"📅 Session ID: {self.session_id}")
        print(f"⏰ Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Excel File: {self.excel_file}")
        print("=" * 70 + "\n")

    def _setup_file_logging(self):
        """Setup detailed file logging."""
        log_file = self.log_dir / f"trading_{self.session_id}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        logger.info(f"Log file: {log_file}")

    def _setup_excel(self):
        """Setup Excel workbook with xlwings for live updates."""
        try:
            if XLWINGS_AVAILABLE:
                # Check if file exists
                if self.excel_file.exists():
                    self.wb = xw.Book(str(self.excel_file))
                else:
                    # Create new workbook
                    self.wb = xw.Book()
                    self.wb.save(str(self.excel_file))

                # Setup sheets
                sheet_names = [s.name for s in self.wb.sheets]

                if 'Live_Trading' not in sheet_names:
                    self.wb.sheets.add('Live_Trading')
                if 'Completed_Orders' not in sheet_names:
                    self.wb.sheets.add('Completed_Orders')
                if 'Summary' not in sheet_names:
                    self.wb.sheets.add('Summary')

                self.live_sheet = self.wb.sheets['Live_Trading']
                self.completed_sheet = self.wb.sheets['Completed_Orders']
                self.summary_sheet = self.wb.sheets['Summary']

                # Clear existing data
                self.live_sheet.range("A1:Z100").clear()
                self.completed_sheet.range("A1:Z100").clear()

                # Set headers
                headers = ['Symbol', 'Date', 'Entry_Time', 'Entry_Price', 'Type',
                          'Qty', 'SL', 'Target', 'Current_Price', 'PnL', 'Status', 'Option_Type']
                self.live_sheet.range("A1").value = headers
                self.completed_sheet.range("A1").value = headers + ['Exit_Time', 'Exit_Price', 'Exit_Reason']

                print("✅ Excel workbook initialized with xlwings (LIVE updates enabled)")
            else:
                print("📝 Using openpyxl for Excel export (file-based updates)")

        except Exception as e:
            logger.error(f"Error setting up Excel: {e}")
            print(f"⚠️ Excel setup error: {e}")

    def log_order(
        self,
        order_id: str,
        symbol: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float,
        status: str,
        **kwargs
    ):
        """
        Log an order.

        Args:
            order_id: Order ID
            symbol: Trading symbol
            transaction_type: BUY/SELL
            quantity: Order quantity
            order_type: MARKET/LIMIT/SL
            price: Order price
            status: Order status
            **kwargs: Additional order details
        """
        now = datetime.now()
        order_record = {
            "timestamp": now,
            "order_id": order_id,
            "symbol": symbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            "status": status,
            "session_id": self.session_id,
            **kwargs
        }

        self.orders.append(order_record)

        # Emoji-based console output
        emoji = "🟢" if transaction_type == "BUY" else "🔴"
        status_emoji = "✅" if status == "COMPLETE" else "⏳"

        print("\n" + "=" * 70)
        print(f"{emoji} ORDER {transaction_type} - {now.strftime('%H:%M:%S')}")
        print("=" * 70)
        print(f"📊 Symbol:     {symbol}")
        print(f"💰 Price:      ₹{price:.2f}")
        print(f"📦 Quantity:   {quantity}")
        print(f"📋 Type:       {order_type}")
        print(f"{status_emoji} Status:     {status}")
        print(f"🔖 Order ID:   {order_id}")
        print("=" * 70)

        # Log to file
        logger.info(f"ORDER | {transaction_type} | {symbol} | Qty: {quantity} | "
                   f"Price: {price:.2f} | Type: {order_type} | Status: {status} | "
                   f"ID: {order_id}")

        # Auto-save to Excel
        self._save_to_excel()
        self._update_live_excel()

    def log_trade(
        self,
        trade_id: str,
        order_id: str,
        symbol: str,
        transaction_type: str,
        quantity: int,
        executed_price: float,
        **kwargs
    ):
        """
        Log an executed trade.

        Args:
            trade_id: Trade ID
            order_id: Related order ID
            symbol: Trading symbol
            transaction_type: BUY/SELL
            quantity: Executed quantity
            executed_price: Execution price
            **kwargs: Additional trade details
        """
        trade_record = {
            "timestamp": datetime.now(),
            "trade_id": trade_id,
            "order_id": order_id,
            "symbol": symbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "executed_price": executed_price,
            "value": quantity * executed_price,
            "session_id": self.session_id,
            **kwargs
        }

        self.trades.append(trade_record)

        logger.info(f"TRADE | {transaction_type} | {symbol} | Qty: {quantity} | "
                   f"Executed: {executed_price:.2f} | Value: {quantity * executed_price:.2f}")

        self._save_to_excel()

    def log_position_open(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target: float,
        option_type: str = "",
        **kwargs
    ):
        """
        Log position opening.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            stop_loss: Stop loss price
            target: Target price
            option_type: CE/PE for options
            **kwargs: Additional details
        """
        now = datetime.now()
        position_record = {
            "open_time": now,
            "close_time": None,
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": None,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "target": target,
            "option_type": option_type,
            "status": "OPEN",
            "pnl": 0,
            "pnl_pct": 0,
            "exit_reason": None,
            "session_id": self.session_id,
            **kwargs
        }

        self.positions.append(position_record)

        # Add to orderbook for live tracking
        self.orderbook[symbol] = {
            'symbol': symbol,
            'date': now.strftime('%Y-%m-%d'),
            'entry_time': now.strftime('%H:%M:%S'),
            'entry_price': entry_price,
            'type': 'BUY',
            'qty': quantity,
            'sl': stop_loss,
            'target': target,
            'current_price': entry_price,
            'pnl': 0,
            'status': 'OPEN',
            'option_type': option_type
        }

        # Calculate percentages
        sl_pct = ((stop_loss / entry_price) - 1) * 100
        target_pct = ((target / entry_price) - 1) * 100

        # Emoji-based console output
        print("\n" + "🟢" * 35)
        print(f"📈 POSITION OPENED - {now.strftime('%H:%M:%S')}")
        print("🟢" * 35)
        print(f"""
📊 Symbol:       {symbol}
🏷️  Option Type:  {option_type}

💵 Entry Price:  ₹{entry_price:.2f}
📦 Quantity:     {quantity}
💰 Value:        ₹{entry_price * quantity:,.2f}

🛑 Stop Loss:    ₹{stop_loss:.2f} ({sl_pct:.1f}%)
🎯 Target:       ₹{target:.2f} ({target_pct:.1f}%)
📊 Risk:Reward:  1:{abs(target_pct/sl_pct):.1f}
""")
        print("🟢" * 35 + "\n")

        logger.info("=" * 60)
        logger.info(f"POSITION OPENED: {symbol}")
        logger.info(f"  Entry Price: {entry_price:.2f}")
        logger.info(f"  Quantity: {quantity}")
        logger.info(f"  Stop Loss: {stop_loss:.2f} ({sl_pct:.1f}%)")
        logger.info(f"  Target: {target:.2f} ({target_pct:.1f}%)")
        logger.info(f"  Option Type: {option_type}")
        logger.info("=" * 60)

        self._save_to_excel()
        self._update_live_excel()

    def log_position_close(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str
    ):
        """
        Log position closing.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_reason: Reason for exit (SL/TARGET/TSL/MANUAL)
        """
        now = datetime.now()

        # Find the open position
        for pos in self.positions:
            if pos["symbol"] == symbol and pos["status"] == "OPEN":
                pos["close_time"] = now
                pos["exit_price"] = exit_price
                pos["status"] = "CLOSED"
                pos["exit_reason"] = exit_reason

                # Calculate P&L
                entry = pos["entry_price"]
                qty = pos["quantity"]
                pos["pnl"] = (exit_price - entry) * qty
                pos["pnl_pct"] = ((exit_price / entry) - 1) * 100

                # Log P&L history
                self.pnl_history.append({
                    "timestamp": now,
                    "symbol": symbol,
                    "pnl": pos["pnl"],
                    "pnl_pct": pos["pnl_pct"],
                    "exit_reason": exit_reason
                })

                # Update orderbook and move to completed
                if symbol in self.orderbook:
                    order = self.orderbook[symbol].copy()
                    order['exit_time'] = now.strftime('%H:%M:%S')
                    order['exit_price'] = exit_price
                    order['pnl'] = pos["pnl"]
                    order['status'] = 'CLOSED'
                    order['exit_reason'] = exit_reason
                    self.completed_orders.append(order)
                    del self.orderbook[symbol]

                # Emoji-based console output
                pnl_emoji = "💰" if pos["pnl"] >= 0 else "📉"
                result_emoji = "✅" if pos["pnl"] >= 0 else "❌"
                reason_emoji = {"SL": "🛑", "TARGET": "🎯", "TSL": "📈", "MANUAL": "👤", "TIME": "⏰"}.get(exit_reason, "📋")

                print("\n" + "🔴" * 35)
                print(f"{reason_emoji} POSITION CLOSED - {exit_reason}")
                print("🔴" * 35)
                print(f"""
📊 Symbol:       {symbol}

💵 Entry Price:  ₹{entry:.2f}
💵 Exit Price:   ₹{exit_price:.2f}
🔻 Stop Loss:    ₹{pos['stop_loss']:.2f}

{pnl_emoji} P&L:          ₹{pos['pnl']:,.2f} ({pos['pnl_pct']:.1f}%)
📦 Quantity:     {qty}

⏰ Entry Time:   {pos['open_time'].strftime('%H:%M:%S')}
⏰ Exit Time:    {now.strftime('%H:%M:%S')}

{result_emoji} Result:       {"PROFIT" if pos["pnl"] >= 0 else "LOSS"}
📝 Reason:       {exit_reason}
""")
                print("🔴" * 35 + "\n")

                logger.info("=" * 60)
                logger.info(f"POSITION CLOSED: {symbol}")
                logger.info(f"  Entry: {entry:.2f} | Exit: {exit_price:.2f}")
                logger.info(f"  P&L: {pos['pnl']:.2f} ({pos['pnl_pct']:.1f}%)")
                logger.info(f"  Reason: {exit_reason}")
                logger.info("=" * 60)

                break

        self._save_to_excel()
        self._update_live_excel()

    def log_tsl_update(self, symbol: str, old_tsl: float, new_tsl: float):
        """Log trailing stop loss update."""
        print(f"📈 TSL UPDATE | {symbol} | ₹{old_tsl:.2f} → ₹{new_tsl:.2f}")

        # Update orderbook
        if symbol in self.orderbook:
            self.orderbook[symbol]['sl'] = new_tsl

        logger.info(f"TSL UPDATE | {symbol} | Old: {old_tsl:.2f} -> New: {new_tsl:.2f}")
        self._update_live_excel()

    def update_current_price(self, symbol: str, current_price: float):
        """Update current price and P&L for a position."""
        if symbol in self.orderbook:
            entry_price = self.orderbook[symbol]['entry_price']
            qty = self.orderbook[symbol]['qty']
            pnl = (current_price - entry_price) * qty

            self.orderbook[symbol]['current_price'] = current_price
            self.orderbook[symbol]['pnl'] = pnl

            self._update_live_excel()

    def _update_live_excel(self):
        """Update Excel sheets in real-time using xlwings."""
        try:
            if XLWINGS_AVAILABLE and self.wb is not None:
                # Update Live Trading sheet
                if self.orderbook:
                    orderbook_df = pd.DataFrame(list(self.orderbook.values()))
                    self.live_sheet.range("A2:Z100").clear()
                    self.live_sheet.range("A2").value = orderbook_df.values.tolist()

                # Update Completed Orders sheet
                if self.completed_orders:
                    completed_df = pd.DataFrame(self.completed_orders)
                    self.completed_sheet.range("A2:Z100").clear()
                    self.completed_sheet.range("A2").value = completed_df.values.tolist()

                # Update Summary
                summary = self._generate_summary()
                self.summary_sheet.range("A1").value = [
                    ['Metric', 'Value'],
                    ['Session ID', summary['session_id']],
                    ['Total Orders', summary['total_orders']],
                    ['Open Positions', summary['open_positions']],
                    ['Closed Trades', summary['total_trades']],
                    ['Winning Trades', summary['winning_trades']],
                    ['Losing Trades', summary['losing_trades']],
                    ['Win Rate %', f"{summary['win_rate']:.1f}%"],
                    ['Total P&L', f"₹{summary['total_pnl']:,.2f}"],
                    ['Last Updated', datetime.now().strftime('%H:%M:%S')]
                ]

        except Exception as e:
            logger.debug(f"Excel update error: {e}")

    def log_scan(self, symbol: str, signal_type: str, details: Dict):
        """Log market scan results."""
        logger.debug(f"SCAN | {symbol} | Signal: {signal_type} | {details}")

    def log_indicator(self, symbol: str, indicators: Dict):
        """Log indicator values."""
        logger.debug(f"INDICATORS | {symbol} | {indicators}")

    def _save_to_excel(self):
        """Save all data to Excel file."""
        try:
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                # Orders sheet
                if self.orders:
                    orders_df = pd.DataFrame(self.orders)
                    orders_df.to_excel(writer, sheet_name='Orders', index=False)

                # Trades sheet
                if self.trades:
                    trades_df = pd.DataFrame(self.trades)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)

                # Positions sheet
                if self.positions:
                    positions_df = pd.DataFrame(self.positions)
                    positions_df.to_excel(writer, sheet_name='Positions', index=False)

                # P&L History sheet
                if self.pnl_history:
                    pnl_df = pd.DataFrame(self.pnl_history)
                    pnl_df.to_excel(writer, sheet_name='PnL_History', index=False)

                # Summary sheet
                summary_data = self._generate_summary()
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            logger.debug(f"Excel saved: {self.excel_file}")

        except Exception as e:
            logger.error(f"Error saving Excel: {e}")

    def _generate_summary(self) -> Dict:
        """Generate trading session summary."""
        total_trades = len([p for p in self.positions if p["status"] == "CLOSED"])
        winning = len([p for p in self.positions if p["status"] == "CLOSED" and p["pnl"] > 0])
        losing = len([p for p in self.positions if p["status"] == "CLOSED" and p["pnl"] < 0])

        total_pnl = sum(p["pnl"] for p in self.positions if p["status"] == "CLOSED")

        return {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "last_update": datetime.now(),
            "total_orders": len(self.orders),
            "total_trades": total_trades,
            "open_positions": len([p for p in self.positions if p["status"] == "OPEN"]),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": (winning / total_trades * 100) if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": total_pnl / total_trades if total_trades > 0 else 0
        }

    def print_summary(self):
        """Print trading session summary to console."""
        summary = self._generate_summary()
        pnl_emoji = "💰" if summary['total_pnl'] >= 0 else "📉"

        print("\n" + "=" * 70)
        print("📊 TRADING SESSION SUMMARY")
        print("=" * 70)
        print(f"🔖 Session ID:      {summary['session_id']}")
        print(f"⏰ Session Start:   {summary['session_start']}")
        print("-" * 70)
        print(f"📋 Total Orders:    {summary['total_orders']}")
        print(f"📈 Closed Trades:   {summary['total_trades']}")
        print(f"👁️  Open Positions:  {summary['open_positions']}")
        print("-" * 70)
        print(f"✅ Winning Trades:  {summary['winning_trades']}")
        print(f"❌ Losing Trades:   {summary['losing_trades']}")
        print(f"📊 Win Rate:        {summary['win_rate']:.1f}%")
        print("-" * 70)
        print(f"{pnl_emoji} Total P&L:       ₹{summary['total_pnl']:,.2f}")
        print(f"📈 Avg P&L/Trade:   ₹{summary['avg_pnl_per_trade']:,.2f}")
        print("=" * 70)
        print(f"\n📁 Excel file: {self.excel_file}")
        print()

    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions."""
        return [p for p in self.positions if p["status"] == "OPEN"]

    def get_closed_positions(self) -> List[Dict]:
        """Get list of closed positions."""
        return [p for p in self.positions if p["status"] == "CLOSED"]

    def close_excel(self):
        """Close Excel workbook and save."""
        try:
            if XLWINGS_AVAILABLE and self.wb is not None:
                self.wb.save()
                print(f"💾 Excel saved: {self.excel_file}")
        except Exception as e:
            logger.error(f"Error closing Excel: {e}")


# Global trade logger instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get or create global trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger


def init_trade_logger(log_dir: str = "logs", excel_file: str = "trade_log.xlsx") -> TradeLogger:
    """Initialize global trade logger with custom settings."""
    global _trade_logger
    _trade_logger = TradeLogger(log_dir=log_dir, excel_file=excel_file)
    return _trade_logger


if __name__ == "__main__":
    # Test the trade logger
    print("Trade Logger Module Test")
    print("=" * 50)

    logger = TradeLogger()

    # Simulate some trades
    logger.log_order(
        order_id="TEST001",
        symbol="NIFTY 30JAN2026 24000 CE",
        transaction_type="BUY",
        quantity=25,
        order_type="MARKET",
        price=150.50,
        status="COMPLETE"
    )

    logger.log_trade(
        trade_id="TRD001",
        order_id="TEST001",
        symbol="NIFTY 30JAN2026 24000 CE",
        transaction_type="BUY",
        quantity=25,
        executed_price=151.00
    )

    logger.log_position_open(
        symbol="NIFTY 30JAN2026 24000 CE",
        entry_price=151.00,
        quantity=25,
        stop_loss=128.35,
        target=218.95,
        option_type="CE"
    )

    logger.print_summary()
    print(f"\nCheck Excel file: {logger.excel_file}")
