"""
AUTOMATED DAILY SCANNER V3.1 - TWO-TIER STRATEGY (STREAMLINED CSV OUTPUT)
Implements optimal momentum breakout scanning with clean, organized CSV output

TIER 1 (Weekly): Enhanced Momentum V2 → Build breakout-ready watchlist
TIER 2 (Daily): Flexible Entry Validator → Find precise entry timing

STREAMLINED CHANGES:
- ONE comprehensive BUY signals CSV per scan
- Removed duplicate CSV generation
- Clean file naming convention
- All essential data preserved

Author: Advanced Stock Scanner Suite
Version: 3.1.1 - CSV Streamlined
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys
import time
import logging
from colorama import Fore, init
import requests
import json

init(autoreset=True)
warnings.filterwarnings('ignore')


def deduplicate_results(df, logger=None):
    """Smart deduplication - keeps best signal per symbol"""
    if df.empty: return df
    initial_count = len(df)

    # Determine score column
    score_col = 'Score'
    if 'Total_Score' in df.columns:
        score_col = 'Total_Score'
    elif 'Score' not in df.columns:
        # Fallback if neither exists
        return df.drop_duplicates(subset=['Symbol'], keep='first')

    sort_columns = ['Symbol', score_col]

    # Prefer recent data if available
    if 'Price_Date' in df.columns:
        sort_columns.append('Price_Date')
        ascending_order = [True, False, False]
    else:
        ascending_order = [True, False]

    # Sort: Symbol ASC, Score DESC, Date DESC
    df_sorted = df.sort_values(sort_columns, ascending=ascending_order)

    # Deduplicate
    df_deduped = df_sorted.drop_duplicates(subset=['Symbol'], keep='first')

    if logger and (initial_count - len(df_deduped) > 0):
        logger.info(f"Deduplication: {initial_count} -> {len(df_deduped)} duplicates removed")

    return df_deduped


# Fix Windows console encoding
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:

    USE_WATCHLIST_MODE = True
    WATCHLIST_FILE = 'watchlist_momentum_current.csv'

    # Fallback if watchlist doesn't exist
    STOCK_UNIVERSES = {
        '1': ('Nifty_50.csv', 'Nifty 50'),
        '2': ('Nifty_Next_50.csv', 'Nifty Next 50'),
        '3': ('Highest_Market_Cap_Companies.csv', 'Large Cap'),
        '4': ('Mid_Cap_Stocks.csv', 'Mid Cap'),
        '5': ('Small_Cap_Stocks.csv', 'Small Cap'),
        '6': 'ALL'
    }
    DEFAULT_UNIVERSE = 'Nifty_Next_50.csv'
    STOCK_UNIVERSE = 'Nifty_Next_50.csv'
    MIN_GRADE = 'A'
    RESOURCE_PATH = 'resources/'
    # ========== WEEKLY SCAN SETTINGS (TIER 1) ==========
    WEEKLY_SCAN_PERIOD = 60
    MIN_TOTAL_SCORE = 16
    MIN_RS_SCORE = 55
    BREAKOUT_READY_ONLY = False
    # ========== TELEGRAM SETTINGS ==========
    SEND_TELEGRAM = True
    TELEGRAM_BOT_TOKEN = "7668822476:AAEeSzWdt7DgzOs3Fsbz5_oZpPL8xoUpLH8"
    TELEGRAM_CHAT_ID = "7745188241"
    # ========== NOTIFICATION SETTINGS ==========
    SEND_BUY_ONLY = True
    ATTACH_CSV = True
    SEND_WEEKLY_SUMMARY = True
    # ========== LOGGING ==========
    LOG_FILE = 'automated_scanner_enhanced.log'
    LOG_LEVEL = logging.INFO
    USE_EMOJIS = not sys.platform.startswith('win')
    # ========== RETRY SETTINGS ==========
    MAX_RETRIES = 3
    RETRY_DELAY = 300

# EMOJI HELPERS

def emoji(code, fallback=''):
    """Return emoji if enabled, otherwise fallback text"""
    if Config.USE_EMOJIS:
        return code
    return fallback

E_START = emoji('🚀', '[START]')
E_CHECK = emoji('✅', '[OK]')
E_WARN = emoji('⚠️', '[WARN]')
E_ERROR = emoji('❌', '[ERROR]')
E_TIME = emoji('⏰', '')
E_CHART = emoji('📊', '')
E_TARGET = emoji('🎯', '')
E_TELEGRAM = emoji('📱', '[TELEGRAM]')
E_SAVE = emoji('💾', '[SAVE]')
E_CALENDAR = emoji('📅', '')
E_RETRY = emoji('🔄', '[RETRY]')
E_WEEKLY = emoji('📈', '[WEEKLY]')
E_DAILY = emoji('📉', '[DAILY]')

# LOGGING SETUP
class SafeFormatter(logging.Formatter):
    """Custom formatter that handles emoji encoding errors gracefully"""

    def format(self, record):
        formatted = super().format(record)

        if sys.platform.startswith('win'):
            try:
                formatted.encode('cp1252')
            except UnicodeEncodeError:
                formatted = ''.join(
                    char for char in formatted
                    if ord(char) < 65536 and char.isprintable() or char in '\n\r\t'
                )

        return formatted

def setup_logging():
    """Configure logging with safe emoji handling"""
    file_handler = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(Config.LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(Config.LOG_LEVEL)
    console_handler.setFormatter(SafeFormatter('%(asctime)s | %(levelname)s | %(message)s'))

    logger = logging.getLogger(__name__)
    logger.setLevel(Config.LOG_LEVEL)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

# TELEGRAM NOTIFIER

class TelegramNotifier:
    """Send Telegram notifications for Enhanced Momentum scan results"""

    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = Config.SEND_TELEGRAM and self.bot_token and self.chat_id

        if self.enabled:
            logger.info(f"{E_TELEGRAM} Telegram notifications enabled")
        else:
            logger.info(f"{E_WARN} Telegram notifications disabled")

    def send_message(self, message, parse_mode='HTML'):
        """Send message to Telegram"""
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info(f"{E_TELEGRAM} Telegram message sent successfully")
                return True
            else:
                logger.error(f"{E_ERROR} Telegram failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"{E_ERROR} Telegram error: {e}")
            return False

    def format_weekly_watchlist(self, watchlist_df):
        """Format weekly Enhanced Momentum watchlist for Telegram"""
        if watchlist_df is None or watchlist_df.empty:
            return None
        date_str = datetime.now().strftime('%d %B %Y')
        # Group by stage
        breakout_ready = watchlist_df[watchlist_df['Stage'] == 'BREAKOUT_READY']
        accumulation = watchlist_df[watchlist_df['Stage'] == 'ACCUMULATION']

        message = f"🔥 <b>ENHANCED MOMENTUM WATCHLIST</b>\n"
        message += f"📅 {date_str}\n"
        message += f"━━━━━━━━━━━━━━━━━\n\n"
        message += f"✅ <b>{len(watchlist_df)} Quality Stocks</b> (60-day validated)\n\n"

        # Breakout Ready stocks
        if len(breakout_ready) > 0:
            message += f"🚀 <b>BREAKOUT READY</b> ({len(breakout_ready)} stocks)\n"
            message += "━━━━━━━━━━━━━━━━━\n"

            for idx, row in breakout_ready.head(10).iterrows():
                symbol = row['Symbol']
                score = row.get('Total_Score', 0)
                price = row.get('Current_Price', 0)
                stop = row.get('Stop_Price', 0)
                rr = row.get('Risk_Reward', 0)
                rs = row.get('RS_Score', 0)

                message += f"• <code>{symbol:<12}</code> Score: {score}\n"
                message += f"  💰 ₹{price:.2f} | Stop: ₹{stop:.2f}\n"
                message += f"  📊 R:R: {rr:.2f}:1 | RS: {rs}\n\n"

            if len(breakout_ready) > 10:
                message += f"  ... and {len(breakout_ready) - 10} more\n"

            message += "\n"

        # Accumulation stocks
        if len(accumulation) > 0:
            message += f"📊 <b>ACCUMULATION</b> ({len(accumulation)} stocks)\n"
            message += "━━━━━━━━━━━━━━━━━\n"

            for idx, row in accumulation.head(5).iterrows():
                symbol = row['Symbol']
                score = row.get('Total_Score', 0)
                price = row.get('Current_Price', 0)
                base = row.get('Base_Quality', 0)

                message += f"• <code>{symbol:<12}</code> Score: {score}\n"
                message += f"  💰 ₹{price:.2f} | Base: {base}/5\n"

            if len(accumulation) > 5:
                message += f"  ... and {len(accumulation) - 5} more\n"

            message += "\n"

        # Statistics
        message += "━━━━━━━━━━━━━━━━━\n"
        message += f"📈 <b>STATISTICS</b>\n"
        message += f"Avg Score: {watchlist_df['Total_Score'].mean():.1f}\n"
        message += f"Avg R:R: {watchlist_df['Risk_Reward'].mean():.2f}:1\n"
        message += f"Avg RS: {watchlist_df['RS_Score'].mean():.1f}\n\n"

        message += "💡 <i>Monitor these stocks daily for entry signals</i>\n"
        message += "🎯 Run Tier 2 scan for BUY/WAIT decisions"

        return message

    def format_daily_signals(self, results):
        """Format daily entry signals for Telegram"""
        if not results:
            return None

        buy_stocks = results.get('buy', [])
        wait_stocks = results.get('wait', [])
        skip_stocks = results.get('skip', [])

        date_str = datetime.now().strftime('%d %B %Y')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%d %B')

        message = f"🎯 <b>DAILY ENTRY SIGNALS</b>\n"
        message += f"📅 {date_str}\n"
        message += f"━━━━━━━━━━━━━━━━━\n\n"

        # BUY SIGNALS
        if buy_stocks:
            message += f"🟢 <b>BUY TOMORROW ({tomorrow})</b>\n"
            message += f"✅ {len(buy_stocks)} stocks ready to enter\n"
            message += "━━━━━━━━━━━━━━━━━\n\n"

            for i, stock in enumerate(buy_stocks[:10], 1):
                symbol = stock.get('Symbol', 'N/A')
                grade = stock.get('Grade', 'N/A')
                score = stock.get('Score', 0)
                price = stock.get('Current_Price', 0)
                entry_low = stock.get('Entry_Low', 0)
                entry_high = stock.get('Entry_High', 0)
                stop = stock.get('Stop_Loss', 0)
                target1 = stock.get('Target_1', 0)
                risk_pct = stock.get('Risk_%', 0)

                message += f"<b>#{i}. {symbol}</b> | Grade: {grade} | Score: {score}/100\n"
                message += f"💰 Price: ₹{price:.2f}\n"
                message += f"🎯 Entry: ₹{entry_low:.2f} - ₹{entry_high:.2f}\n"
                message += f"🛑 Stop: ₹{stop:.2f} ({risk_pct:.1f}% risk)\n"
                message += f"📈 Target: ₹{target1:.2f}\n\n"

            if len(buy_stocks) > 10:
                message += f"... and {len(buy_stocks) - 10} more BUY signals\n\n"

        else:
            message += "🟡 <b>NO BUY SIGNALS TODAY</b>\n"
            message += "⏳ Market conditions not ideal for entry\n\n"

        # WAIT SIGNALS
        if wait_stocks:
            message += f"━━━━━━━━━━━━━━━━━\n"
            message += f"🟡 <b>WAIT LIST</b> ({len(wait_stocks)} stocks)\n"
            message += "━━━━━━━━━━━━━━━━━\n\n"

            for stock in wait_stocks[:5]:
                symbol = stock.get('Symbol', 'N/A')
                grade = stock.get('Grade', 'N/A')
                price = stock.get('Current_Price', 0)
                vwap = stock.get('VWAP', None)
                reason = stock.get('Reason', 'Waiting for better entry')

                message += f"• <code>{symbol:<12}</code> Grade: {grade} | ₹{price:.2f}\n"

                if vwap and vwap > 0:
                    message += f"  Wait for dip to ₹{vwap * 0.99:.2f}\n"
                else:
                    message += f"  {reason}\n"

            if len(wait_stocks) > 5:
                message += f"\n... and {len(wait_stocks) - 5} more WAIT signals\n"

            message += "\n"

        # Summary
        message += "━━━━━━━━━━━━━━━━━\n"
        message += f"📊 <b>SUMMARY</b>\n"
        message += f"🟢 BUY: {len(buy_stocks)}\n"
        message += f"🟡 WAIT: {len(wait_stocks)}\n"
        message += f"🔴 SKIP: {len(skip_stocks)}\n\n"

        if buy_stocks:
            message += "⚡ <b>ACTION REQUIRED:</b>\n"
            message += "1. Verify prices on NSE/BSE\n"
            message += "2. Set entry alerts\n"
            message += "3. Enter tomorrow at open\n"
            message += "4. Set stop loss IMMEDIATELY\n\n"

        message += "⚠️ <i>Always use stop loss. Max 3-5% per position.</i>"

        return message

    def send_weekly_update(self, watchlist_df):
        """Send weekly watchlist update"""
        message = self.format_weekly_watchlist(watchlist_df)
        if message:
            if len(message) > 4000:
                parts = self._split_message(message, 4000)
                for part in parts:
                    self.send_message(part)
                    time.sleep(1)
            else:
                self.send_message(message)

    def send_daily_signals(self, results):
        """Send daily entry signals"""
        message = self.format_daily_signals(results)
        if message:
            if len(message) > 4000:
                parts = self._split_message(message, 4000)
                for part in parts:
                    self.send_message(part)
                    time.sleep(1)
            else:
                self.send_message(message)

    def _split_message(self, message, max_length=4000):
        """Split long message into chunks"""
        lines = message.split('\n')
        chunks = []
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) + 1 < max_length:
                current_chunk += line + "\n"
            else:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

# UNIVERSE SELECTION MENU

def select_universe():
    """Interactive menu to select stock universe"""
    resource_path = Config.RESOURCE_PATH
    print(f"{Fore.CYAN}{'=' * 100}")
    print(f"{Fore.CYAN}📂 SELECT STOCK UNIVERSE TO SCAN")
    print(f"{Fore.CYAN}{'=' * 100}\n")

    print(f"{Fore.WHITE}Available stock lists:")
    for key, value in Config.STOCK_UNIVERSES.items():
        if key == '6':
            print(f"{Fore.WHITE}  [6] ALL files in resources folder")
            continue
        filename, description = value
        file_path = os.path.join(resource_path, filename)
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"{Fore.WHITE}  [{key}] {description:20s} {exists}")

    print(f"{Fore.WHITE}  [0] Exit\n")
    choice = input(f"{Fore.CYAN}Enter your choice (0-6): ").strip()
    if choice == '0':
        return []
    symbols = []
    if choice == '6':
        print(f"\n{Fore.YELLOW}⚠️  WARNING: Scanning ALL files will be VERY slow!")
        confirm = input(f"{Fore.YELLOW}Are you sure? (y/n): ").strip().lower()
        if confirm != 'y':
            return []

        all_files = [f for f in os.listdir(resource_path) if f.endswith('.csv')]
        for filename in all_files:
            file_path = os.path.join(resource_path, filename)
            try:
                df = pd.read_csv(file_path)
                if 'Symbol' in df.columns:
                    stock_symbols = df['Symbol'].apply(
                        lambda x: str(x) if str(x).endswith('.NS') else f"{str(x)}.NS"
                    ).tolist()
                    symbols.extend(stock_symbols)
                    print(f"{Fore.GREEN}✅ Loaded {len(stock_symbols)} stocks from {filename}")
            except Exception as e:
                print(f"{Fore.RED}Skipping {filename}: {e}")
    elif choice in Config.STOCK_UNIVERSES:
        filename, description = Config.STOCK_UNIVERSES[choice]
        file_path = os.path.join(resource_path, filename)
        if not os.path.exists(file_path):
            print(f"\n{Fore.RED}❌ File not found: {file_path}\n")
            return []
        df = pd.read_csv(file_path)
        if 'Symbol' not in df.columns:
            print(f"\n{Fore.RED}❌ 'Symbol' column not found in CSV\n")
            return []
        symbols = df['Symbol'].apply(
            lambda x: str(x) if str(x).endswith('.NS') else f"{str(x)}.NS"
        ).tolist()
        print(f"\n{Fore.GREEN}✅ Loaded {len(symbols)} stocks from {description}")
    else:
        print(f"\n{Fore.RED}❌ Invalid choice\n")
        return []

    if not symbols:
        print(f"\n{Fore.RED}❌ No symbols loaded\n")
        return []

    print(f"{Fore.YELLOW}⏱️  Total stocks to scan: {len(symbols)}")
    return symbols

# TWO-TIER ENHANCED MOMENTUM SCANNER (STREAMLINED)

class TwoTierEnhancedScanner:
    def __init__(self):
        self.symbols = []
        self.telegram_notifier = TelegramNotifier() if Config.SEND_TELEGRAM else None
        self.results_file = None  # Track the single output file

    def load_watchlist(self):
        """Load symbols from watchlist CSV"""
        if os.path.exists(Config.WATCHLIST_FILE):
            df = pd.read_csv(Config.WATCHLIST_FILE)
            if 'Symbol' in df.columns:
                return df['Symbol'].apply(
                    lambda x: str(x) if str(x).endswith('.NS') else f"{str(x)}.NS"
                ).tolist()
        logger.warning(f"{E_WARN} Watchlist not found: {Config.WATCHLIST_FILE}")
        return []

    def load_fallback_universe(self):
        """Load fallback universe from config"""
        resource_path = Config.RESOURCE_PATH
        if Config.STOCK_UNIVERSE == 'ALL':
            symbols = []
            all_files = [f for f in os.listdir(resource_path) if f.endswith('.csv')]
            for filename in all_files:
                file_path = os.path.join(resource_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    if 'Symbol' in df.columns:
                        stock_symbols = df['Symbol'].apply(
                            lambda x: str(x) if str(x).endswith('.NS') else f"{str(x)}.NS"
                        ).tolist()
                        symbols.extend(stock_symbols)
                        logger.info(f"Loaded {len(stock_symbols)} from {filename}")
                except Exception as e:
                    logger.warning(f"Skipping {filename}: {e}")
            return symbols
        else:
            file_path = os.path.join(resource_path, Config.DEFAULT_UNIVERSE)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'Symbol' in df.columns:
                    return df['Symbol'].apply(
                        lambda x: str(x) if str(x).endswith('.NS') else f"{str(x)}.NS"
                    ).tolist()
            logger.warning(f"Fallback file {Config.DEFAULT_UNIVERSE} not found")
            return []

    def run_weekly_scan(self):
        """TIER 1: Weekly Enhanced Momentum scan to build watchlist"""
        logger.info("=" * 100)
        logger.info(f"{E_WEEKLY} TIER 1: WEEKLY ENHANCED MOMENTUM SCAN (Build Watchlist)")
        logger.info("=" * 100)
        logger.info(f"{E_TIME} Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.symbols:
            symbols = self.symbols
            logger.info(f"{E_CHECK} Using selected universe: {len(symbols)} stocks")
        else:
            symbols = self.load_fallback_universe()
            if not symbols:
                logger.error(f"{E_ERROR} No symbols for weekly scan")
                return None
        try:
            # Import Enhanced Momentum V2 scanner
            from enhanced_momentum_V2_prefetch import SmartMoneyScanner, DataCache
            end_date = datetime.now()
            start_date = end_date - timedelta(days=Config.WEEKLY_SCAN_PERIOD)
            all_signals = []
            current_date = start_date
            scan_count = 0
            total_weeks = int((end_date - start_date).days / 7) + 1
            scan_name = 'weekly_momentum_watchlist'
            # Create data cache for speed
            logger.info(f"{E_TIME} Creating data cache for {len(symbols)} stocks...")
            data_cache = DataCache(symbols, start_date, end_date)

            # Create scanner with cache
            scanner = SmartMoneyScanner(symbols, data_cache=data_cache)

            while current_date <= end_date:
                scan_count += 1
                logger.info(f"Scanning week ending {current_date.strftime('%Y-%m-%d')} ({scan_count}/{total_weeks})")
                results = scanner.scan(end_date=current_date, show_details=False)
                if not results.empty:
                    all_signals.append(results)
                    logger.info(f"Found {len(results)} signals for {current_date.strftime('%Y-%m-%d')}")
                current_date += timedelta(days=7)

            if all_signals:
                all_results = pd.concat(all_signals, ignore_index=True)
                logger.info(f"{E_CHECK} Total raw signals collected: {len(all_results)}")

                # Apply quality filter based on config
                if Config.BREAKOUT_READY_ONLY:
                    # Only BREAKOUT_READY stocks
                    quality_filter = (
                            (all_results['Stage'] == 'BREAKOUT_READY') &
                            (all_results['Total_Score'] >= Config.MIN_TOTAL_SCORE) &
                            (all_results['RS_Score'] >= Config.MIN_RS_SCORE) &
                            (all_results['Base_Quality'] >= 4)
                    )
                    watchlist = all_results[quality_filter].copy()
                else:
                    # Both BREAKOUT_READY and ACCUMULATION
                    quality_filter = (
                            (all_results['Total_Score'] >= Config.MIN_TOTAL_SCORE) &
                            (all_results['RS_Score'] >= Config.MIN_RS_SCORE) &
                            (all_results['Base_Quality'] >= 4)
                    )
                    watchlist = all_results[quality_filter].copy()

                if watchlist.empty:
                    logger.warning(f"{E_WARN} No quality signals for watchlist - broadening filter")
                    # Broaden filter
                    broad_filter = (
                            (all_results['Total_Score'] >= (Config.MIN_TOTAL_SCORE - 2)) &
                            (all_results['RS_Score'] >= (Config.MIN_RS_SCORE - 5))
                    )
                    watchlist = all_results[broad_filter].copy()

                if watchlist.empty:
                    logger.warning(f"{E_WARN} Still no signals - empty watchlist")
                    return None

                # Sort by Total_Score and limit to top 30
                watchlist = watchlist.sort_values('Total_Score', ascending=False).head(30)

                # Save watchlist
                watchlist.to_csv(Config.WATCHLIST_FILE, index=False)
                logger.info(f"{E_SAVE} Watchlist saved: {Config.WATCHLIST_FILE} ({len(watchlist)} stocks)")

                # Log summary
                logger.info(f"Watchlist stages: {watchlist['Stage'].value_counts().to_dict()}")
                logger.info(f"Avg Total Score: {watchlist['Total_Score'].mean():.1f}")
                logger.info(f"Avg RS Score: {watchlist['RS_Score'].mean():.1f}")
                logger.info(f"Avg R:R Ratio: {watchlist['Risk_Reward'].mean():.2f}:1")

                # Send Telegram notification
                if self.telegram_notifier:
                    logger.info(f"{E_TELEGRAM} Sending Telegram notification...")
                    self.telegram_notifier.send_weekly_update(watchlist)

                return watchlist

            else:
                logger.warning(f"{E_WARN} No signals found in backtest period")
                return None

        except ImportError as e:
            logger.error(f"{E_ERROR} Could not import enhanced_momentum_V2_prefetch.py: {e}")
            logger.error(f"{E_ERROR} Make sure the file is in the same directory")
            return None
        except Exception as e:
            logger.error(f"{E_ERROR} Weekly scan failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def run_daily_scan(self):

        logger.info("=" * 100)
        logger.info(f"{E_DAILY} TIER 2: DAILY ENTRY TIMING (Streamlined Output)")
        logger.info("=" * 100)
        logger.info(f"{E_TIME} Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ========== CHECK FOR WATCHLIST ==========
        watchlist_file = Config.WATCHLIST_FILE

        if os.path.exists(watchlist_file):
            try:
                watchlist_df = pd.read_csv(watchlist_file)
                if 'Symbol' in watchlist_df.columns and len(watchlist_df) > 0:
                    logger.info(f"{E_CHECK} Found validated watchlist: {len(watchlist_df)} stocks")
                    logger.info(f"{E_CHECK} Watchlist file: {watchlist_file}")
                else:
                    logger.warning(f"{E_WARN} Watchlist file empty or invalid")
                    watchlist_file = None
            except Exception as e:
                logger.error(f"{E_ERROR} Error loading watchlist: {e}")
                watchlist_file = None
        else:
            logger.warning(f"{E_WARN} No Tier 1 watchlist found: {watchlist_file}")
            watchlist_file = None

        # ========== FALLBACK: Create temp watchlist from selected symbols ==========
        if not watchlist_file and self.symbols:
            logger.info(f"{E_WARN} Creating temporary watchlist from {len(self.symbols)} selected symbols")

            temp_watchlist = pd.DataFrame({
                'Symbol': [s.replace('.NS', '') for s in self.symbols]
            })

            temp_file = 'temp_watchlist.csv'
            temp_watchlist.to_csv(temp_file, index=False)
            watchlist_file = temp_file

            logger.info(f"{E_CHECK} Temporary watchlist created: {temp_file}")
            print(f"\n{Fore.YELLOW}{'=' * 100}")
            print(f"{Fore.YELLOW}⚠️  NO TIER 1 WATCHLIST - Using simple symbol list")
            print(f"{Fore.YELLOW}{'=' * 100}")
            print(f"{Fore.YELLOW}Technical levels will be calculated live (60-day data)")
            print(f"{Fore.YELLOW}For best results, run Tier 1 weekly scan first")
            print(f"{Fore.YELLOW}{'=' * 100}\n")

        # ========== FINAL CHECK ==========
        if not watchlist_file:
            logger.error(f"{E_ERROR} No watchlist available for Tier 2 scan")

            print(f"\n{Fore.RED}{'=' * 100}")
            print(f"{Fore.RED}❌ CANNOT RUN TIER 2 WITHOUT INPUT")
            print(f"{Fore.RED}{'=' * 100}\n")
            print(f"{Fore.YELLOW}Options:")
            print(f"{Fore.YELLOW}  1. Run Tier 1 to build validated watchlist (Recommended)")
            print(f"{Fore.YELLOW}  2. Select a stock universe for simple symbol scan")
            print(f"{Fore.YELLOW}{'=' * 100}\n")

            return None
        # ========== RUN FLEXIBLE ENTRY VALIDATOR ==========
        try:
            from one_click_entry_system import FlexibleWatchlistEntryValidator

            logger.info(f"{E_CHART} Running flexible entry validator on: {watchlist_file}")

            validator = FlexibleWatchlistEntryValidator(watchlist_file=watchlist_file)
            results = validator.validate_all()
            results = deduplicate_results(results, logger)

            if results.empty:
                logger.warning(f"{E_WARN} No signals found")
                return None
            # Deduplicate results
            logger.info(f"📊 Signals before deduplication: {len(results)}")
            results = deduplicate_results(results, logger)
            logger.info(f"✅ Signals after deduplication: {len(results)}")
            logger.info(f"   Breakdown: {results['Decision'].value_counts().to_dict()}")

            # Apply grade filter
            grade_order = {'A+': 1, 'A': 2, 'B+': 3, 'B': 4, 'C': 5}
            min_grade_rank = grade_order.get(Config.MIN_GRADE, 3)
            results['Grade_Rank'] = results['Grade'].map(grade_order)
            results = results[results['Grade_Rank'] <= min_grade_rank].drop(columns=['Grade_Rank'])

            if results.empty:
                logger.warning(f"{E_WARN} No signals meet minimum grade requirement ({Config.MIN_GRADE})")
                return None

            # STREAMLINED: Track single output file from validator
            # The validator now creates only ONE BUY_SIGNALS_*.csv file
            # Find the most recent BUY_SIGNALS file
            import glob
            buy_signal_files = sorted(glob.glob('BUY_SIGNALS_*.csv'), reverse=True)
            if buy_signal_files:
                self.results_file = buy_signal_files[0]
                logger.info(f"{E_SAVE} BUY signals saved to: {self.results_file}")
            else:
                # Fallback: create timestamp-based filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                self.results_file = f'BUY_SIGNALS_{timestamp}.csv'

            # Parse results for notifications
            buy_stocks = results[results['Decision'] == 'BUY'].to_dict('records')
            wait_stocks = results[results['Decision'] == 'WAIT'].to_dict('records')
            skip_stocks = results[results['Decision'] == 'SKIP'].to_dict('records')

            logger.info(f"{E_CHECK} Scan complete: {len(buy_stocks)} BUY, {len(wait_stocks)} WAIT, {len(skip_stocks)} SKIP")

            results_dict = {
                'buy': buy_stocks,
                'wait': wait_stocks,
                'skip': skip_stocks,
                'file': self.results_file  # Single output file
            }
            # Send Telegram notification
            if self.telegram_notifier:
                logger.info(f"{E_TELEGRAM} Sending Telegram notification...")
                self.telegram_notifier.send_daily_signals(results_dict)
            # Clean up temp file if used
            if watchlist_file == 'temp_watchlist.csv' and os.path.exists('temp_watchlist.csv'):
                os.remove('temp_watchlist.csv')
                logger.info(f"{E_CHECK} Cleaned up temporary watchlist")

            return results_dict

        except ImportError as e:
            logger.error(f"{E_ERROR} Could not import FlexibleWatchlistEntryValidator: {e}")
            logger.error(f"{E_ERROR} Make sure one_click_entry_system.py has FlexibleWatchlistEntryValidator class")
            return None
        except Exception as e:
            logger.error(f"{E_ERROR} Daily scan failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def run_with_retry(self, scan_type='daily'):
        """Run scan with retry logic"""
        for attempt in range(1, Config.MAX_RETRIES + 1):
            logger.info(f"{E_RETRY} Attempt {attempt}/{Config.MAX_RETRIES}")

            try:
                if scan_type == 'weekly':
                    results = self.run_weekly_scan()
                elif scan_type == 'daily':
                    results = self.run_daily_scan()
                else:  # combined
                    weekly_results = self.run_weekly_scan()
                    daily_results = self.run_daily_scan()
                    results = {'weekly': weekly_results, 'daily': daily_results}

                # ✅ FIX: Return results immediately on success
                if results is not None:
                    logger.info(f"{E_CHECK} Scan completed successfully!")
                    return results
                
                # If results is None, log and retry
                logger.warning(f"{E_WARN} Scan returned no results")
                
            except Exception as e:
                logger.error(f"{E_ERROR} Scan failed with error: {e}")
                import traceback
                logger.error(traceback.format_exc())

            # Retry logic
            if attempt < Config.MAX_RETRIES:
                logger.warning(f"{E_TIME} Retrying in {Config.RETRY_DELAY} seconds...")
                time.sleep(Config.RETRY_DELAY)
            else:
                logger.error(f"{E_ERROR} All retry attempts failed")

                if self.telegram_notifier:
                    error_msg = f"❌ <b>SCANNER FAILED</b>\n\n"
                    error_msg += f"Type: {scan_type.upper()}\n"
                    error_msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    error_msg += f"Attempts: {Config.MAX_RETRIES}\n\n"
                    error_msg += f"Check logs: {Config.LOG_FILE}"
                    self.telegram_notifier.send_message(error_msg)

                return None  # ✅ Changed from False to None for consistency

# MAIN EXECUTION

def main():
    print("\n" + "=" * 100)
    print(f"{E_START if Config.USE_EMOJIS else ''} AUTOMATED SCANNER V3.1.1 - STREAMLINED OUTPUT")
    print("=" * 100)
    print("\nTwo-Tier Approach:")
    print(f"  TIER 1 (Weekly): Enhanced Momentum V2 → Build breakout-ready watchlist")
    print(f"  TIER 2 (Daily): Flexible validator → Works with ANY CSV\n")

    print(f"{Fore.GREEN}{'=' * 100}")
    print(f"{Fore.GREEN}🎯 STREAMLINED CSV OUTPUT:")
    print(f"{Fore.GREEN}{'=' * 100}")
    print(f"{Fore.WHITE}  ✓ ONE comprehensive BUY signals CSV")
    print(f"{Fore.WHITE}  ✓ Optional WAIT list CSV")
    print(f"{Fore.WHITE}  ✓ No duplicate files")
    print(f"{Fore.WHITE}  ✓ Clean, organized results")
    print(f"{Fore.GREEN}{'=' * 100}\n")

    print(f"\n{Fore.CYAN}{'=' * 100}")
    print(f"{Fore.CYAN}🆕 ENHANCED MOMENTUM FEATURES:")
    print(f"{Fore.CYAN}{'=' * 100}")
    print(f"{Fore.WHITE}  ✓ Complete Wyckoff methodology (base quality, spring patterns)")
    print(f"{Fore.WHITE}  ✓ Institutional block detection")
    print(f"{Fore.WHITE}  ✓ False breakout memory (avoids burned zones)")
    print(f"{Fore.WHITE}  ✓ Risk-Reward filtering (min 2.5:1)")
    print(f"{Fore.WHITE}  ✓ Multi-timeframe relative strength")
    print(f"{Fore.WHITE}  ✓ Volume climax detection")
    print(f"{Fore.WHITE}  ✓ Data caching for 10-20x faster scans")
    print(f"{Fore.CYAN}{'=' * 100}\n")

    # Show Telegram status
    if Config.SEND_TELEGRAM and Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        print(f"{Fore.GREEN}📱 Telegram notifications: ENABLED")
        print(f"{Fore.GREEN}   Bot Token: {Config.TELEGRAM_BOT_TOKEN[:20]}...")
        print(f"{Fore.GREEN}   Chat ID: {Config.TELEGRAM_CHAT_ID}")
    else:
        print(f"{Fore.YELLOW}📱 Telegram notifications: DISABLED")

    print(f"\n{Fore.CYAN}{'=' * 100}")
    print(f"{Fore.CYAN}📊 CURRENT CONFIGURATION:")
    print(f"{Fore.CYAN}{'=' * 100}")
    print(f"{Fore.WHITE}  • Min Total Score: {Config.MIN_TOTAL_SCORE}")
    print(f"{Fore.WHITE}  • Min RS Score: {Config.MIN_RS_SCORE}")
    print(f"{Fore.WHITE}  • Min Grade (Daily): {Config.MIN_GRADE}")
    print(f"{Fore.WHITE}  • Breakout Ready Only: {Config.BREAKOUT_READY_ONLY}")
    print(f"{Fore.CYAN}{'=' * 100}\n")

    print("Options:")
    print("  [1] Run WEEKLY scan (TIER 1 - Build momentum watchlist)")
    print("  [2] Run DAILY scan (TIER 2 - Entry timing)")
    print("  [3] Run BOTH weekly and daily scans")
    print("  [4] Test Telegram connection")
    print("  [5] Exit")
    print("=" * 100)

    choice = input("\nSelect option (1-5): ").strip()

    if choice == '4':
        # Test Telegram
        print(f"\n{Fore.CYAN}Testing Telegram connection...")
        telegram = TelegramNotifier()
        if telegram.enabled:
            test_msg = f"🧪 <b>Test Message</b>\n\n"
            test_msg += f"✅ Telegram bot is working!\n"
            test_msg += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            test_msg += f"Your Enhanced Momentum scanner is ready to send notifications."

            if telegram.send_message(test_msg):
                print(f"{Fore.GREEN}✅ Test message sent successfully!")
            else:
                print(f"{Fore.RED}❌ Failed to send test message")
        else:
            print(f"{Fore.YELLOW}⚠️  Telegram is disabled in config")
        return

    if choice in ['1', '2', '3']:
        # Select universe for manual run
        symbols = select_universe()
        if not symbols:
            print(f"\n{E_CHECK if Config.USE_EMOJIS else '[OK]'} No universe selected. Exiting.")
            return

        # Pass symbols to scanner
        scanner = TwoTierEnhancedScanner()
        scanner.symbols = symbols

        if choice == '1':
            logger.info(f"{E_WEEKLY} Running WEEKLY Enhanced Momentum scan (TIER 1)... with {len(symbols)} stocks")
            scanner.run_with_retry('weekly')
        elif choice == '2':
            logger.info(f"{E_DAILY} Running DAILY entry scan (TIER 2 - STREAMLINED)... with {len(symbols)} stocks")
            scanner.run_with_retry('daily')
        elif choice == '3':
            logger.info(f"{E_START} Running COMBINED weekly and daily scans (STREAMLINED)... with {len(symbols)} stocks")
            scanner.run_with_retry('combined')
    elif choice == '5':
        print(f"\n{E_CHECK if Config.USE_EMOJIS else '[OK]'} Goodbye!\n")
    else:
        print(f"\n{E_ERROR if Config.USE_EMOJIS else '[ERROR]'} Invalid choice\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info(f"\n\n{E_CHECK} Interrupted by user")
    except Exception as e:
        logger.error(f"{E_ERROR} Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())