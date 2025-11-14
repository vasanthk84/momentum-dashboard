"""
WATCHLIST ENTRY VALIDATOR - TIER 2 SYSTEM (FLEXIBLE) - STREAMLINED CSV OUTPUT
Validates entry timing for ANY watchlist CSV file

STREAMLINED CHANGES:
- ONE comprehensive BUY signals CSV (no duplicates)
- Clean filename format
- All essential columns preserved
- Zero impact on core logic/calculations

Author: Advanced Stock Scanner Suite
Version: 5.3 - CSV Streamlined
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
from colorama import init, Fore, Back, Style

warnings.filterwarnings('ignore')
init(autoreset=True)


class FlexibleWatchlistEntryValidator:
    """
    TIER 2: Validates entry timing for ANY watchlist
    Works standalone OR with Tier 1 output
    """

    def __init__(self, watchlist_file=None):
        self.watchlist_file = watchlist_file
        self.nifty_trend = None
        self.market_strength = 0
        self.watchlist_type = None  # 'tier1', 'simple', or 'custom'

    def check_market_trend(self):
        """Check Nifty 50 trend - critical for entry decisions"""
        try:
            print(f"\n{Fore.CYAN}{'=' * 100}")
            print(f"{Fore.CYAN}📊 CHECKING OVERALL MARKET TREND (NIFTY 50)")
            print(f"{Fore.CYAN}{'=' * 100}\n")

            nifty = yf.download('^NSEI', period='5d', progress=False)

            if nifty.empty:
                print(f"{Fore.YELLOW}⚠️  Could not fetch Nifty data, assuming neutral market\n")
                self.nifty_trend = 'NEUTRAL'
                self.market_strength = 50
                return

            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)

            current_price = nifty['Close'].iloc[-1]
            sma_5 = nifty['Close'].tail(5).mean()
            price_change_1d = (nifty['Close'].iloc[-1] / nifty['Close'].iloc[-2] - 1) * 100
            price_change_5d = (nifty['Close'].iloc[-1] / nifty['Close'].iloc[0] - 1) * 100

            if current_price > sma_5 and price_change_1d > 0.5:
                self.nifty_trend = 'STRONG_BULLISH'
                self.market_strength = 90
                color = Fore.GREEN
                emoji = "🚀"
            elif current_price > sma_5 and price_change_1d > 0:
                self.nifty_trend = 'BULLISH'
                self.market_strength = 70
                color = Fore.GREEN
                emoji = "📈"
            elif abs(price_change_1d) < 0.5:
                self.nifty_trend = 'NEUTRAL'
                self.market_strength = 50
                color = Fore.YELLOW
                emoji = "➡️"
            elif current_price < sma_5 and price_change_1d < 0:
                self.nifty_trend = 'BEARISH'
                self.market_strength = 30
                color = Fore.RED
                emoji = "📉"
            else:
                self.nifty_trend = 'STRONG_BEARISH'
                self.market_strength = 10
                color = Fore.RED
                emoji = "⚠️"

        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Error checking market trend: {e}")
            self.nifty_trend = 'NEUTRAL'
            self.market_strength = 50

    def detect_watchlist_type(self, df):
        """Detect what type of watchlist CSV this is"""
        columns = set(df.columns)

        # Tier 1 watchlist signature
        tier1_cols = {'Symbol', 'Grade', 'Wyckoff_Phase', 'POC_Price', 'Value_Area_Low', 'Value_Area_High'}
        if tier1_cols.issubset(columns):
            return 'tier1'

        # Simple watchlist (just symbols, maybe grades)
        if 'Symbol' in columns and len(columns) <= 3:
            return 'simple'

        # Custom watchlist
        if 'Symbol' in columns:
            return 'custom'

        return None

    def normalize_watchlist(self, df):
        """Normalize different CSV formats to standard structure"""

        # Ensure Symbol column exists
        if 'Symbol' not in df.columns:
            # Try common alternatives
            for col in ['symbol', 'Ticker', 'ticker', 'Stock', 'stock']:
                if col in df.columns:
                    df = df.rename(columns={col: 'Symbol'})
                    break

            if 'Symbol' not in df.columns:
                raise ValueError("No 'Symbol' column found. CSV must have a column with stock symbols.")

        # Add missing columns with defaults
        defaults = {
            'Grade': 'A',
            'Wyckoff_Phase': 'N/A',
            'Current_Price': None,
            'POC_Price': None,
            'Value_Area_Low': None,
            'Value_Area_High': None,
            'Total_Score': 20,
            'RS_Score': 60
        }

        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val

        return df

    def load_watchlist(self):
        """Load watchlist from any CSV format"""
        if not self.watchlist_file:
            print(f"\n{Fore.RED}❌ No watchlist file specified")
            return None

        if not os.path.exists(self.watchlist_file):
            print(f"\n{Fore.RED}❌ Watchlist file not found: {self.watchlist_file}")
            return None

        try:
            watchlist = pd.read_csv(self.watchlist_file)

            if watchlist.empty:
                print(f"\n{Fore.RED}❌ Watchlist is empty")
                return None

            # CLEAN DEDUPLICATION
            if 'Symbol' in watchlist.columns:
                watchlist = watchlist.drop_duplicates(subset=['Symbol'], keep='first')
                watchlist = watchlist.reset_index(drop=True)

            # Detect type
            self.watchlist_type = self.detect_watchlist_type(watchlist)

            if self.watchlist_type is None:
                print(f"\n{Fore.RED}❌ Invalid watchlist format - must have 'Symbol' column")
                return None

            # Normalize
            watchlist = self.normalize_watchlist(watchlist)

            # Display info
            print(f"\n{Fore.GREEN}✅ Loaded watchlist: {len(watchlist)} stocks")
            print(f"{Fore.CYAN}📁 File: {self.watchlist_file}")
            print(f"{Fore.CYAN}📋 Type: {self.watchlist_type.upper()}")

            if self.watchlist_type == 'tier1':
                print(f"{Fore.CYAN}✨ Full Tier 1 data available (60-day validated)")
            elif self.watchlist_type == 'simple':
                print(f"{Fore.YELLOW}💡 Simple watchlist - Will calculate technical levels live")
            else:
                print(f"{Fore.CYAN}📊 Custom watchlist - Using available columns")

            print(f"\n{Fore.CYAN}{'=' * 100}")
            print(f"{Fore.CYAN}📋 WATCHLIST STOCKS:")
            print(f"{Fore.CYAN}{'=' * 100}\n")

            for idx, row in watchlist.iterrows():
                grade_str = f"Grade: {row['Grade']:<3}" if row['Grade'] != 'A' else "Grade: N/A"
                phase_str = f"Phase: {row['Wyckoff_Phase']:<3}" if row['Wyckoff_Phase'] != 'N/A' else ""

                print(f"   {idx + 1}. {row['Symbol']:<12} | {grade_str} {phase_str}".strip())

            print()
            return watchlist

        except Exception as e:
            print(f"\n{Fore.RED}❌ Error loading watchlist: {e}")
            print(f"{Fore.YELLOW}💡 Ensure CSV has at least a 'Symbol' column with stock tickers\n")
            return None

    def get_current_price_data(self, symbol):
        """Fetch current price and recent data"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"

            ticker = yf.Ticker(symbol)
            data = ticker.history(period='60d')

            if data.empty or len(data) < 2:
                return None, None, None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Calculate ATR
            high = data['High']
            low = data['Low']
            close = data['Close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            data['ATR'] = tr.rolling(window=14).mean()

            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2]
            current_volume = data['Volume'].iloc[-1]
            avg_volume_5d = data['Volume'].tail(5).mean()
            avg_volume_20d = data['Volume'].tail(20).mean()
            last_date = data.index[-1].strftime('%Y-%m-%d')

            return current_price, last_date, {
                'prev_close': prev_close,
                'volume': current_volume,
                'avg_volume_5d': avg_volume_5d,
                'avg_volume_20d': avg_volume_20d,
                'volume_ratio': current_volume / avg_volume_5d if avg_volume_5d > 0 else 1,
                'data': data
            }

        except Exception as e:
            print(f"      {Fore.RED}Error fetching {symbol}: {e}")
            return None, None, None

    def calculate_vwap(self, data):
        """Calculate current VWAP from recent data"""
        try:
            recent = data.tail(20)
            typical_price = (recent['High'] + recent['Low'] + recent['Close']) / 3
            vwap = (typical_price * recent['Volume']).sum() / recent['Volume'].sum()
            return vwap
        except:
            return None

    def calculate_volume_profile(self, data, bins=50):
        """Calculate POC and Value Area from historical data"""
        try:
            recent = data.tail(60)

            if len(recent) < 10:
                return None, None, None

            # Create price bins
            price_min = recent['Low'].min()
            price_max = recent['High'].max()
            price_bins = np.linspace(price_min, price_max, bins + 1)

            # Calculate volume at each price level
            volume_at_price, _ = np.histogram(
                recent['Close'],
                bins=price_bins,
                weights=recent['Volume']
            )

            # Point of Control (highest volume)
            poc_index = np.argmax(volume_at_price)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2

            # Value Area (70% of volume)
            total_volume = volume_at_price.sum()
            target_volume = total_volume * 0.70

            sorted_indices = np.argsort(volume_at_price)[::-1]
            cumulative_vol = np.cumsum(volume_at_price[sorted_indices])
            va_indices = sorted_indices[:np.searchsorted(cumulative_vol, target_volume) + 1]

            value_area_low = price_bins[np.min(va_indices)]
            value_area_high = price_bins[np.max(va_indices) + 1]

            return poc_price, value_area_low, value_area_high

        except:
            return None, None, None

    def enrich_watchlist_data(self, watchlist_row, market_data):
        """
        Calculate missing technical levels for non-Tier1 watchlists
        Returns enriched data dict
        """
        data = market_data['data']
        current_price = data['Close'].iloc[-1]

        # Use Tier 1 data if available, otherwise calculate
        if self.watchlist_type == 'tier1' and watchlist_row['POC_Price'] is not None:
            poc = watchlist_row['POC_Price']
            va_low = watchlist_row['Value_Area_Low']
            va_high = watchlist_row['Value_Area_High']
            signal_price = watchlist_row['Current_Price']
        else:
            # Calculate live
            poc, va_low, va_high = self.calculate_volume_profile(data)
            signal_price = data['Close'].iloc[-5] if len(data) >= 5 else current_price

            if poc is None:
                # Fallback to simple support/resistance
                poc = data['Close'].tail(20).mean()
                va_low = data['Low'].tail(20).min()
                va_high = data['High'].tail(20).max()

        # Calculate VWAP
        vwap = self.calculate_vwap(data)
        if vwap is None:
            vwap = current_price

        return {
            'signal_price': signal_price,
            'poc': poc,
            'va_low': va_low,
            'va_high': va_high,
            'vwap': vwap,
            'grade': watchlist_row['Grade'],
            'wyckoff_phase': watchlist_row['Wyckoff_Phase'],
            'total_score': watchlist_row.get('Total_Score', 20),
            'rs_score': watchlist_row.get('RS_Score', 60),
            'atr': data['ATR'].iloc[-1] if 'ATR' in data.columns else current_price * 0.02
        }

    def validate_entry_timing(self, enriched_data, current_price, market_data):
        """
        Validate if NOW is good entry timing
        Returns: decision, score, reasons
        """
        reasons = []
        buy_score = 0
        wait_factors = []
        skip_factors = []

        # Unpack enriched data
        signal_price = enriched_data['signal_price']
        grade = enriched_data['grade']
        wyckoff_phase = enriched_data['wyckoff_phase']
        poc = enriched_data['poc']
        va_low = enriched_data['va_low']
        va_high = enriched_data['va_high']
        vwap = enriched_data['vwap']
        total_score = enriched_data['total_score']
        rs_score = enriched_data['rs_score']
        atr = enriched_data['atr']

        # Current market data
        volume_ratio = market_data['volume_ratio']
        prev_close = market_data['prev_close']
        data = market_data['data']

        # Key metrics
        price_change_from_signal = (current_price - signal_price) / signal_price * 100
        vwap_deviation = ((current_price - vwap) / vwap) * 100
        distance_from_vwap = abs(vwap_deviation)
        distance_from_poc = abs(current_price - poc) / poc * 100
        in_value_area = va_low <= current_price <= va_high
        price_momentum = (current_price - prev_close) / prev_close * 100

        # Trend confirmation
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else ma20
        above_ma20 = current_price > ma20
        above_ma50 = current_price > ma50
        ma_aligned = ma20 > ma50

        # ENTRY TIMING LOGIC
        # 1. BASE QUALITY
        if grade == 'A+':
            buy_score += 25
            reasons.append("Grade A+ (high quality)")
        elif grade == 'A':
            buy_score += 20
            reasons.append("Grade A (good quality)")
        elif grade == 'B+':
            buy_score += 12
            reasons.append("Grade B+ (moderate quality)")
        elif grade == 'B':
            buy_score += 8
            reasons.append("Grade B")

        if wyckoff_phase == 'D':
            buy_score += 20
            reasons.append("Phase D - Markup phase")
        elif wyckoff_phase == 'C':
            buy_score += 15
            reasons.append("Phase C - Spring confirmed")
        elif wyckoff_phase == 'B':
            buy_score += 8
            reasons.append("Phase B - Building base")

        # 2. PRICE MOVEMENT
        if abs(price_change_from_signal) < 2:
            buy_score += 15
            reasons.append(f"Price stable ({price_change_from_signal:+.1f}%)")
        elif 2 <= price_change_from_signal <= 5:
            buy_score += 8
            reasons.append(f"Moderate gain ({price_change_from_signal:+.1f}%)")
        elif price_change_from_signal > 5:
            skip_factors.append(f"Price ran +{price_change_from_signal:.1f}% (too late)")
            buy_score -= 20
        elif price_change_from_signal < -5:
            skip_factors.append(f"Price dropped {price_change_from_signal:.1f}% (setup broke)")
            buy_score -= 15
        elif -5 <= price_change_from_signal < -2:
            wait_factors.append(f"Price weak ({price_change_from_signal:.1f}%)")
            buy_score -= 5

        # 3. NEAR ENTRY ZONE (VWAP/POC)
        if distance_from_vwap < 1:
            buy_score += 15
            reasons.append("At VWAP (perfect entry)")
        elif distance_from_vwap < 2:
            if -2 < vwap_deviation < 1:
                buy_score += 12
                reasons.append("Near VWAP (good entry)")
            else:
                buy_score += 5
        elif vwap_deviation > 3:
            wait_factors.append(f"Extended above VWAP (+{vwap_deviation:.1f}%)")
            buy_score -= 10
        elif vwap_deviation < -3:
            wait_factors.append(f"Weak vs VWAP ({vwap_deviation:.1f}%)")
            buy_score -= 8

        # 4. VALUE AREA POSITION
        if in_value_area:
            buy_score += 10
            reasons.append("In Value Area (fair price)")
        elif current_price > va_high:
            buy_score += 8
            reasons.append("Above Value Area (showing strength)")
        else:
            wait_factors.append("Below Value Area (wait for support)")
            buy_score -= 10

        # 5. VOLUME CONFIRMATION
        if volume_ratio > 1.5:
            buy_score += 12
            reasons.append(f"Strong volume ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.2:
            buy_score += 8
            reasons.append(f"Good volume ({volume_ratio:.1f}x)")
        elif volume_ratio < 0.7:
            wait_factors.append(f"Low volume ({volume_ratio:.1f}x)")
            buy_score -= 8

        # 6. MARKET ALIGNMENT
        if self.market_strength > 70:
            buy_score += 18
            reasons.append("Strong market tailwind")
        elif self.market_strength > 50:
            buy_score += 8
            reasons.append("Favorable market")
        elif self.market_strength < 40:
            wait_factors.append("Weak market environment")
            buy_score -= 12
        elif self.market_strength < 30:
            wait_factors.append("Very weak market")
            buy_score -= 18

        # 7. MOMENTUM
        if 0 < price_momentum < 2:
            buy_score += 10
            reasons.append("Positive momentum")
        elif price_momentum >= 2:
            buy_score += 5
            reasons.append("Strong momentum")
        elif price_momentum > 3:
            wait_factors.append("Running too fast")
            buy_score -= 5
        elif price_momentum < -2:
            wait_factors.append("Negative momentum")
            buy_score -= 10

        # 8. TREND CONFIRMATION
        if above_ma20 and above_ma50 and ma_aligned:
            buy_score += 15
            reasons.append("Strong uptrend (above MAs)")
        elif above_ma20 and ma_aligned:
            buy_score += 10
            reasons.append("Above MA20 (bullish)")
        elif not above_ma20:
            wait_factors.append("Below MA20 (weak trend)")
            buy_score -= 10

        # DECISION
        buy_score = max(0, min(100, buy_score))

        # Adaptive thresholds
        if self.watchlist_type == 'tier1':
            buy_threshold = 75
            wait_threshold = 50
        else:
            buy_threshold = 60
            wait_threshold = 40

        if buy_score >= buy_threshold:
            decision = 'BUY'
        elif buy_score >= wait_threshold:
            decision = 'WAIT'
        else:
            decision = 'SKIP'

        # Override for critical factors
        if len(skip_factors) >= 2:
            decision = 'SKIP'
        elif decision == 'BUY':
            if self.market_strength < 20:
                decision = 'WAIT'
                wait_factors.append("Market too weak (override)")
            elif self.market_strength < 30 and buy_score < 80:
                decision = 'WAIT'
                wait_factors.append("Weak market, marginal setup")

        return decision, buy_score, reasons, wait_factors, skip_factors, {
            'current_price': current_price,
            'signal_price': signal_price,
            'price_change_%': price_change_from_signal,
            'vwap': vwap,
            'vwap_deviation_%': vwap_deviation,
            'poc': poc,
            'va_low': va_low,
            'va_high': va_high,
            'distance_from_vwap_%': distance_from_vwap,
            'volume_ratio': volume_ratio,
            'in_value_area': in_value_area,
            'price_momentum_%': price_momentum,
            'above_ma20': above_ma20,
            'above_ma50': above_ma50,
            'ma_aligned': ma_aligned,
            'atr': atr
        }

    def calculate_entry_params(self, current_price, vwap, poc, va_low, atr):
        """Calculate entry price, stop loss, and targets"""

        # ATR-based entry range
        entry_low = current_price - (0.5 * atr)
        entry_high = current_price + (0.5 * atr)

        # Support level
        if abs(current_price - vwap) < abs(current_price - poc):
            support_level = vwap
        else:
            support_level = max(poc, va_low)

        # ATR-based stop
        stop_loss = support_level - (1.0 * atr)

        # Ensure stop isn't too tight
        min_stop = current_price - (1.5 * atr)
        stop_loss = min(stop_loss, min_stop)

        risk_per_share = current_price - stop_loss
        risk_pct = (risk_per_share / current_price) * 100

        target_1 = current_price + (risk_per_share * 2)
        target_2 = current_price + (risk_per_share * 3)

        return {
            'entry_low': entry_low,
            'entry_high': entry_high,
            'stop_loss': stop_loss,
            'risk_pct': risk_pct,
            'target_1': target_1,
            'target_2': target_2
        }

    def save_buy_signals(self, buy_stocks):
        """
        Save BUY signals to ONE comprehensive CSV file
        STREAMLINED: Single file with all essential data
        """
        if not buy_stocks:
            print(f"{Fore.YELLOW}📋 No BUY signals to save\n")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'BUY_SIGNALS_{timestamp}.csv'

        # Prepare comprehensive data for single CSV
        buy_data = []
        for stock in buy_stocks:
            buy_data.append({
                # === IDENTIFICATION ===
                'Symbol': stock['Symbol'],
                'Grade': stock['Grade'],
                'Wyckoff_Phase': stock['Wyckoff_Phase'],
                'Score': stock['Score'],

                # === PRICING ===
                'Current_Price': round(stock['Current_Price'], 2),
                'Signal_Price': round(stock['Signal_Price'], 2),
                'Price_Change_%': round(stock['Price_Change_%'], 2),
                'Price_Date': stock['Price_Date'],

                # === ENTRY PLAN ===
                'Entry_Low': round(stock['Entry_Low'], 2),
                'Entry_High': round(stock['Entry_High'], 2),
                'Stop_Loss': round(stock['Stop_Loss'], 2),
                'Risk_%': round(stock['Risk_%'], 2),

                # === TARGETS ===
                'Target_1': round(stock['Target_1'], 2),
                'Target_1_Gain_%': round(((stock['Target_1'] / stock['Current_Price'] - 1) * 100), 2),
                'Target_2': round(stock['Target_2'], 2),
                'Target_2_Gain_%': round(((stock['Target_2'] / stock['Current_Price'] - 1) * 100), 2),

                # === TECHNICAL LEVELS ===
                'VWAP': round(stock['VWAP'], 2),
                'VWAP_Deviation_%': round(stock.get('VWAP_Deviation_%', 0), 2),
                'POC': round(stock['POC'], 2),
                'Value_Area_Low': round(stock['Value_Area_Low'], 2),
                'Value_Area_High': round(stock['Value_Area_High'], 2),
                'In_Value_Area': stock['In_Value_Area'],

                # === VOLUME & TREND ===
                'Volume_Ratio': round(stock['Volume_Ratio'], 2),
                'Above_MA20': stock.get('Above_MA20', 'N/A'),
                'Above_MA50': stock.get('Above_MA50', 'N/A'),
                'MA_Aligned': stock.get('MA_Aligned', 'N/A'),

                # === MARKET CONTEXT ===
                'Market_Trend': self.nifty_trend,
                'Market_Strength': self.market_strength,

                # === KEY REASONS (Top 3) ===
                'Buy_Reason_1': stock['Reasons'][0] if len(stock['Reasons']) > 0 else '',
                'Buy_Reason_2': stock['Reasons'][1] if len(stock['Reasons']) > 1 else '',
                'Buy_Reason_3': stock['Reasons'][2] if len(stock['Reasons']) > 2 else ''
            })

        # Save comprehensive BUY signals
        buy_df = pd.DataFrame(buy_data)
        buy_df.to_csv(filename, index=False)

        print(f"{Fore.GREEN}💾 BUY signals saved to: {filename}")
        print(f"{Fore.GREEN}   📊 {len(buy_stocks)} stocks with complete entry plan\n")

        return filename

    def validate_all(self):
        """
        Main function - validate entry timing for all watchlist stocks
        STREAMLINED: Reduced CSV output, cleaner saves
        """

        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 100}")
        print(f"{Fore.CYAN}{Style.BRIGHT}🎯 WATCHLIST ENTRY VALIDATOR - TIER 2 (STREAMLINED)")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 100}\n")
        print(f"{Fore.WHITE}⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.WHITE}📋 Watchlist File: {self.watchlist_file}\n")

        # Step 1: Check market
        self.check_market_trend()

        # Step 2: Load watchlist
        watchlist = self.load_watchlist()
        if watchlist is None or watchlist.empty:
            return pd.DataFrame()

        # Step 3: Validate each watchlist stock
        print(f"{Fore.CYAN}{'=' * 100}")
        print(f"{Fore.CYAN}💡 VALIDATING ENTRY TIMING FOR WATCHLIST STOCKS")
        print(f"{Fore.CYAN}{'=' * 100}\n")

        buy_stocks = []
        wait_stocks = []
        skip_stocks = []

        for idx, stock_row in watchlist.iterrows():
            symbol = stock_row['Symbol']
            print(f"   {idx + 1}/{len(watchlist)} Checking {symbol}...", end=' ')

            # Get current price
            current_price, price_date, market_data = self.get_current_price_data(symbol)

            if current_price is None:
                print(f"{Fore.YELLOW}❌ Could not fetch")
                continue

            print(f"✅ ₹{current_price:.2f} ({price_date})")

            # Enrich data (calculate missing values if needed)
            enriched_data = self.enrich_watchlist_data(stock_row, market_data)

            # Validate entry timing
            decision, score, reasons, wait_factors, skip_factors, metrics = \
                self.validate_entry_timing(enriched_data, current_price, market_data)

            # Calculate entry parameters
            entry_params = self.calculate_entry_params(
                current_price,
                metrics['vwap'],
                metrics['poc'],
                metrics['va_low'],
                metrics['atr']
            )

            # Store result
            result = {
                'Symbol': symbol,
                'Decision': decision,
                'Score': score,
                'Grade': enriched_data['grade'],
                'Wyckoff_Phase': enriched_data['wyckoff_phase'],
                'Signal_Price': enriched_data['signal_price'],
                'Current_Price': current_price,
                'Price_Date': price_date,
                'Price_Change_%': metrics['price_change_%'],
                'VWAP': metrics['vwap'],
                'VWAP_Deviation_%': metrics['vwap_deviation_%'],
                'POC': metrics['poc'],
                'Value_Area_Low': metrics['va_low'],
                'Value_Area_High': metrics['va_high'],
                'In_Value_Area': metrics['in_value_area'],
                'Volume_Ratio': metrics['volume_ratio'],
                'Above_MA20': metrics['above_ma20'],
                'Above_MA50': metrics['above_ma50'],
                'MA_Aligned': metrics['ma_aligned'],
                'Entry_Low': entry_params['entry_low'],
                'Entry_High': entry_params['entry_high'],
                'Stop_Loss': entry_params['stop_loss'],
                'Risk_%': entry_params['risk_pct'],
                'Target_1': entry_params['target_1'],
                'Target_2': entry_params['target_2'],
                'Reasons': reasons,
                'Wait_Factors': wait_factors,
                'Skip_Factors': skip_factors
            }

            if decision == 'BUY':
                buy_stocks.append(result)
            elif decision == 'WAIT':
                wait_stocks.append(result)
            else:
                skip_stocks.append(result)

        # Sort by score
        buy_stocks.sort(key=lambda x: x['Score'], reverse=True)
        wait_stocks.sort(key=lambda x: x['Score'], reverse=True)
        skip_stocks.sort(key=lambda x: x['Score'], reverse=True)

        # ================================================================
        # DISPLAY RESULTS
        # ================================================================

        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 100}")
        print(f"{Fore.CYAN}{Style.BRIGHT}📊 ENTRY TIMING RESULTS")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 100}\n")

        # BUY STOCKS
        if buy_stocks:
            print(
                f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT} 🟢 BUY NOW - ENTER TOMORROW ({len(buy_stocks)} stocks) {Style.RESET_ALL}\n")
            print(f"{Fore.GREEN}{'─' * 100}\n")

            for i, stock in enumerate(buy_stocks, 1):
                print(f"{Fore.GREEN}{Style.BRIGHT}#{i}. {stock['Symbol']}{Style.RESET_ALL} "
                      f"{Fore.WHITE}| Grade: {stock['Grade']} | Score: {stock['Score']}/100 | Phase: {stock['Wyckoff_Phase']}")
                print(f"{Fore.WHITE}{'─' * 100}")
                print(f"{Fore.WHITE}   📋 Signal Price: ₹{stock['Signal_Price']:.2f}")
                print(f"{Fore.WHITE}   💰 Current Price: ₹{stock['Current_Price']:.2f} "
                      f"({stock['Price_Change_%']:+.1f}%) as of {stock['Price_Date']}")
                print(f"{Fore.WHITE}   📊 Volume: {stock['Volume_Ratio']:.2f}x average "
                      f"{'✅' if stock['Volume_Ratio'] > 1.5 else '⚠️'}")
                print(f"{Fore.WHITE}   📈 VWAP: ₹{stock['VWAP']:.2f} "
                      f"({'Above' if stock['VWAP_Deviation_%'] > 0 else 'Below'} by {abs(stock['VWAP_Deviation_%']):.1f}%)")
                print(f"{Fore.WHITE}   🎯 Trend: MA20={stock['Above_MA20']} | MA50={stock['Above_MA50']} | Aligned={stock['MA_Aligned']}")

                print(f"\n{Fore.GREEN}{Style.BRIGHT}   🎯 ENTRY PLAN:")
                print(f"{Fore.GREEN}   ├─ Entry Range: ₹{stock['Entry_Low']:.2f} - ₹{stock['Entry_High']:.2f} (ATR-based)")
                print(f"{Fore.GREEN}   ├─ Stop Loss:   ₹{stock['Stop_Loss']:.2f} (Risk: {stock['Risk_%']:.1f}%)")
                print(f"{Fore.GREEN}   ├─ Target 1:    ₹{stock['Target_1']:.2f} "
                      f"(+{((stock['Target_1'] / stock['Current_Price'] - 1) * 100):.1f}%)")
                print(f"{Fore.GREEN}   └─ Target 2:    ₹{stock['Target_2']:.2f} "
                      f"(+{((stock['Target_2'] / stock['Current_Price'] - 1) * 100):.1f}%)")

                print(f"\n{Fore.WHITE}   💡 Why BUY Now:")
                for reason in stock['Reasons'][:5]:
                    print(f"{Fore.WHITE}      ✓ {reason}")

                print(f"\n{Fore.YELLOW}   ⚠️  Position Size: Max 3-5% of portfolio")
                print(f"{Fore.YELLOW}   ⚠️  Set stop loss IMMEDIATELY\n")
        else:
            print(f"{Fore.YELLOW}📊 No BUY signals today - Keep monitoring\n")

        # WAIT STOCKS
        if wait_stocks:
            print(
                f"\n{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT} 🟡 WAIT - Monitor These ({len(wait_stocks)} stocks) {Style.RESET_ALL}\n")
            print(f"{Fore.YELLOW}{'─' * 100}\n")

            for stock in wait_stocks:
                print(f"{Fore.YELLOW}{Style.BRIGHT}{stock['Symbol']}{Style.RESET_ALL} "
                      f"{Fore.WHITE}| Grade: {stock['Grade']} | Score: {stock['Score']}/100")
                print(f"{Fore.WHITE}   Current: ₹{stock['Current_Price']:.2f} | "
                      f"VWAP: ₹{stock['VWAP']:.2f} ({stock['VWAP_Deviation_%']:+.1f}%) | "
                      f"POC: ₹{stock['POC']:.2f}")
                print(f"{Fore.WHITE}   Trend: MA20={stock['Above_MA20']} | MA50={stock['Above_MA50']}")

                print(f"{Fore.YELLOW}   💡 Wait for:")
                for factor in stock['Wait_Factors'][:3]:
                    print(f"{Fore.YELLOW}      • {factor}")

                print(f"{Fore.WHITE}   ✅ Enter if: Dips to ₹{stock['Entry_Low']:.2f}-₹{stock['Entry_High']:.2f}\n")

        # SKIP STOCKS
        if skip_stocks:
            print(
                f"\n{Back.RED}{Fore.WHITE}{Style.BRIGHT} 🔴 SKIP - Setup Deteriorated ({len(skip_stocks)} stocks) {Style.RESET_ALL}\n")
            print(f"{Fore.RED}{'─' * 100}\n")

            for stock in skip_stocks:
                print(f"{Fore.RED}{stock['Symbol']} {Fore.WHITE}| Score: {stock['Score']}/100")
                print(f"{Fore.RED}   ❌ Reasons:")
                for factor in stock['Skip_Factors'][:2]:
                    print(f"{Fore.RED}      • {factor}")
                print()

        # Summary
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 100}")
        print(f"{Fore.CYAN}{Style.BRIGHT}📋 SUMMARY")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 100}\n")
        print(f"{Fore.GREEN}   🟢 BUY:  {len(buy_stocks)} stocks (enter tomorrow)")
        print(f"{Fore.YELLOW}   🟡 WAIT: {len(wait_stocks)} stocks (monitor)")
        print(f"{Fore.RED}   🔴 SKIP: {len(skip_stocks)} stocks (remove from watchlist)\n")

        # ================================================================
        # STREAMLINED CSV SAVING
        # ================================================================

        print(f"{Fore.CYAN}{'=' * 100}")
        print(f"{Fore.CYAN}💾 SAVING RESULTS (STREAMLINED)")
        print(f"{Fore.CYAN}{'=' * 100}\n")

        saved_files = []

        # SAVE BUY SIGNALS (PRIMARY OUTPUT)
        if buy_stocks:
            buy_file = self.save_buy_signals(buy_stocks)
            if buy_file:
                saved_files.append(buy_file)

        # SAVE WAIT LIST (OPTIONAL)
        if wait_stocks:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            wait_filename = f'WAIT_LIST_{timestamp}.csv'
            wait_df = pd.DataFrame([{
                'Symbol': r['Symbol'],
                'Grade': r['Grade'],
                'Score': r['Score'],
                'Current_Price': round(r['Current_Price'], 2),
                'VWAP': round(r['VWAP'], 2),
                'VWAP_Dev_%': round(r['VWAP_Deviation_%'], 2),
                'POC': round(r['POC'], 2),
                'Entry_Low': round(r['Entry_Low'], 2),
                'Entry_High': round(r['Entry_High'], 2),
                'Above_MA20': r['Above_MA20'],
                'Wait_For': ' | '.join(r['Wait_Factors'][:3])
            } for r in wait_stocks])
            wait_df.to_csv(wait_filename, index=False)
            print(f"{Fore.YELLOW}💾 WAIT list saved to: {wait_filename}\n")
            saved_files.append(wait_filename)

        # CREATE RESULTS DATAFRAME FOR RETURN (used by two_tier scanner)
        all_results = buy_stocks + wait_stocks + skip_stocks
        if all_results:
            results_df = pd.DataFrame([{
                'Symbol': r['Symbol'],
                'Decision': r['Decision'],
                'Score': r['Score'],
                'Grade': r['Grade'],
                'Signal_Price': round(r['Signal_Price'], 2),
                'Current_Price': round(r['Current_Price'], 2),
                'Price_Date': r['Price_Date'],
                'Price_Change_%': round(r['Price_Change_%'], 2),
                'VWAP_Deviation_%': round(r['VWAP_Deviation_%'], 2),
                'Entry_Low': round(r['Entry_Low'], 2),
                'Entry_High': round(r['Entry_High'], 2),
                'Stop_Loss': round(r['Stop_Loss'], 2),
                'Risk_%': round(r['Risk_%'], 2),
                'Target_1': round(r['Target_1'], 2),
                'Target_2': round(r['Target_2'], 2),
                'Wyckoff_Phase': r['Wyckoff_Phase'],
                'Above_MA20': r['Above_MA20'],
                'Above_MA50': r['Above_MA50'],
                'Volume_Ratio': round(r['Volume_Ratio'], 2)
            } for r in all_results])
        else:
            results_df = pd.DataFrame()

        if saved_files:
            print(f"{Fore.CYAN}📁 Files saved:")
            for f in saved_files:
                print(f"{Fore.CYAN}   • {f}")
            print()

        return results_df


def select_watchlist_file():
    """Interactive file selection with multiple options"""

    print(f"\n{Fore.CYAN}{'=' * 100}")
    print(f"{Fore.CYAN}📁 SELECT WATCHLIST FILE")
    print(f"{Fore.CYAN}{'=' * 100}\n")

    # Look for available files
    available_files = []

    # Check for Tier 1 watchlists
    if os.path.exists('watchlist_momentum_current.csv'):
        available_files.append(('watchlist_momentum_current.csv', 'Current Tier 1 Watchlist (Recommended)'))

    if os.path.exists('watchlist_current.csv'):
        available_files.append(('watchlist_current.csv', 'Current Tier 1 Watchlist'))

    # Check for timestamped watchlists
    import glob
    tier1_files = glob.glob('watchlist_momentum_*.csv') + glob.glob('institutional_buildup_*.csv')
    for f in tier1_files[:3]:  # Show up to 3 recent files
        if f not in [x[0] for x in available_files]:
            file_date = f.split('_')[-1].replace('.csv', '')
            available_files.append((f, f'Tier 1 Scan ({file_date})'))

    # Check for other CSV files
    other_csvs = [f for f in glob.glob('*.csv') if 'watchlist' not in f.lower()
                  and 'institutional' not in f.lower() and 'BUY_SIGNAL' not in f and 'WAIT_LIST' not in f]
    for f in other_csvs[:3]:  # Show up to 3
        available_files.append((f, 'Custom CSV'))

    # Display options
    if available_files:
        print(f"{Fore.WHITE}Available watchlist files:\n")
        for i, (filename, description) in enumerate(available_files, 1):
            print(f"  [{i}] {description}")
            print(f"      📄 {filename}\n")

        print(f"  [0] Enter custom file path")
        print(f"{Fore.CYAN}{'=' * 100}\n")

        choice = input(f"{Fore.CYAN}Select file (0-{len(available_files)}): ").strip()

        try:
            choice_num = int(choice)
            if choice_num == 0:
                custom_path = input(f"{Fore.CYAN}Enter CSV file path: ").strip()
                return custom_path if custom_path else None
            elif 1 <= choice_num <= len(available_files):
                return available_files[choice_num - 1][0]
            else:
                print(f"{Fore.RED}❌ Invalid choice")
                return None
        except ValueError:
            print(f"{Fore.RED}❌ Invalid input")
            return None
    else:
        print(f"{Fore.YELLOW}⚠️  No watchlist files found in current directory\n")
        custom_path = input(f"{Fore.CYAN}Enter CSV file path (or press Enter to exit): ").strip()
        return custom_path if custom_path else None


def main():
    """Main execution"""

    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 100}")
    print(f"{Fore.CYAN}{Style.BRIGHT}🎯 WATCHLIST ENTRY VALIDATOR (TIER 2 - STREAMLINED)")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 100}\n")
    print(f"{Fore.WHITE}This validator works with ANY watchlist CSV:")
    print(f"{Fore.WHITE}  ✓ Tier 1 watchlists (full institutional data)")
    print(f"{Fore.WHITE}  ✓ Simple CSVs (just symbols)")
    print(f"{Fore.WHITE}  ✓ Custom CSVs (your own format)")
    print(f"{Fore.WHITE}  ✓ Calculates missing technical levels automatically\n")

    print(f"{Fore.GREEN}🎯 STREAMLINED OUTPUT:")
    print(f"{Fore.GREEN}  ✓ ONE comprehensive BUY signals CSV")
    print(f"{Fore.GREEN}  ✓ Optional WAIT list CSV")
    print(f"{Fore.GREEN}  ✓ No duplicate files")
    print(f"{Fore.GREEN}  ✓ Clean, organized results\n")

    # Check time
    now = datetime.now()
    if now.hour < 15 or (now.hour == 15 and now.minute < 30):
        if now.weekday() < 5:
            print(f"{Fore.YELLOW}⚠️  Market hasn't closed yet")
            print(f"{Fore.YELLOW}   Best results after 3:30 PM\n")

            proceed = input(f"{Fore.YELLOW}Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return

    # File selection
    watchlist_file = select_watchlist_file()

    if not watchlist_file:
        print(f"\n{Fore.YELLOW}👋 No file selected. Exiting...\n")
        return

    # Run validation
    print(f"\n{Fore.CYAN}{'=' * 100}")
    print(f"{Fore.CYAN}🚀 STARTING VALIDATION")
    print(f"{Fore.CYAN}{'=' * 100}")

    validator = FlexibleWatchlistEntryValidator(watchlist_file)
    results = validator.validate_all()

    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 100}")
    print(f"{Fore.CYAN}{Style.BRIGHT}✅ VALIDATION COMPLETE!")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 100}\n")

    if not results.empty:
        buy_count = len(results[results['Decision'] == 'BUY'])
        if buy_count > 0:
            print(f"{Fore.GREEN}{Style.BRIGHT}🎉 {buy_count} STOCKS READY TO BUY TOMORROW!")
            print(f"{Fore.GREEN}Check the detailed entry plans above.")
            print(f"{Fore.GREEN}BUY signals saved to comprehensive CSV file.\n")
        else:
            print(f"{Fore.YELLOW}⏳ No BUY signals today - Stocks not at entry points yet.\n")
    else:
        print(f"{Fore.YELLOW}💡 No results. Check if watchlist is valid.\n")

    print(f"{Fore.CYAN}{'=' * 100}")
    print(f"{Fore.CYAN}📖 WORKFLOW OPTIONS:")
    print(f"{Fore.CYAN}{'=' * 100}\n")
    print(f"{Fore.WHITE}OPTION 1 - Two-Tier System (Recommended):")
    print(f"{Fore.WHITE}  • Sunday: Run Tier 1 deep scan → Creates validated watchlist")
    print(f"{Fore.WHITE}  • Daily:  Run this Tier 2 validator → Checks entry timing")
    print(f"{Fore.WHITE}  • Benefit: Thorough 60-day analysis + daily entry precision\n")

    print(f"{Fore.WHITE}OPTION 2 - Standalone (Quick):")
    print(f"{Fore.WHITE}  • Create simple CSV with stock symbols")
    print(f"{Fore.WHITE}  • Run this validator with your CSV")
    print(f"{Fore.WHITE}  • Benefit: Fast daily scans without full analysis")
    print(f"{Fore.WHITE}  • Tradeoff: Less historical context\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n{Fore.YELLOW}💡 CSV FORMAT HELP:")
        print(f"{Fore.YELLOW}Your CSV must have at least these columns:")
        print(f"{Fore.YELLOW}  • Symbol (required) - Stock ticker (e.g., RELIANCE or RELIANCE.NS)")
        print(f"{Fore.YELLOW}  • Grade (optional) - Quality grade (A+, A, B+, B, C)")
        print(f"{Fore.YELLOW}  • Other columns are optional\n")
