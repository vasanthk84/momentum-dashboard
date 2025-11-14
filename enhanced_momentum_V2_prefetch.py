import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added this import

warnings.filterwarnings('ignore')


class DataCache:
    """Pre-fetch and cache historical data for all symbols to avoid repeated yfinance calls."""

    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols + ['^NSEI'] if '^NSEI' not in symbols else symbols
        # Buffer for 6mo period + extra days for indicator calculation
        self.start_fetch = start_date - timedelta(days=250)
        self.end_fetch = end_date
        self.cache = {}  # symbol: full_df
        self._fetch_all()

    def _fetch_single(self, symbol):
        """Fetch data for one symbol (for parallel use)."""
        try:
            stock = yf.Ticker(symbol)
            # Use 'Asia/Kolkata' for .NS symbols for consistent timezone handling
            start_fetch_ts = pd.Timestamp(self.start_fetch, tz='Asia/Kolkata')
            end_fetch_ts = pd.Timestamp(self.end_fetch, tz='Asia/Kolkata')
            df = stock.history(start=start_fetch_ts, end=end_fetch_ts)
            if len(df) >= 50:  # Minimum data check
                return symbol, df
        except Exception:
            pass
        return symbol, None

    def _fetch_all(self):
        """Parallel fetch all data."""
        print("📡 Pre-fetching historical data for all stocks...")
        # Increased workers for faster fetching
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(self._fetch_single, symbol) for symbol in self.symbols]
            for future in as_completed(futures):
                symbol, df = future.result()
                if df is not None:
                    self.cache[symbol] = df
                    print(f"   ✓ Cached {symbol} ({len(df)} rows)", end='\r')
        print("\n✅ Data caching complete. Cached stocks:", len(self.cache))

    def get_slice(self, symbol, end_date):
        """Get data slice up to end_date."""
        if symbol not in self.cache:
            return None
        df = self.cache[symbol]

        # Use a rolling 6-month period for consistency with the original period='6mo'
        slice_end = pd.Timestamp(end_date, tz='Asia/Kolkata')
        slice_start = slice_end - timedelta(days=180)  # 6 months lookback approximation

        # Filter the cached data
        sliced_df = df[(df.index.tz_convert('Asia/Kolkata').date >= slice_start.date()) &
                       (df.index.tz_convert('Asia/Kolkata').date <= slice_end.date())].copy()

        if len(sliced_df) < 50:
            return None

        # Ensure the index is a simple date/datetime for indicator calculation
        return sliced_df.tz_localize(None).reset_index().set_index('Date')

    def get_custom_slice(self, symbol, start_date, end_date):
        """Get custom slice between start_date and end_date."""
        if symbol not in self.cache:
            return None
        df = self.cache[symbol]

        slice_start = pd.Timestamp(start_date, tz='Asia/Kolkata')
        slice_end = pd.Timestamp(end_date, tz='Asia/Kolkata')

        sliced_df = df[(df.index.tz_convert('Asia/Kolkata') >= slice_start) &
                       (df.index.tz_convert('Asia/Kolkata') <= slice_end)].copy()

        if len(sliced_df) < 30:
            return None

        return sliced_df.tz_localize(None).reset_index().set_index('Date')


class SmartMoneyScanner:
    def __init__(self, symbols_list, data_cache=None):
        """
        Initialize scanner with list of stock symbols
        symbols_list: List of NSE stock symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])
        """
        self.symbols = symbols_list
        self.results = []
        self.data_cache = data_cache

    def fetch_data(self, symbol, period='6mo', end_date=None):
        """Fetch historical data for a symbol"""
        if self.data_cache:
            return self.data_cache.get_slice(symbol, end_date or datetime.now().date())

        try:
            stock = yf.Ticker(symbol)
            if end_date:
                start_date = end_date - timedelta(days=180)
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)
            if len(df) < 50:
                return None
            return df
        except Exception as e:
            return None

    def calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return obv

    def calculate_ad_line(self, df):
        """Calculate Accumulation/Distribution Line"""
        ad_line = []
        ad_value = 0

        for i in range(len(df)):
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            close = df['Close'].iloc[i]
            volume = df['Volume'].iloc[i]

            if high != low:
                mfm = ((close - low) - (high - close)) / (high - low)
                ad_value += mfm * volume
            ad_line.append(ad_value)

        return ad_line

    def calculate_mfi(self, df, period=14):
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        positive_flow = []
        negative_flow = []

        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        positive_flow = [0] + positive_flow
        negative_flow = [0] + negative_flow

        mfi_values = []
        for i in range(period, len(df)):
            pos_sum = sum(positive_flow[i - period + 1:i + 1])
            neg_sum = sum(negative_flow[i - period + 1:i + 1])

            if neg_sum == 0:
                mfi_values.append(100)
            else:
                money_ratio = pos_sum / neg_sum
                mfi = 100 - (100 / (1 + money_ratio))
                mfi_values.append(mfi)

        return [np.nan] * period + mfi_values

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range for volatility"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def analyze_base_quality(self, df):
        """NEW: Rate the quality of consolidation base"""
        base_score = 0
        signals = []
        recent_50 = df.tail(50)

        # 1. Flat/tight base (best)
        volatility = recent_50['Close'].std() / recent_50['Close'].mean()
        if volatility < 0.03:  # Very tight
            base_score += 3
            signals.append("Extremely tight base (VCP)")
        elif volatility < 0.05:
            base_score += 2
            signals.append("Tight consolidation")

        # 2. Decreasing volume during consolidation (coiling)
        early_vol = recent_50['Volume'].iloc[:25].mean()
        late_vol = recent_50['Volume'].iloc[25:].mean()
        if late_vol < early_vol * 0.8:
            base_score += 2
            signals.append("Volume coiling (dry up)")

        # 3. Multiple tests of resistance (strength building)
        resistance = recent_50['High'].max()
        tests = (recent_50['High'] > resistance * 0.98).sum()
        if 3 <= tests <= 5:
            base_score += 2
            signals.append(f"Resistance tested {tests}x")

        return base_score, signals

    def detect_institutional_activity(self, df):
        """NEW: Detect large block trades suggesting institutional buying"""
        score = 0
        signals = []
        recent = df.tail(20)

        # Find unusually large volume days with price support
        avg_vol = df['Volume'].tail(60).mean()

        block_trades = 0
        for idx in range(len(recent)):
            day = recent.iloc[idx]
            if day['Volume'] > avg_vol * 2:  # 2x volume spike
                # Check if price held (didn't crash on volume)
                if day['High'] != day['Low']:
                    close_position = (day['Close'] - day['Low']) / (day['High'] - day['Low'])
                    if close_position > 0.6:  # Closed in upper 40%
                        block_trades += 1

        if block_trades >= 2:
            score = min(block_trades, 3)
            signals.append(f"Institutional blocks detected ({block_trades})")

        return score, signals

    def check_false_breakout_history(self, df):
        """NEW: Penalize stocks with recent failed breakouts"""
        last_60 = df.tail(60)

        # Look for price spikes that failed
        for i in range(10, len(last_60) - 5):
            window = last_60.iloc[i - 10:i]
            resistance = window['High'].max()

            if last_60['High'].iloc[i] > resistance * 1.01:  # Breakout
                # Did it fail? (Close back below within 5 days)
                next_5 = last_60.iloc[i:min(i + 5, len(last_60))]
                if len(next_5) > 0 and (next_5['Close'] < resistance).any():
                    return -3, ["Recent false breakout (penalty)"]

        return 0, []

    def calculate_risk_reward(self, df):
        """NEW: Only take setups with good risk-reward ratio"""
        current_price = df['Close'].iloc[-1]
        atr = self.calculate_atr(df, 14).iloc[-1]

        # Risk: 2x ATR
        stop = current_price - (2 * atr)
        risk = current_price - stop

        # Reward: Distance to next major resistance
        last_60 = df.tail(60)
        resistance = last_60['High'].max()

        # Look for next level beyond current resistance
        breakout_target = resistance * 1.08  # 8% above resistance
        reward = breakout_target - current_price

        rr_ratio = reward / risk if risk > 0 else 0

        return rr_ratio, stop

    def detect_volume_climax(self, df):
        """NEW: Detect volume climax selloff followed by recovery (spring)"""
        last_30 = df.tail(30)
        avg_vol = df['Volume'].tail(90).mean()

        for i in range(5, len(last_30) - 3):
            day = last_30.iloc[i]

            # Volume spike 3x+ with price down
            if day['Volume'] > avg_vol * 3 and day['Close'] < day['Open']:
                # Check recovery next 3 days
                next_3 = last_30.iloc[i + 1:i + 4]
                if len(next_3) >= 3 and (next_3['Close'] > day['Close']).all():
                    return 3, ["Volume climax + recovery (spring)"]

        return 0, []

    def fetch_nifty_data(self, start, end):
        if self.data_cache:
            return self.data_cache.get_custom_slice('^NSEI', start, end)
        else:
            return None

    def calculate_relative_strength(self, df, end_date=None):
        """ENHANCED: Better RS using multiple timeframes"""
        if len(df) < 60:
            return 50

        try:
            nifty_df = self.fetch_nifty_data(df.index[-60].date(), df.index[-1].date())
            if nifty_df is None or len(nifty_df) < 30:
                # Fallback to original fetch if cache miss
                nifty = yf.Ticker('^NSEI')
                nifty_df = nifty.history(start=df.index[-60], end=df.index[-1])
                if len(nifty_df) < 30:
                    return 50

            # Calculate RS for 3 timeframes
            timeframes = [10, 30, 60]
            rs_scores = []

            for tf in timeframes:
                if len(df) >= tf and len(nifty_df) >= tf:
                    stock_change = (df['Close'].iloc[-1] - df['Close'].iloc[-tf]) / df['Close'].iloc[-tf]
                    nifty_change = (nifty_df['Close'].iloc[-1] - nifty_df['Close'].iloc[-tf]) / nifty_df['Close'].iloc[
                        -tf]

                    rs_ratio = stock_change / nifty_change if nifty_change != 0 else 1
                    rs_scores.append(rs_ratio)

            # Weighted average (recent = more weight)
            if len(rs_scores) == 3:
                weighted_rs = (rs_scores[0] * 0.5 + rs_scores[1] * 0.3 + rs_scores[2] * 0.2)
            else:
                weighted_rs = rs_scores[0] if rs_scores else 1

            # Normalize to 0-100
            rs_final = min(100, max(0, 50 + (weighted_rs - 1) * 50))

            # Bonus: Consistent outperformance
            if len(rs_scores) > 0 and all(r > 1 for r in rs_scores):
                rs_final = min(100, rs_final + 10)

            return rs_final
        except:
            # Fallback to simple calculation
            if len(df) >= 30:
                price_30d_ago = df['Close'].iloc[-30]
                current_price = df['Close'].iloc[-1]
                pct_change = ((current_price - price_30d_ago) / price_30d_ago) * 100

                if pct_change > 20:
                    return 90
                elif pct_change > 10:
                    return 75
                elif pct_change > 5:
                    return 60
                elif pct_change > 0:
                    return 50
                else:
                    return 40
            return 50

    def detect_breakout_setup(self, df):
        """
        Detect if stock is in ideal breakout setup position
        Returns: (is_setup, setup_strength, signals)
        """
        recent = df.tail(5)
        last_20 = df.tail(20)

        setup_score = 0
        setup_signals = []

        # 1. Volume Expansion in last 3 days
        recent_vol = recent['Volume'].tail(3).mean()
        prior_vol = df['Volume'].tail(20).head(17).mean()
        if recent_vol > prior_vol * 1.5:
            setup_score += 3
            setup_signals.append(f"Volume expanding ({recent_vol / prior_vol:.2f}x)")

        # 2. Price approaching resistance but not extended
        resistance = last_20['High'].max()
        current_price = df['Close'].iloc[-1]
        distance_to_resistance = ((resistance - current_price) / current_price) * 100

        if 0 < distance_to_resistance < 3:  # Within 3% of breakout
            setup_score += 4
            setup_signals.append(f"Near resistance ({distance_to_resistance:.2f}% away)")

        # 3. Tightening price action (Volatility Contraction Pattern - VCP)
        atr_20 = self.calculate_atr(df, 20)
        atr_current = atr_20.iloc[-1]
        atr_avg = atr_20.tail(60).mean()

        if atr_current < atr_avg * 0.7:  # ATR contracted by 30%+
            setup_score += 3
            setup_signals.append("Volatility contraction (VCP)")

        # 4. Higher lows in last 10 days
        last_10_lows = df['Low'].tail(10).values
        higher_lows_count = sum([1 for i in range(1, len(last_10_lows))
                                 if last_10_lows[i] >= last_10_lows[i - 1] * 0.98])

        if higher_lows_count >= 6:  # At least 60% higher lows
            setup_score += 2
            setup_signals.append("Strong higher lows structure")

        # 5. Price above key moving averages
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]

        if current_price > ma20 > ma50:
            setup_score += 2
            setup_signals.append("Price > MA20 > MA50 (bullish)")

        return setup_score >= 8, setup_score, setup_signals

    def detect_accumulation(self, df, symbol, end_date=None):
        """Enhanced accumulation detection with all new filters"""
        if df is None or len(df) < 50:
            return None

        # Calculate indicators
        df['OBV'] = self.calculate_obv(df)
        df['AD_Line'] = self.calculate_ad_line(df)
        df['MFI'] = self.calculate_mfi(df)
        df['Volume_MA'] = df['Volume'].rolling(window=50).mean()
        df['Price_MA20'] = df['Close'].rolling(window=20).mean()
        df['Price_MA50'] = df['Close'].rolling(window=50).mean()
        df['ATR'] = self.calculate_atr(df, 14)

        # Recent data
        recent = df.tail(30)
        last_60 = df.tail(60)

        # NOISE FILTER 1: Reject stocks with declining trend
        ma50_slope = (df['Price_MA50'].iloc[-1] - df['Price_MA50'].iloc[-10]) / df['Price_MA50'].iloc[-10] * 100
        if ma50_slope < -5:
            return None

        # NOISE FILTER 2: Reject very low liquidity stocks
        avg_daily_value = (df['Close'] * df['Volume']).tail(20).mean()
        if avg_daily_value < 10_000_000:
            return None

        # NEW FILTERS - QUALITY CHECKS
        base_quality_score, base_signals = self.analyze_base_quality(df)
        if base_quality_score < 4:  # Minimum quality threshold
            return None

        inst_score, inst_signals = self.detect_institutional_activity(df)
        false_breakout_penalty, fb_signals = self.check_false_breakout_history(df)

        # Calculate risk-reward
        rr_ratio, stop_price = self.calculate_risk_reward(df)
        if rr_ratio < 2.5:  # Minimum 2.5:1 R:R
            return None

        volume_climax_score, vc_signals = self.detect_volume_climax(df)

        # Enhanced RS check
        rs_score = self.calculate_relative_strength(df, end_date)
        if rs_score < 60:  # Only take stocks outperforming market
            return None

        # Accumulation scoring
        score = 0
        signals = []

        # 1. Volume Analysis
        avg_volume = df['Volume_MA'].iloc[-1]
        recent_volume = recent['Volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0

        price_change = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0] * 100

        if volume_ratio > 1.3 and abs(price_change) < 5:
            score += 3
            signals.append(f"High volume ({volume_ratio:.2f}x) + low volatility")

        # 2. OBV Divergence
        obv_trend = (recent['OBV'].iloc[-1] - recent['OBV'].iloc[0]) / abs(recent['OBV'].iloc[0]) if recent['OBV'].iloc[
                                                                                                         0] != 0 else 0

        if obv_trend > 0.05 and abs(price_change) < 5:
            score += 3
            signals.append(f"OBV rising ({obv_trend * 100:.2f}%) in consolidation")

        # 3. A/D Line Divergence
        ad_trend = (recent['AD_Line'].iloc[-1] - recent['AD_Line'].iloc[0]) / abs(recent['AD_Line'].iloc[0]) if \
        recent['AD_Line'].iloc[0] != 0 else 0

        if ad_trend > 0.05 and price_change < 5:
            score += 2
            signals.append(f"A/D accumulation ({ad_trend * 100:.2f}%)")

        # 4. MFI Analysis
        recent_mfi = recent['MFI'].dropna()
        avg_mfi = 0
        if len(recent_mfi) > 0:
            avg_mfi = recent_mfi.mean()
            if 50 < avg_mfi < 70:
                score += 2
                signals.append(f"MFI optimal range ({avg_mfi:.2f})")

        # 5. Tight Consolidation
        high_low_range = (recent['High'].max() - recent['Low'].min()) / recent['Close'].iloc[0] * 100
        if 5 < high_low_range < 15:
            score += 2
            signals.append(f"Healthy base ({high_low_range:.2f}% range)")

        # 6. Volume on Down Days (Absorption)
        down_days = recent[recent['Close'] < recent['Open']]
        if len(down_days) > 0:
            down_volume_avg = down_days['Volume'].mean()
            if down_volume_avg > avg_volume * 1.3:
                score += 3
                signals.append("Strong absorption on dips")

        # 7. Pocket Pivot Detection
        last_5_days = df.tail(5)
        for i in range(len(last_5_days)):
            day = last_5_days.iloc[i]
            if day['Close'] > day['Open']:
                last_10 = df.tail(10)
                down_days_vol = last_10[last_10['Close'] < last_10['Open']]['Volume']
                if len(down_days_vol) > 0 and day['Volume'] > down_days_vol.max():
                    score += 3
                    signals.append("Pocket pivot detected")
                    break

        # 8. Spring Pattern
        if len(last_60) > 20:
            support_level = last_60['Low'].iloc[:40].min()
            recent_low = recent['Low'].min()
            current_price = df['Close'].iloc[-1]

            if recent_low < support_level * 0.98 and current_price > support_level * 1.02:
                score += 3
                signals.append("Spring/shakeout recovered")

        # ADD NEW SCORES
        score += base_quality_score
        score += inst_score
        score += false_breakout_penalty
        score += volume_climax_score

        # Combine all signals
        all_signals = signals + base_signals + inst_signals + fb_signals + vc_signals

        # Check for breakout setup
        is_breakout_setup, setup_score, setup_signals = self.detect_breakout_setup(df)

        # Enhanced threshold: Need both accumulation AND breakout setup
        if score >= 10 and is_breakout_setup:
            current_price = df['Close'].iloc[-1]

            return {
                'Symbol': symbol,
                'Accumulation_Score': score,
                'Breakout_Score': setup_score,
                'Total_Score': score + setup_score,
                'Current_Price': current_price,
                'Stop_Price': stop_price,
                'Risk_Reward': rr_ratio,
                'Price_Change_30d': price_change,
                'Volume_Ratio': volume_ratio,
                'OBV_Trend': obv_trend * 100,
                'MFI': avg_mfi,
                'ATR': df['ATR'].iloc[-1],
                'ATR_Ratio': (df['ATR'].iloc[-1] / df['ATR'].tail(60).mean()),
                'RS_Score': rs_score,
                'Base_Quality': base_quality_score,
                'Distance_to_Resistance': self.get_resistance_distance(df),
                'Signals': all_signals + setup_signals,
                'Stage': 'BREAKOUT_READY' if setup_score >= 10 else 'ACCUMULATION'
            }

        return None

    def get_resistance_distance(self, df):
        """Calculate distance to nearest resistance"""
        last_60 = df.tail(60)
        resistance = last_60['High'].max()
        current_price = df['Close'].iloc[-1]
        return ((resistance - current_price) / current_price) * 100

    def scan(self, end_date=None, show_details=False):
        """Scan all symbols for accumulation patterns"""
        self.results = []

        for i, symbol in enumerate(self.symbols, 1):
            if not show_details:
                print(f"Processing {i}/{len(self.symbols)}: {symbol}", end='\r')

            df = self.fetch_data(symbol, end_date=end_date)
            result = self.detect_accumulation(df, symbol, end_date)

            if result:
                self.results.append(result)

        if not show_details:
            print(" " * 100, end='\r')

        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df = results_df.sort_values('Total_Score', ascending=False)
            return results_df
        else:
            return pd.DataFrame()


class Backtester:
    def __init__(self, symbols_list, start_date, end_date):
        self.symbols = symbols_list
        self.start_date = start_date
        self.end_date = end_date
        self.trades = []
        self.data_cache = DataCache(self.symbols, self.start_date, self.end_date)

    def run_backtest(self, holding_periods=[10, 20, 30, 60]):
        print(f"\n{'=' * 100}")
        print(f"🔍 ENHANCED BACKTEST: {self.start_date.strftime('%Y-%m-%d')} TO {self.end_date.strftime('%Y-%m-%d')}")
        print(f"{'=' * 100}\n")

        current_date = self.start_date
        test_dates = []

        while current_date <= self.end_date:
            test_dates.append(current_date)
            current_date += timedelta(days=7)

        total_signals = 0
        breakout_ready_count = 0

        for test_idx, scan_date in enumerate(test_dates, 1):
            print(f"\n{'─' * 100}")
            print(f"📅 Enhanced Momentum V2 Scanning: {scan_date.strftime('%Y-%m-%d')} [{test_idx}/{len(test_dates)}]")
            print(f"{'─' * 100}")

            scanner = SmartMoneyScanner(self.symbols, data_cache=self.data_cache)
            results = scanner.scan(end_date=scan_date, show_details=False)

            if not results.empty:
                breakout_ready = results[results['Stage'] == 'BREAKOUT_READY']
                accumulation = results[results['Stage'] == 'ACCUMULATION']

                if len(breakout_ready) > 0:
                    print(f"\n🚀 BREAKOUT READY ({len(breakout_ready)} stocks):")
                    for idx, signal in breakout_ready.iterrows():
                        print(f"  🔥 {signal['Symbol']:<20} Total: {signal['Total_Score']:<3} "
                              f"(Accum: {signal['Accumulation_Score']} + Breakout: {signal['Breakout_Score']})")
                        print(
                            f"     Price: ₹{signal['Current_Price']:.2f} | Stop: ₹{signal['Stop_Price']:.2f} | R:R: {signal['Risk_Reward']:.2f}:1")
                        print(
                            f"     RS: {signal['RS_Score']:.0f} | Base Quality: {signal['Base_Quality']} | Volume: {signal['Volume_Ratio']:.2f}x")
                        print()
                    breakout_ready_count += len(breakout_ready)

                if len(accumulation) > 0:
                    print(f"📊 Accumulation Phase ({len(accumulation)} stocks):")
                    for idx, signal in accumulation.head(3).iterrows():
                        print(f"  • {signal['Symbol']:<20} Score: {signal['Accumulation_Score']}")
                    print()

                total_signals += len(results)

                # Calculate returns
                for _, signal in results.iterrows():
                    symbol = signal['Symbol']
                    entry_price = signal['Current_Price']
                    stop_price = signal['Stop_Price']

                    future_data = self.fetch_future_data(symbol, scan_date, max(holding_periods))

                    if future_data is not None:
                        trade_result = {
                            'Signal_Date': scan_date,
                            'Symbol': symbol,
                            'Stage': signal['Stage'],
                            'Accumulation_Score': signal['Accumulation_Score'],
                            'Breakout_Score': signal['Breakout_Score'],
                            'Total_Score': signal['Total_Score'],
                            'Entry_Price': entry_price,
                            'Stop_Price': stop_price,
                            'Risk_Reward': signal['Risk_Reward'],
                            'RS_Score': signal['RS_Score'],
                            'Base_Quality': signal['Base_Quality'],
                            'Distance_to_Resistance': signal['Distance_to_Resistance']
                        }

                        for period in holding_periods:
                            if period < len(future_data):
                                period_data = future_data.iloc[:period + 1]

                                # Check if stop hit
                                stop_hit = (period_data['Low'] < stop_price).any()

                                if stop_hit:
                                    exit_price = stop_price
                                    returns = ((exit_price - entry_price) / entry_price) * 100
                                    trade_result[f'Return_{period}d'] = returns
                                    trade_result[f'Stop_Hit_{period}d'] = True
                                else:
                                    exit_price = future_data['Close'].iloc[period]
                                    returns = ((exit_price - entry_price) / entry_price) * 100
                                    trade_result[f'Return_{period}d'] = returns
                                    trade_result[f'Stop_Hit_{period}d'] = False

                                max_dd = ((period_data['Close'].min() - entry_price) / entry_price) * 100
                                trade_result[f'Max_DD_{period}d'] = max_dd
                            else:
                                trade_result[f'Return_{period}d'] = None
                                trade_result[f'Max_DD_{period}d'] = None
                                trade_result[f'Stop_Hit_{period}d'] = None

                        self.trades.append(trade_result)
            else:
                print("\n❌ No signals found")

        print(f"\n{'=' * 100}")
        print(f"✅ COMPLETE - Total: {total_signals} | Breakout Ready: {breakout_ready_count}")
        print(f"{'=' * 100}\n")

        return self.analyze_results(holding_periods)

    def fetch_future_data(self, symbol, scan_date, days_ahead):
        try:
            start = scan_date + timedelta(days=1)
            end = scan_date + timedelta(days=days_ahead + 30)
            stock = yf.Ticker(symbol)
            df = stock.history(start=start, end=end)
            return df if len(df) > 0 else None
        except:
            return None

    def analyze_results(self, holding_periods):
        if not self.trades:
            print("⚠️  No trades to analyze")
            return None

        trades_df = pd.DataFrame(self.trades)

        print("\n" + "=" * 100)
        print("📊 ENHANCED BACKTEST RESULTS")
        print("=" * 100 + "\n")

        # Compare stages
        breakout_trades = trades_df[trades_df['Stage'] == 'BREAKOUT_READY']
        accum_trades = trades_df[trades_df['Stage'] == 'ACCUMULATION']

        print("🔥 BREAKOUT READY vs 📊 ACCUMULATION COMPARISON (30-day returns):\n")

        comparison_data = []

        for stage, stage_df in [('BREAKOUT_READY', breakout_trades), ('ACCUMULATION', accum_trades)]:
            valid = stage_df[stage_df['Return_30d'].notna()]
            if len(valid) > 0:
                stop_hit_rate = valid['Stop_Hit_30d'].mean() * 100 if 'Stop_Hit_30d' in valid else 0
                comparison_data.append({
                    'Stage': stage,
                    'Signals': len(valid),
                    'Avg_Return': f"{valid['Return_30d'].mean():.2f}%",
                    'Median_Return': f"{valid['Return_30d'].median():.2f}%",
                    'Win_Rate': f"{(valid['Return_30d'] > 0).sum() / len(valid) * 100:.2f}%",
                    'Big_Winners_10%+': f"{(valid['Return_30d'] > 10).sum()} ({(valid['Return_30d'] > 10).sum() / len(valid) * 100:.1f}%)",
                    'Avg_Max_DD': f"{valid['Max_DD_30d'].mean():.2f}%",
                    'Stop_Hit_Rate': f"{stop_hit_rate:.1f}%",
                    'Avg_R:R': f"{valid['Risk_Reward'].mean():.2f}:1"
                })

        if comparison_data:
            print(pd.DataFrame(comparison_data).to_string(index=False))

        # Overall performance by holding period
        print("\n" + "=" * 100)
        print("📈 PERFORMANCE BY HOLDING PERIOD")
        print("=" * 100 + "\n")

        results_summary = []
        for period in holding_periods:
            return_col = f'Return_{period}d'
            valid = trades_df[trades_df[return_col].notna()]

            if len(valid) > 0:
                stop_col = f'Stop_Hit_{period}d'
                stop_hit_rate = valid[stop_col].mean() * 100 if stop_col in valid else 0

                results_summary.append({
                    'Period': f'{period}d',
                    'Trades': len(valid),
                    'Avg_Return': f'{valid[return_col].mean():.2f}%',
                    'Median_Return': f'{valid[return_col].median():.2f}%',
                    'Win_Rate': f'{(valid[return_col] > 0).sum() / len(valid) * 100:.2f}%',
                    'Best': f'{valid[return_col].max():.2f}%',
                    'Worst': f'{valid[return_col].min():.2f}%',
                    'Stop_Hit': f'{stop_hit_rate:.1f}%'
                })

        print(pd.DataFrame(results_summary).to_string(index=False))

        # Performance by Base Quality
        print("\n" + "=" * 100)
        print("🏆 PERFORMANCE BY BASE QUALITY (30-day returns)")
        print("=" * 100 + "\n")

        quality_analysis = []
        for threshold in [4, 5, 6, 7]:
            filtered = trades_df[trades_df['Base_Quality'] >= threshold]
            if len(filtered) > 0:
                valid = filtered[filtered['Return_30d'].notna()]
                if len(valid) > 0:
                    quality_analysis.append({
                        'Base_Quality': f'>= {threshold}',
                        'Signals': len(valid),
                        'Avg_Return': f"{valid['Return_30d'].mean():.2f}%",
                        'Median_Return': f"{valid['Return_30d'].median():.2f}%",
                        'Win_Rate': f"{(valid['Return_30d'] > 0).sum() / len(valid) * 100:.2f}%",
                        'Big_Winners': f"{(valid['Return_30d'] > 10).sum()} ({(valid['Return_30d'] > 10).sum() / len(valid) * 100:.1f}%)"
                    })

        if quality_analysis:
            print(pd.DataFrame(quality_analysis).to_string(index=False))

        # Performance by RS Score
        print("\n" + "=" * 100)
        print("💪 PERFORMANCE BY RELATIVE STRENGTH (30-day returns)")
        print("=" * 100 + "\n")

        rs_analysis = []
        for threshold in [60, 70, 80, 90]:
            filtered = trades_df[trades_df['RS_Score'] >= threshold]
            if len(filtered) > 0:
                valid = filtered[filtered['Return_30d'].notna()]
                if len(valid) > 0:
                    rs_analysis.append({
                        'RS_Score': f'>= {threshold}',
                        'Signals': len(valid),
                        'Avg_Return': f"{valid['Return_30d'].mean():.2f}%",
                        'Win_Rate': f"{(valid['Return_30d'] > 0).sum() / len(valid) * 100:.2f}%",
                        'Big_Winners': f"{(valid['Return_30d'] > 10).sum()}"
                    })

        if rs_analysis:
            print(pd.DataFrame(rs_analysis).to_string(index=False))

        # Top performers
        # In the analyze_results method, around line 863
        # Top performers
        print("\n" + "=" * 100)
        print("🏅 TOP 10 BEST TRADES (30-day)")
        print("=" * 100 + "\n")

        # Convert 'Return_30d' to numeric, coercing errors to NaN
        trades_df['Return_30d'] = pd.to_numeric(trades_df['Return_30d'], errors='coerce')

        # Filter out rows where Return_30d is NaN to avoid issues with nlargest
        valid_trades_df = trades_df[trades_df['Return_30d'].notna()]

        # Select top 10 trades
        top = valid_trades_df.nlargest(10, 'Return_30d')[['Signal_Date', 'Symbol', 'Stage',
                                                          'Total_Score', 'Base_Quality', 'RS_Score',
                                                          'Entry_Price', 'Return_30d', 'Risk_Reward']]
        top['Signal_Date'] = top['Signal_Date'].dt.strftime('%Y-%m-%d')
        top['Return_30d'] = top['Return_30d'].apply(lambda x: f'{x:.2f}%')
        top['Risk_Reward'] = top['Risk_Reward'].apply(lambda x: f'{x:.2f}:1')
        print(top.to_string(index=False))

        # Worst trades
        print("\n" + "=" * 100)
        print("⚠️  TOP 10 WORST TRADES (30-day)")
        print("=" * 100 + "\n")

        worst = trades_df.nsmallest(10, 'Return_30d')[['Signal_Date', 'Symbol', 'Stage',
                                                       'Total_Score', 'Base_Quality', 'RS_Score',
                                                       'Entry_Price', 'Return_30d', 'Stop_Hit_30d']]
        worst['Signal_Date'] = worst['Signal_Date'].dt.strftime('%Y-%m-%d')
        worst['Return_30d'] = worst['Return_30d'].apply(lambda x: f'{x:.2f}%')
        print(worst.to_string(index=False))

        # Performance by score threshold
        print("\n" + "=" * 100)
        print("📈 PERFORMANCE BY TOTAL SCORE (30-day returns)")
        print("=" * 100 + "\n")

        score_analysis = []
        for threshold in [18, 20, 22, 25]:
            filtered = trades_df[trades_df['Total_Score'] >= threshold]
            if len(filtered) > 0:
                valid = filtered[filtered['Return_30d'].notna()]
                if len(valid) > 0:
                    score_analysis.append({
                        'Total_Score': f'>= {threshold}',
                        'Signals': len(valid),
                        'Avg_Return': f"{valid['Return_30d'].mean():.2f}%",
                        'Win_Rate': f"{(valid['Return_30d'] > 0).sum() / len(valid) * 100:.2f}%",
                        'Big_Winners': f"{(valid['Return_30d'] > 10).sum()} ({(valid['Return_30d'] > 10).sum() / len(valid) * 100:.1f}%)"
                    })

        if score_analysis:
            print(pd.DataFrame(score_analysis).to_string(index=False))

        # Stock-wise performance
        print("\n" + "=" * 100)
        print("📊 TOP PERFORMING STOCKS (by average 30-day return)")
        print("=" * 100 + "\n")

        stock_performance = trades_df.groupby('Symbol').agg({
            'Return_30d': ['count', 'mean', 'median'],
            'Total_Score': 'mean',
            'Base_Quality': 'mean',
            'RS_Score': 'mean'
        }).round(2)
        stock_performance.columns = ['Signals', 'Avg_Return', 'Median_Return', 'Avg_Score', 'Avg_Base_Quality',
                                     'Avg_RS']
        stock_performance = stock_performance[stock_performance['Signals'] >= 2]
        stock_performance = stock_performance.sort_values('Avg_Return', ascending=False).head(10)
        print(stock_performance.to_string())

        # Risk-Reward Analysis
        print("\n" + "=" * 100)
        print("⚖️  RISK-REWARD ANALYSIS (30-day returns)")
        print("=" * 100 + "\n")

        rr_analysis = []
        for threshold in [2.5, 3.0, 3.5, 4.0]:
            filtered = trades_df[trades_df['Risk_Reward'] >= threshold]
            if len(filtered) > 0:
                valid = filtered[filtered['Return_30d'].notna()]
                if len(valid) > 0:
                    rr_analysis.append({
                        'R:R_Ratio': f'>= {threshold}:1',
                        'Signals': len(valid),
                        'Avg_Return': f"{valid['Return_30d'].mean():.2f}%",
                        'Win_Rate': f"{(valid['Return_30d'] > 0).sum() / len(valid) * 100:.2f}%"
                    })

        if rr_analysis:
            print(pd.DataFrame(rr_analysis).to_string(index=False))

        # Key Insights
        print("\n" + "=" * 100)
        print("💡 KEY INSIGHTS")
        print("=" * 100 + "\n")

        valid_30d = trades_df[trades_df['Return_30d'].notna()]
        if len(valid_30d) > 0:
            # Best combination
            best_combo = valid_30d[
                (valid_30d['Base_Quality'] >= 5) &
                (valid_30d['RS_Score'] >= 70) &
                (valid_30d['Stage'] == 'BREAKOUT_READY')
                ]

            if len(best_combo) > 0:
                print(f"🎯 HIGH QUALITY SETUPS (Base≥5, RS≥70, Breakout Ready):")
                print(f"   Signals: {len(best_combo)}")
                print(f"   Avg Return: {best_combo['Return_30d'].mean():.2f}%")
                print(f"   Win Rate: {(best_combo['Return_30d'] > 0).sum() / len(best_combo) * 100:.2f}%")
                print(
                    f"   Big Winners (>10%): {(best_combo['Return_30d'] > 10).sum()} ({(best_combo['Return_30d'] > 10).sum() / len(best_combo) * 100:.1f}%)")

            print(f"\n📊 OVERALL STATISTICS:")
            print(f"   Total Signals: {len(valid_30d)}")
            print(f"   Average Return: {valid_30d['Return_30d'].mean():.2f}%")
            print(f"   Median Return: {valid_30d['Return_30d'].median():.2f}%")
            print(f"   Win Rate: {(valid_30d['Return_30d'] > 0).sum() / len(valid_30d) * 100:.2f}%")
            print(f"   Stop Hit Rate: {valid_30d['Stop_Hit_30d'].mean() * 100:.1f}%")
            print(f"   Avg Risk:Reward: {valid_30d['Risk_Reward'].mean():.2f}:1")

        # Save results
        trades_df['Signal_Date'] = trades_df['Signal_Date'].dt.strftime('%Y-%m-%d')
        trades_df.to_csv(f'enhanced_momentum_V1_results_{end_date.strftime("%Y%m%d")}.csv', index=False)

        print("\n💾 Results saved to: enhanced_backtest_results_final.csv\n")

        return trades_df


# Expanded stock list
NIFTY_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS',
    'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS', 'ADANIPORTS.NS', 'TATASTEEL.NS',
    'JSWSTEEL.NS', 'INDUSINDBK.NS', 'BAJAJFINSV.NS', 'M&M.NS', 'HINDALCO.NS',
    'POWERGRID.NS', 'NTPC.NS', 'COALINDIA.NS', 'ONGC.NS', 'GRASIM.NS',
    'DIVISLAB.NS', 'DRREDDY.NS', 'CIPLA.NS', 'TATACONSUM.NS', 'EICHERMOT.NS',
    'TATAMOTORS.NS', 'APOLLOHOSP.NS', 'BPCL.NS', 'BRITANNIA.NS',
    'HEROMOTOCO.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'UPL.NS', 'BAJAJ-AUTO.NS'
]

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("🚀 ENHANCED SMART MONEY SCANNER - FINAL VERSION")
    print("=" * 100)
    print("\n✨ NEW FEATURES:")
    print("   • Base Quality Analysis (tight consolidation, volume coiling)")
    print("   • Institutional Block Detection")
    print("   • False Breakout Memory (avoids burned zones)")
    print("   • Risk-Reward Filtering (minimum 2.5:1)")
    print("   • Volume Climax Detection (spring patterns)")
    print("   • Enhanced Multi-Timeframe Relative Strength")
    print("   • Stop-Loss Integration (2x ATR)")
    print("\n")

    start_date = datetime(2025, 9, 1)
    end_date = datetime(2025, 10, 10)

    days_available = (end_date - start_date).days
    print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Prompt for CSV file selection
    resources_path = 'resources'
    if not os.path.exists(resources_path):
        print(f"Error: {resources_path} directory not found.")
        exit(1)

    csv_files = [f for f in os.listdir(resources_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {resources_path}.")
        exit(1)

    print("\nAvailable CSV files in resources:")
    for i, f in enumerate(csv_files, 1):
        print(f"{i}. {f}")

    try:
        choice = int(input("\nChoose file number to scan: ")) - 1
        if choice < 0 or choice >= len(csv_files):
            raise ValueError("Invalid choice.")
    except (ValueError, IndexError):
        print("Invalid choice. Exiting.")
        exit(1)

    selected_file = csv_files[choice]
    file_path = os.path.join(resources_path, selected_file)
    df_symbols = pd.read_csv(file_path)
    symbols_list = df_symbols['Symbol'].tolist()

    print(f"\n📊 Loaded {len(symbols_list)} symbols from {selected_file}")
    print(f"⏱️  Duration: {days_available} days ({days_available // 7} weekly scans)")

    backtester = Backtester(
        symbols_list=symbols_list,
        start_date=start_date,
        end_date=end_date
    )

    results = backtester.run_backtest(holding_periods=[10, 20, 30, 60])

    # Dynamic file naming for output
    output_filename = f'enhanced_momentum_V2_{os.path.splitext(selected_file)[0]}_{end_date.strftime("%Y%m%d")}.csv'
    if hasattr(results, 'to_csv'):
        results.to_csv(output_filename, index=False)
        print(f"\n💾 Results saved to: {output_filename}")

    print("\n✅ ENHANCED BACKTEST COMPLETE!")
    print("\n" + "=" * 100)