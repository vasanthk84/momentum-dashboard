import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import glob
warnings.filterwarnings('ignore')

class StockScanner:
    def __init__(self, resources_folder='resources'):
        """Initialize the stock scanner with stocks from CSV files."""
        self.resources_folder = resources_folder
        self.stocks = []
        self.lookback_period = 200  # Days to fetch (12+ months)
        self.results = []
        
        # Load stocks from CSV files
        self.load_stocks_from_csv()
        
    def load_stocks_from_csv(self):
        """Load stock symbols from all CSV files in the resources folder."""
        print(f"Loading stocks from '{self.resources_folder}' folder...")
        
        if not os.path.exists(self.resources_folder):
            print(f"ERROR: '{self.resources_folder}' folder not found!")
            print("Please create a 'resources' folder and add your CSV files.")
            return
        
        # Find all CSV files in the resources folder
        csv_files = glob.glob(os.path.join(self.resources_folder, '*.csv'))
        
        if not csv_files:
            print(f"ERROR: No CSV files found in '{self.resources_folder}' folder!")
            return
        
        print(f"Found {len(csv_files)} CSV file(s):\n")
        
        all_symbols = set()  # Use set to avoid duplicates
        
        for csv_file in csv_files:
            try:
                filename = os.path.basename(csv_file)
                print(f"  Reading: {filename}")
                
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Try to find the symbol column (common names)
                symbol_column = None
                for col in df.columns:
                    col_lower = col.lower().strip()
                    if col_lower in ['symbol', 'ticker', 'stock', 'code', 'stock symbol', 'stock_symbol']:
                        symbol_column = col
                        break
                
                if symbol_column is None:
                    # If no symbol column found, try the first column
                    print(f"    Warning: No 'Symbol' column found. Using first column: '{df.columns[0]}'")
                    symbol_column = df.columns[0]
                
                # Extract symbols
                symbols = df[symbol_column].dropna().astype(str).str.strip()
                
                # Add .NS suffix if not present (for NSE stocks)
                for symbol in symbols:
                    symbol = symbol.upper()
                    # Remove any existing suffixes
                    symbol = symbol.replace('.NS', '').replace('.BO', '').replace('.BSE', '')
                    # Add .NS for NSE
                    all_symbols.add(f"{symbol}.NS")
                
                print(f"    Loaded {len(symbols)} symbols")
                
            except Exception as e:
                print(f"    ERROR reading {filename}: {str(e)}")
                continue
        
        self.stocks = sorted(list(all_symbols))
        print(f"\nTotal unique stocks loaded: {len(self.stocks)}\n")
        
        if len(self.stocks) == 0:
            print("ERROR: No stocks loaded. Please check your CSV files.")
        
    def fetch_data(self, symbol):
        """Fetch historical data for a stock."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_period)
            
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if len(df) < 250:  # Need at least 250 days for proper analysis
                return None
            
            return df
        except Exception as e:
            # Only print errors for non-connection issues
            if "No data found" not in str(e):
                print(f"  Error fetching {symbol}: {str(e)}")
            return None
    
    def calculate_ma(self, df, period=200):
        """Calculate moving average."""
        return df['Close'].rolling(window=period).mean()
    
    def check_ma_slope_positive(self, df, ma_column):
        """Check if 200 DMA has positive slope (upward trending)."""
        # Check slope over last 60 days
        if len(df) < 60:
            return False, 0
        
        ma_60_days_ago = df[ma_column].iloc[-60]
        ma_current = df[ma_column].iloc[-1]
        
        # Calculate percentage change
        slope_pct = ((ma_current - ma_60_days_ago) / ma_60_days_ago) * 100
        
        # MA should be rising (positive slope)
        return slope_pct > 0, slope_pct
    
    def check_was_in_uptrend(self, df, ma_column, lookback_start=300, lookback_end=100):
        """Check if stock was in uptrend earlier (6-12 months ago)."""
        if len(df) < lookback_start:
            return False, 0
        
        # Check period from 10-6 months ago
        historical_period = df.iloc[-lookback_start:-lookback_end]
        
        if len(historical_period) < 30:
            return False, 0
        
        # Calculate percentage of days above MA during that period
        above_ma = (historical_period['Close'] > historical_period[ma_column]).sum()
        total_days = len(historical_period)
        percentage_above = (above_ma / total_days) * 100
        
        # Stock should have been above MA at least 70% of the time during uptrend
        return percentage_above >= 70, percentage_above
    
    def check_correction_phase(self, df, ma_column, lookback_days=100):
        """Check if stock went through correction phase (was below MA recently)."""
        if len(df) < lookback_days:
            return False, {}
        
        # Check last 100 days (excluding last 5 days for current touch)
        correction_period = df.iloc[-lookback_days:-5]
        
        if len(correction_period) < 20:
            return False, {}
        
        # Stock should have been below MA for significant time during correction
        below_ma = (correction_period['Close'] < correction_period[ma_column]).sum()
        total_days = len(correction_period)
        percentage_below = (below_ma / total_days) * 100
        
        # Should have been below MA at least 50% of correction period
        if percentage_below < 50:
            return False, {'pct_below': percentage_below}
        
        # Calculate max drawdown during correction
        max_price_in_period = correction_period['Close'].max()
        min_price_in_period = correction_period['Close'].min()
        max_drawdown = ((min_price_in_period - max_price_in_period) / max_price_in_period) * 100
        
        return True, {
            'pct_below_ma': percentage_below,
            'max_drawdown': max_drawdown
        }
    
    def check_approaching_ma_from_below(self, df, ma_column, max_distance=0.20):
        """Check if price is approaching/touching 200 DMA from BELOW."""
        current_price = df['Close'].iloc[-1]
        current_ma = df[ma_column].iloc[-1]
        
        # Calculate distance from MA (negative if below, positive if above)
        distance_from_ma = ((current_price - current_ma) / current_ma)
        diff_percentage = abs(distance_from_ma) * 100
        
        # Price should be below MA or slightly above (within 20% range)
        if distance_from_ma > 0.10:  # More than 10% above MA - already broke out
            return False, {}
        
        # Should be within 20% of MA (can be below)
        if diff_percentage > 20:
            return False, {}
        
        # Check if price is rising towards MA (positive momentum in recent days)
        # Compare last 10 days vs previous 10 days
        last_10 = df['Close'].tail(10).mean()
        prev_10 = df['Close'].iloc[-20:-10].mean()
        
        recent_momentum = ((last_10 - prev_10) / prev_10) * 100
        is_rising = recent_momentum > 0
        
        # Also check if price has been rising in last 20 days
        price_20_days_ago = df['Close'].iloc[-20]
        price_change_20d = ((current_price - price_20_days_ago) / price_20_days_ago) * 100
        
        return True, {
            'current_price': current_price,
            'ma_value': current_ma,
            'distance_from_ma_pct': diff_percentage,
            'position': 'below' if current_price < current_ma else 'above',
            'rising_into_ma': is_rising,
            'recent_momentum_pct': recent_momentum,
            'price_change_20d': price_change_20d
        }
    
    def check_no_sharp_crash(self, df, window=60):
        """Check that correction was not a sharp crash."""
        recent = df.tail(window)
        
        if len(recent) < 20:
            return False, {}
        
        # Calculate daily returns
        returns = recent['Close'].pct_change()
        
        # Check for sharp single-day drops (> 8% in one day is crash)
        max_single_drop = returns.min() * 100
        if max_single_drop < -8:
            return False, {'max_drop': max_single_drop, 'reason': 'Sharp crash'}
        
        # Count number of days with > 5% drop
        big_drop_days = (returns < -0.05).sum()
        if big_drop_days > 3:  # More than 3 days with 5%+ drops
            return False, {'big_drop_days': big_drop_days, 'reason': 'Multiple large drops'}
        
        return True, {
            'max_single_drop': max_single_drop,
            'big_drop_days': big_drop_days
        }
    
    def check_volume_behavior(self, df):
        """Check for reasonable volume behavior."""
        recent = df.tail(30)
        
        # Calculate average volume
        avg_volume_recent = recent['Volume'].mean()
        avg_volume_overall = df['Volume'].mean()
        
        # Volume should be reasonable (not extreme panic)
        volume_ratio = avg_volume_recent / avg_volume_overall
        
        return True, {'volume_ratio': volume_ratio}
    
    def analyze_stock(self, symbol):
        """Perform complete analysis on a stock."""
        df = self.fetch_data(symbol)
        if df is None or len(df) < 250:
            return None
        
        # Calculate 200 DMA
        df['MA200'] = self.calculate_ma(df, 200)
        
        # Remove NaN values
        df = df.dropna()
        
        if len(df) < 250:
            return None
        
        # Step 1: Check if 200 DMA has positive slope (still in long-term uptrend)
        is_ma_positive, ma_slope = self.check_ma_slope_positive(df, 'MA200')
        if not is_ma_positive:
            return None
        
        # Step 2: Check if stock was in uptrend earlier (before correction)
        was_uptrend, pct_above_historical = self.check_was_in_uptrend(df, 'MA200')
        if not was_uptrend:
            return None
        
        # Step 3: Check if stock went through correction phase
        had_correction, correction_info = self.check_correction_phase(df, 'MA200')
        if not had_correction:
            return None
        
        # Step 4: Check if currently approaching/touching MA from below
        is_approaching, approach_info = self.check_approaching_ma_from_below(df, 'MA200')
        if not is_approaching:
            return None
        
        # Step 5: Check for no sharp crash
        is_smooth, crash_info = self.check_no_sharp_crash(df)
        if not is_smooth:
            return None
        
        # Step 6: Check volume
        is_healthy_volume, volume_info = self.check_volume_behavior(df)
        
        # Calculate additional metrics
        current_price = df['Close'].iloc[-1]
        ma_200 = df['MA200'].iloc[-1]
        
        # Calculate price change over correction period
        price_60_days_ago = df['Close'].iloc[-60] if len(df) > 60 else current_price
        price_change_60d = ((current_price - price_60_days_ago) / price_60_days_ago) * 100
        
        return {
            'symbol': symbol.replace('.NS', ''),
            'current_price': round(current_price, 2),
            'ma_200': round(ma_200, 2),
            'distance_from_ma_pct': round(approach_info['distance_from_ma_pct'], 2),
            'position': approach_info['position'],
            'rising_into_ma': approach_info['rising_into_ma'],
            'recent_momentum_pct': round(approach_info['recent_momentum_pct'], 2),
            'price_change_20d_pct': round(approach_info['price_change_20d'], 2),
            'ma_slope_60d': round(ma_slope, 2),
            'was_above_pct_historical': round(pct_above_historical, 1),
            'was_below_pct_correction': round(correction_info['pct_below_ma'], 1),
            'max_drawdown_pct': round(correction_info['max_drawdown'], 2),
            'max_single_drop_pct': round(crash_info['max_single_drop'], 2),
            'volume_ratio': round(volume_info['volume_ratio'], 2),
            'price_change_60d_pct': round(price_change_60d, 2)
        }
    
    def scan_all_stocks(self):
        """Scan all stocks in the list."""
        if len(self.stocks) == 0:
            print("No stocks to scan. Exiting.")
            return []
        
        print("Starting stock scan...")
        print(f"Scanning {len(self.stocks)} stocks...\n")
        print("This may take several minutes depending on the number of stocks.\n")
        
        processed = 0
        for symbol in self.stocks:
            processed += 1
            if processed % 10 == 0:
                print(f"Progress: {processed}/{len(self.stocks)} stocks processed...")
            
            result = self.analyze_stock(symbol)
            if result:
                self.results.append(result)
                print(f"  ✓ MATCH FOUND: {symbol.replace('.NS', '')}")
        
        print(f"\nCompleted: {processed}/{len(self.stocks)} stocks processed.")
        return self.results
    
    def display_results(self):
        """Display the scan results."""
        if not self.results:
            print("\n" + "="*100)
            print("No stocks found matching the criteria.")
            print("="*100)
            print("\nThis could mean:")
            print("1. No stocks are currently in the retest pattern")
            print("2. Market conditions don't show this specific setup")
            print("3. Try adjusting the criteria (tolerance, lookback periods, etc.)")
            return
        
        print("\n" + "="*100)
        print("STOCKS APPROACHING/RETESTING 200 DMA FROM BELOW - COLGATE PATTERN")
        print("="*100)
        
        df_results = pd.DataFrame(self.results)
        
        # Sort by distance from MA (closest first)
        df_results = df_results.sort_values('distance_from_ma_pct')
        
        print(f"\nFound {len(df_results)} stocks:\n")
        print(df_results.to_string(index=False))
        
        print("\n" + "="*100)
        print("COLUMN DEFINITIONS:")
        print("="*100)
        print("distance_from_ma_pct: How close to 200 DMA (lower = closer, can be up to 20%)")
        print("position: Whether currently above or below MA")
        print("rising_into_ma: Whether price momentum is positive (rising)")
        print("recent_momentum_pct: 10-day momentum (positive = rising)")
        print("price_change_20d_pct: Price change over last 20 days")
        print("ma_slope_60d: 60-day MA slope (positive = uptrend intact)")
        print("was_above_pct_historical: % days above MA in earlier uptrend phase")
        print("was_below_pct_correction: % days below MA during correction")
        print("max_drawdown_pct: Maximum drop during correction")
        print("max_single_drop_pct: Largest single-day drop")
        print("volume_ratio: Recent volume vs average")
        print("price_change_60d_pct: Price change over last 60 days")
        print("="*100)
        
        # Save to CSV
        output_file = 'stock_scan_results.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to '{output_file}'")


def main():
    """Main execution function."""
    print("="*100)
    print("INDIAN STOCK SCANNER - 200 DMA RETEST PATTERN (LIKE COLGATE)")
    print("="*100)
    print("\nPattern Criteria:")
    print("1. Stock was in uptrend earlier (above 200 DMA 6-12 months ago)")
    print("2. Went through correction phase (traded below 200 DMA)")
    print("3. No sharp crash or panic selling during correction")
    print("4. 200 DMA is still rising (long-term uptrend intact)")
    print("5. Currently within 20% of 200 DMA (approaching from below)")
    print("6. Price showing positive momentum (rising in recent days)")
    print("\nThis pattern suggests potential support and reversal near 200 DMA\n")
    print("="*100)
    
    # Initialize scanner (will read from 'resources' folder by default)
    scanner = StockScanner(resources_folder='resources')
    
    if len(scanner.stocks) > 0:
        scanner.scan_all_stocks()
        scanner.display_results()
    else:
        print("\nPlease ensure:")
        print("1. A 'resources' folder exists in the same directory as this script")
        print("2. CSV files are present in the 'resources' folder")
        print("3. CSV files contain a column named 'Symbol', 'Ticker', or similar")
        print("\nExample CSV format:")
        print("Symbol")
        print("RELIANCE")
        print("TCS")
        print("INFY")


if __name__ == "__main__":
    main()