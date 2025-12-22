"""
DATABASE PERSISTENCE LAYER FOR STOCK SCANNER
============================================
Drop-in replacement for CSV file operations
NO changes to scanner logic - just intercepts save/load operations

Database: SQLite (no external dependencies)
Features:
  ✅ Complete audit trail - nothing deleted
  ✅ Fast queries - indexed by date, symbol, scan type
  ✅ Backward compatible - still generates CSV for immediate use
  ✅ Export historical data anytime
  ✅ Track performance over time

Author: Scanner Persistence Module
Version: 1.0
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
import os
from pathlib import Path

# ============================================================================
# DATABASE SCHEMA
# ============================================================================

class ScanDatabase:
    """SQLite database for persistent scan storage"""
    
    def __init__(self, db_path='scan_history.db'):
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Table 1: Weekly Watchlist Scans (TIER 1)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                stage TEXT NOT NULL,
                total_score REAL,
                accumulation_score REAL,
                breakout_score REAL,
                current_price REAL,
                signal_price REAL,
                stop_price REAL,
                risk_reward REAL,
                rs_score REAL,
                base_quality INTEGER,
                distance_to_resistance REAL,
                poc_price REAL,
                value_area_low REAL,
                value_area_high REAL,
                signals TEXT,
                universe TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(scan_date, symbol, universe)
            )
        ''')
        
        # Table 2: Daily Entry Signals (TIER 2)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                decision TEXT NOT NULL,
                grade TEXT,
                score INTEGER,
                wyckoff_phase TEXT,
                signal_price REAL,
                current_price REAL,
                price_date DATE,
                price_change_pct REAL,
                vwap REAL,
                vwap_deviation_pct REAL,
                poc REAL,
                value_area_low REAL,
                value_area_high REAL,
                in_value_area BOOLEAN,
                volume_ratio REAL,
                above_ma20 BOOLEAN,
                above_ma50 BOOLEAN,
                ma_aligned BOOLEAN,
                entry_low REAL,
                entry_high REAL,
                stop_loss REAL,
                risk_pct REAL,
                target_1 REAL,
                target_2 REAL,
                reasons TEXT,
                wait_factors TEXT,
                skip_factors TEXT,
                market_trend TEXT,
                market_strength INTEGER,
                universe TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(scan_date, symbol, decision, universe)
            )
        ''')
        
        # Table 3: Performance Tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                symbol TEXT NOT NULL,
                entry_date DATE NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                return_pct REAL,
                status TEXT,
                stop_hit BOOLEAN DEFAULT 0,
                target_hit BOOLEAN DEFAULT 0,
                days_held INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES daily_signals(id)
            )
        ''')
        
        # Table 4: Scan Metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TIMESTAMP NOT NULL,
                scan_type TEXT NOT NULL,
                universe TEXT NOT NULL,
                total_signals INTEGER,
                buy_signals INTEGER,
                wait_signals INTEGER,
                skip_signals INTEGER,
                avg_score REAL,
                avg_rs_score REAL,
                scan_duration_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_weekly_date ON weekly_watchlist(scan_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_weekly_symbol ON weekly_watchlist(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_signals(scan_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_symbol ON daily_signals(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_decision ON daily_signals(decision)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_symbol ON performance_tracking(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_date ON performance_tracking(entry_date)')
        
        self.conn.commit()
        print(f"✅ Database initialized: {self.db_path}")
    
    def save_weekly_watchlist(self, df, universe='Unknown'):
        """Save weekly watchlist to database"""
        if df is None or df.empty:
            return
        
        scan_date = datetime.now()
        cursor = self.conn.cursor()
        
        saved_count = 0
        for _, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO weekly_watchlist (
                        scan_date, symbol, stage, total_score, accumulation_score,
                        breakout_score, current_price, signal_price, stop_price,
                        risk_reward, rs_score, base_quality, distance_to_resistance,
                        poc_price, value_area_low, value_area_high, signals, universe
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    scan_date,
                    row.get('Symbol', ''),
                    row.get('Stage', ''),
                    row.get('Total_Score', 0),
                    row.get('Accumulation_Score', 0),
                    row.get('Breakout_Score', 0),
                    row.get('Current_Price', 0),
                    row.get('Signal_Price', 0) if pd.notna(row.get('Signal_Price')) else row.get('Current_Price', 0),
                    row.get('Stop_Price', 0),
                    row.get('Risk_Reward', 0),
                    row.get('RS_Score', 0),
                    row.get('Base_Quality', 0),
                    row.get('Distance_to_Resistance', 0),
                    row.get('POC_Price', 0) if pd.notna(row.get('POC_Price')) else 0,
                    row.get('Value_Area_Low', 0) if pd.notna(row.get('Value_Area_Low')) else 0,
                    row.get('Value_Area_High', 0) if pd.notna(row.get('Value_Area_High')) else 0,
                    json.dumps(row.get('Signals', [])) if isinstance(row.get('Signals'), list) else '[]',
                    universe
                ))
                saved_count += 1
            except Exception as e:
                print(f"⚠️  Failed to save {row.get('Symbol', 'Unknown')}: {e}")
        
        self.conn.commit()
        print(f"💾 Saved {saved_count} weekly watchlist stocks to database")
    
    def save_daily_signals(self, results_dict, universe='Unknown'):
        """Save daily signals to database"""
        if not results_dict:
            return
        
        scan_date = datetime.now()
        cursor = self.conn.cursor()
        
        total_saved = 0
        
        for decision_type in ['buy', 'wait', 'skip']:
            stocks = results_dict.get(decision_type, [])
            
            for stock in stocks:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_signals (
                            scan_date, symbol, decision, grade, score, wyckoff_phase,
                            signal_price, current_price, price_date, price_change_pct,
                            vwap, vwap_deviation_pct, poc, value_area_low, value_area_high,
                            in_value_area, volume_ratio, above_ma20, above_ma50, ma_aligned,
                            entry_low, entry_high, stop_loss, risk_pct, target_1, target_2,
                            reasons, wait_factors, skip_factors, market_trend, market_strength,
                            universe
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        scan_date,
                        stock.get('Symbol', ''),
                        decision_type.upper(),
                        stock.get('Grade', ''),
                        stock.get('Score', 0),
                        stock.get('Wyckoff_Phase', ''),
                        stock.get('Signal_Price', 0),
                        stock.get('Current_Price', 0),
                        stock.get('Price_Date', ''),
                        stock.get('Price_Change_%', 0),
                        stock.get('VWAP', 0),
                        stock.get('VWAP_Deviation_%', 0),
                        stock.get('POC', 0),
                        stock.get('Value_Area_Low', 0),
                        stock.get('Value_Area_High', 0),
                        stock.get('In_Value_Area', False),
                        stock.get('Volume_Ratio', 0),
                        stock.get('Above_MA20', False),
                        stock.get('Above_MA50', False),
                        stock.get('MA_Aligned', False),
                        stock.get('Entry_Low', 0),
                        stock.get('Entry_High', 0),
                        stock.get('Stop_Loss', 0),
                        stock.get('Risk_%', 0),
                        stock.get('Target_1', 0),
                        stock.get('Target_2', 0),
                        json.dumps(stock.get('Reasons', [])),
                        json.dumps(stock.get('Wait_Factors', [])),
                        json.dumps(stock.get('Skip_Factors', [])),
                        stock.get('Market_Trend', ''),
                        stock.get('Market_Strength', 0),
                        universe
                    ))
                    total_saved += 1
                except Exception as e:
                    print(f"⚠️  Failed to save {stock.get('Symbol', 'Unknown')}: {e}")
        
        self.conn.commit()
        print(f"💾 Saved {total_saved} daily signals to database")
        
        # Save metadata
        self._save_scan_metadata(scan_date, 'DAILY', universe, results_dict)
    
    def _save_scan_metadata(self, scan_date, scan_type, universe, results_dict):
        """Save scan metadata for reporting"""
        cursor = self.conn.cursor()
        
        buy_count = len(results_dict.get('buy', []))
        wait_count = len(results_dict.get('wait', []))
        skip_count = len(results_dict.get('skip', []))
        
        cursor.execute('''
            INSERT INTO scan_metadata (
                scan_date, scan_type, universe, total_signals,
                buy_signals, wait_signals, skip_signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            scan_date, scan_type, universe,
            buy_count + wait_count + skip_count,
            buy_count, wait_count, skip_count
        ))
        
        self.conn.commit()
    
    def get_scan_history(self, days=30, scan_type='DAILY', universe=None):
        """Retrieve scan history"""
        query = '''
            SELECT * FROM scan_metadata 
            WHERE scan_type = ? 
            AND scan_date >= datetime('now', '-' || ? || ' days')
        '''
        params = [scan_type, days]
        
        if universe:
            query += ' AND universe = ?'
            params.append(universe)
        
        query += ' ORDER BY scan_date DESC'
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def get_buy_signals_by_date(self, date_str, universe=None):
        """Get all BUY signals from a specific date"""
        query = '''
            SELECT * FROM daily_signals 
            WHERE decision = 'BUY'
            AND DATE(scan_date) = DATE(?)
        '''
        params = [date_str]
        
        if universe:
            query += ' AND universe = ?'
            params.append(universe)
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def get_performance_data(self, days=30):
        """Get performance tracking data"""
        query = '''
            SELECT 
                p.symbol,
                p.entry_date,
                p.entry_price,
                p.current_price,
                p.return_pct,
                p.status,
                p.days_held,
                d.grade,
                d.score,
                d.stop_loss,
                d.target_1
            FROM performance_tracking p
            LEFT JOIN daily_signals d ON p.signal_id = d.id
            WHERE p.entry_date >= date('now', '-' || ? || ' days')
            ORDER BY p.entry_date DESC
        '''
        
        df = pd.read_sql_query(query, self.conn, params=[days])
        return df
    
    def export_to_csv(self, table_name, output_file=None):
        """Export any table to CSV for backup/analysis"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'{table_name}_export_{timestamp}.csv'
        
        df = pd.read_sql_query(f'SELECT * FROM {table_name}', self.conn)
        df.to_csv(output_file, index=False)
        print(f"📊 Exported {len(df)} rows to {output_file}")
        return output_file
    
    def get_statistics(self):
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Weekly watchlist count
        cursor.execute('SELECT COUNT(*) FROM weekly_watchlist')
        stats['weekly_total'] = cursor.fetchone()[0]
        
        # Daily signals count
        cursor.execute('SELECT COUNT(*) FROM daily_signals')
        stats['daily_total'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM daily_signals WHERE decision = 'BUY'")
        stats['buy_signals'] = cursor.fetchone()[0]
        
        # Unique scan dates
        cursor.execute('SELECT COUNT(DISTINCT DATE(scan_date)) FROM scan_metadata')
        stats['unique_scan_dates'] = cursor.fetchone()[0]
        
        # Database size
        stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


# ============================================================================
# WRAPPER FUNCTIONS - DROP-IN REPLACEMENT FOR CSV OPERATIONS
# ============================================================================

# Global database instance
_db = None

def get_db():
    """Get or create database instance"""
    global _db
    if _db is None:
        _db = ScanDatabase()
    return _db

def save_watchlist_with_db(df, csv_filename, universe='Unknown'):
    """
    Drop-in replacement for watchlist CSV save
    Saves to BOTH database and CSV (for backward compatibility)
    """
    # Original CSV save (for immediate use by Tier 2)
    df.to_csv(csv_filename, index=False)
    print(f"📄 CSV saved: {csv_filename} ({len(df)} stocks)")
    
    # NEW: Also save to database for history
    db = get_db()
    db.save_weekly_watchlist(df, universe)
    
    return csv_filename

def save_daily_signals_with_db(results_dict, csv_filename, universe='Unknown'):
    """
    Drop-in replacement for daily signals CSV save
    Saves to BOTH database and CSV
    """
    # Results dict should have 'buy', 'wait', 'skip' keys
    # Original CSV save is already handled by validator
    # Just add DB save
    
    db = get_db()
    db.save_daily_signals(results_dict, universe)
    
    print(f"📊 Daily signals saved to database (Universe: {universe})")
    
    return csv_filename


# ============================================================================
# QUERY HELPER FUNCTIONS
# ============================================================================

def get_historical_performance(symbol=None, days=30):
    """Get historical performance for a symbol or all signals"""
    db = get_db()
    
    if symbol:
        query = '''
            SELECT 
                scan_date,
                symbol,
                current_price,
                stop_loss,
                target_1,
                score,
                grade
            FROM daily_signals
            WHERE symbol = ? 
            AND decision = 'BUY'
            AND scan_date >= datetime('now', '-' || ? || ' days')
            ORDER BY scan_date DESC
        '''
        df = pd.read_sql_query(query, db.conn, params=[symbol, days])
    else:
        df = db.get_performance_data(days)
    
    return df

def get_scan_summary(days=7):
    """Get summary of recent scans"""
    db = get_db()
    return db.get_scan_history(days)

def export_full_history():
    """Export all tables to CSV for backup"""
    db = get_db()
    
    tables = ['weekly_watchlist', 'daily_signals', 'performance_tracking', 'scan_metadata']
    
    print("\n" + "="*80)
    print("📦 EXPORTING FULL SCAN HISTORY")
    print("="*80 + "\n")
    
    for table in tables:
        db.export_to_csv(table)
    
    stats = db.get_statistics()
    print("\n📊 Database Statistics:")
    print(f"  • Weekly watchlist entries: {stats['weekly_total']:,}")
    print(f"  • Daily signal entries: {stats['daily_total']:,}")
    print(f"  • Total BUY signals: {stats['buy_signals']:,}")
    print(f"  • Unique scan dates: {stats['unique_scan_dates']}")
    print(f"  • Database size: {stats['db_size_mb']:.2f} MB")
    print("\n✅ Full history exported!")


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def example_integration():
    """
    Example of how to integrate into your existing scanner
    NO CHANGES TO SCANNER LOGIC - just replace save operations
    """
    
    print("\n" + "="*80)
    print("🔧 HOW TO INTEGRATE")
    print("="*80 + "\n")
    
    print("OPTION 1 - Minimal Changes (Recommended)")
    print("-" * 80)
    print("""
# In your enhanced_momentum_V2_prefetch.py, FIND:
watchlist.to_csv('watchlist_momentum_current.csv', index=False)

# REPLACE WITH:
save_watchlist_with_db(watchlist, 'watchlist_momentum_current.csv', universe='Nifty_50')

# In your one_click_entry_system.py, FIND:
# (after validator.validate_all() returns results)

# ADD THIS LINE:
save_daily_signals_with_db(results_dict, results_file, universe='Nifty_50')
    """)
    
    print("\nOPTION 2 - Auto-Integration")
    print("-" * 80)
    print("""
# Add at the TOP of your scanner files:
from db_persistence import get_db, save_watchlist_with_db, save_daily_signals_with_db

# That's it! Just change your save calls to use the new functions.
    """)
    
    print("\n✅ Benefits:")
    print("  • Complete audit trail - nothing deleted")
    print("  • Fast queries by date, symbol, universe")
    print("  • Backward compatible - CSV still generated")
    print("  • Export to CSV anytime for Excel analysis")
    print("  • Track performance over months/years")


if __name__ == '__main__':
    # Initialize database
    db = get_db()
    
    # Show statistics
    stats = db.get_statistics()
    print("\n" + "="*80)
    print("📊 SCAN DATABASE STATUS")
    print("="*80)
    print(f"Database: {db.db_path}")
    print(f"Size: {stats['db_size_mb']:.2f} MB")
    print(f"Weekly entries: {stats['weekly_total']:,}")
    print(f"Daily entries: {stats['daily_total']:,}")
    print(f"BUY signals: {stats['buy_signals']:,}")
    print(f"Scan dates: {stats['unique_scan_dates']}")
    print("="*80 + "\n")
    
    # Show integration guide
    example_integration()
