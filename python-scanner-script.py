"""
NON-INTERACTIVE WRAPPER for two_tier_enhanced.py

This script is called by the Node.js server.
It takes command-line arguments to bypass the interactive prompts 
in the original script and runs the scan.
"""

import sys
import os
import pandas as pd
import glob
from datetime import datetime

# --- IMPORTANT ---
# This wrapper MUST be in the same directory as the other Python files
# so it can import their logic.
try:
    # We import the *classes* and *functions* from your scripts
    # Note: Using python-scanner-script.py to call these modules requires them to be present.
    # We are using the correct file names from the existing directory structure.
    from two_tier_enhanced import TwoTierEnhancedScanner, Config, logger, deduplicate_results
    # Import necessary classes explicitly, assuming they are defined in one_click_entry_system.py and enhanced_momentum_V2_prefetch.py
    from one_click_entry_system import FlexibleWatchlistEntryValidator
    from enhanced_momentum_V2_prefetch import SmartMoneyScanner, DataCache
except ImportError as e:
    # This block is essential for debugging import issues in the server environment
    print(f"Error: Failed to import scanner modules. {e}", file=sys.stderr)
    print("Please ensure 'python-scanner-script.py' is in the same directory as your other .py files (two_tier_enhanced.py, etc.) AND necessary classes are defined.", file=sys.stderr)
    sys.exit(1)

def load_symbols_from_choice(choice):
    """
    Non-interactive version of select_universe()
    Loads symbols based on the choice argument.
    """
    resource_path = Config.RESOURCE_PATH
    print(f"Wrapper: Loading symbols for choice '{choice}' from path '{resource_path}'")
    
    # HANDLE OPTION 6 (ALL)
    if choice == '6':
        print("Wrapper: Loading ALL files...")
        all_symbols = []
        files = glob.glob(os.path.join(resource_path, "*.csv"))
        
        for f in files:
            try:
                df = pd.read_csv(f)
                if 'Symbol' in df.columns:
                    # Clean up symbol formatting for consistency
                    syms = df['Symbol'].apply(lambda x: str(x).strip() if str(x).endswith('.NS') else f"{str(x).strip()}.NS").tolist()
                    all_symbols.extend(syms)
            except Exception as e:
                print(f"Wrapper: Warning - could not read {f}: {e}")
        
        unique_symbols = list(set(all_symbols))
        print(f"Wrapper: Loaded {len(unique_symbols)} unique stocks from {len(files)} files")
        return unique_symbols

    # HANDLE OPTIONS 1-5
    if choice in Config.STOCK_UNIVERSES:
        val = Config.STOCK_UNIVERSES[choice]
        if isinstance(val, tuple):
            filename, description = val
            path = os.path.join(resource_path, filename)
            if os.path.exists(path):
                print(f"Wrapper: Loading from {description} ({filename})")
                df = pd.read_csv(path)
                return df['Symbol'].apply(lambda x: str(x).strip() if str(x).endswith('.NS') else f"{str(x).strip()}.NS").tolist()
            else:
                print(f"Error: File not found: {path}", file=sys.stderr)
                return []
    
    print(f"Error: Invalid universe choice '{choice}'", file=sys.stderr)
    return []

def run_scan():
    if len(sys.argv) != 3:
        print("Error: Missing arguments.", file=sys.stderr)
        print("Usage: python python-scanner-script.py <main_choice> <universe_choice>", file=sys.stderr)
        sys.exit(1)
        
    main_choice = sys.argv[1]
    universe_choice = sys.argv[2]
    
    print(f"--- Wrapper: Starting Scan ---")
    print(f"Wrapper: Main Choice: {main_choice}")
    print(f"Wrapper: Universe Choice: {universe_choice}")
    
    logger.info("Wrapper: Loading symbols...")
    symbols = load_symbols_from_choice(universe_choice)
    
    if not symbols:
        logger.error(f"Wrapper: No symbols loaded for universe choice {universe_choice}. Exiting.")
        sys.exit(1)
    
    logger.info(f"Wrapper: Loaded {len(symbols)} symbols. Initializing scanner.")
    
    scanner = TwoTierEnhancedScanner()
    scanner.symbols = symbols
    
    # --- Execute Scan ---
    exit_code = 0
    try:
        if main_choice == '1':
            logger.info("Wrapper: Running WEEKLY scan...")
            scanner.run_with_retry('weekly')
        elif main_choice == '2':
            logger.info("Wrapper: Running DAILY scan...")
            scanner.run_with_retry('daily')
        elif main_choice == '3':
            logger.info("Wrapper: Running COMBINED scans...")
            scanner.run_with_retry('combined')
        else:
            logger.error(f"Error: Invalid main choice {main_choice}")
            exit_code = 1
    except Exception as e:
        logger.error(f"Wrapper: Scan failed during execution: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        exit_code = 1
    finally:
        # Crucial Cleanup: Delete temp_watchlist.csv if it exists to avoid File-In-Use errors
        temp_file = 'temp_watchlist.csv'
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Wrapper: Cleaned up temporary watchlist file: {temp_file}")
            except Exception as e:
                print(f"Wrapper: Warning - failed to delete temp file {temp_file}: {e}", file=sys.stderr)
                # We exit with 0 anyway, as Node should catch the close event.

        logger.info("--- Wrapper: Scan complete. ---")
        
        # Ensure the final log message is immediately sent to Node.js before exiting
        sys.stdout.flush() 
        sys.exit(exit_code)

if __name__ == "__main__":
    run_scan()