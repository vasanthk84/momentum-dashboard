// server.js - Stock Scanner Backend with Smart Caching
const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const csv = require('csv-parser');
const createReadStream = require('fs').createReadStream;
const cors = require('cors');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

// Log every incoming request so we can diagnose issues
app.use((req, _res, next) => {
  console.log(`→ ${req.method} ${req.path}  body:`, JSON.stringify(req.body));
  next();
});

// Catch JSON parse errors from express.json() and return a clear 400
app.use((err, _req, res, next) => {
  if (err.type === 'entity.parse.failed') {
    console.error('JSON parse error:', err.message);
    return res.status(400).json({ error: 'Invalid JSON body: ' + err.message });
  }
  next(err);
});

// In production (after `npm run build`), serve the Vite dist output.
// In dev, Vite's own server handles static files and proxies /api to here.
const DIST_DIR = path.join(__dirname, 'dist');
if (fsSync.existsSync(DIST_DIR)) {
  app.use(express.static(DIST_DIR));
  console.log('📦 Serving built React app from dist/');
} else {
  // Fallback: serve project root (supports legacy plain index.html)
  app.use(express.static(path.join(__dirname)));
  console.log('⚡ No dist/ found — serving project root (run `npm run build` to build)');
}

const activeScans = new Map();

// Cache directory setup
const CACHE_DIR = path.join(__dirname, 'scan_cache');
if (!fsSync.existsSync(CACHE_DIR)) {
  fsSync.mkdirSync(CACHE_DIR);
  console.log(`📁 Created cache directory: ${CACHE_DIR}`);
}

// Map choices to descriptive names
const UNIVERSE_MAP = {
  '1': 'Nifty_50',
  '2': 'Nifty_Next_50',
  '3': 'Large_Cap',
  '4': 'Mid_Cap',
  '5': 'Small_Cap',
  '6': 'ALL_Files'
};

const SCAN_TYPE_MAP = {
  '1': 'WEEKLY',
  '2': 'DAILY',
  '3': 'COMBINED'
};

/**
 * Generate cache key based on date, scan type, and universe
 */
function getCacheKey(mainChoice, universeChoice) {
  const date = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
  const scanType = SCAN_TYPE_MAP[mainChoice];
  const universe = UNIVERSE_MAP[universeChoice];
  return `${date}_${scanType}_${universe}`;
}

/**
 * Get cache file paths for a specific cache key
 */
function getCacheFiles(cacheKey) {
  return {
    buy: path.join(CACHE_DIR, `${cacheKey}_BUY.csv`),
    wait: path.join(CACHE_DIR, `${cacheKey}_WAIT.csv`),
    watchlist: path.join(CACHE_DIR, `${cacheKey}_WATCHLIST.csv`),
    metadata: path.join(CACHE_DIR, `${cacheKey}_meta.json`)
  };
}

/**
 * Check if cache exists and is valid
 */
async function checkCache(cacheKey, mainChoice) {
  const files = getCacheFiles(cacheKey);

  try {
    // Check metadata
    const metaExists = fsSync.existsSync(files.metadata);
    if (!metaExists) return false;

    const metadata = JSON.parse(await fs.readFile(files.metadata, 'utf8'));

    // Verify cache is from today
    const today = new Date().toISOString().split('T')[0];
    if (metadata.date !== today) return false;

    // Check required files exist based on scan type
    if (mainChoice === '1' || mainChoice === '3') {
      // WEEKLY or COMBINED needs watchlist
      if (!fsSync.existsSync(files.watchlist)) return false;
    }

    if (mainChoice === '2' || mainChoice === '3') {
      // DAILY or COMBINED needs buy/wait
      if (!fsSync.existsSync(files.buy)) return false;
      // WAIT file is optional
    }

    console.log(`✅ Cache hit: ${cacheKey}`);
    return true;
  } catch (error) {
    console.log(`❌ Cache miss: ${cacheKey} - ${error.message}`);
    return false;
  }
}

/**
 * Move Python output files to cache with proper naming
 */
async function cacheResults(cacheKey, mainChoice) {
  const files = getCacheFiles(cacheKey);
  const today = new Date().toISOString().split('T')[0];

  try {
    // Find and move BUY file
    if (mainChoice === '2' || mainChoice === '3') {
      const buyPattern = /^BUY_SIGNALS_.*\.csv$/;
      const buyFiles = (await fs.readdir('.')).filter(f => buyPattern.test(f));

      if (buyFiles.length > 0) {
        const srcBuy = buyFiles[0];
        await fs.rename(srcBuy, files.buy);
        console.log(`📦 Cached: ${srcBuy} → ${files.buy}`);
      }

      // Find and move WAIT file
      const waitPattern = /^WAIT_LIST_.*\.csv$/;
      const waitFiles = (await fs.readdir('.')).filter(f => waitPattern.test(f));

      if (waitFiles.length > 0) {
        const srcWait = waitFiles[0];
        await fs.rename(srcWait, files.wait);
        console.log(`📦 Cached: ${srcWait} → ${files.wait}`);
      }
    }

    // Find and move WATCHLIST file
    if (mainChoice === '1' || mainChoice === '3') {
      const watchlistFile = 'watchlist_momentum_current.csv';
      if (fsSync.existsSync(watchlistFile)) {
        await fs.rename(watchlistFile, files.watchlist);
        console.log(`📦 Cached: ${watchlistFile} → ${files.watchlist}`);
      }
    }

    // Create metadata
    const metadata = {
      date: today,
      scanType: SCAN_TYPE_MAP[mainChoice],
      universe: UNIVERSE_MAP[Object.keys(UNIVERSE_MAP).find(k => cacheKey.includes(UNIVERSE_MAP[k]))],
      timestamp: new Date().toISOString(),
      cacheKey: cacheKey
    };

    await fs.writeFile(files.metadata, JSON.stringify(metadata, null, 2));
    console.log(`📦 Created metadata: ${files.metadata}`);

    return true;
  } catch (error) {
    console.error(`❌ Cache save error: ${error.message}`);
    return false;
  }
}

/**
 * Parse CSV file and return JSON
 */
function parseCSV(filePath) {
  return new Promise((resolve, reject) => {
    const results = [];
    console.log(`Reading: ${filePath}`);
    createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => results.push(data))
      .on('end', () => {
        console.log(`✓ Parsed ${filePath}: ${results.length} rows`);
        resolve(results);
      })
      .on('error', (error) => {
        console.error(`Error parsing ${filePath}: ${error.message}`);
        reject(error);
      });
  });
}

/**
 * Parse metrics string from CSV
 */
function parseMetrics(metricsString) {
  if (!metricsString) return [];

  let cleaned = metricsString.toString().trim();

  if (cleaned.startsWith('[') && cleaned.endsWith(']')) {
    cleaned = cleaned.slice(1, -1);

    const metrics = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < cleaned.length; i++) {
      const char = cleaned[i];
      const nextChar = cleaned[i + 1];

      if ((char === '"' || char === "'") && (i === 0 || cleaned[i - 1] !== '\\')) {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes && nextChar === ' ') {
        const item = current.trim().replace(/^["']|["']$/g, '');
        if (item) metrics.push(item);
        current = '';
        i++;
      } else {
        current += char;
      }
    }

    const lastItem = current.trim().replace(/^["']|["']$/g, '');
    if (lastItem) metrics.push(lastItem);

    return metrics;
  }

  return cleaned
    .split(',')
    .map(m => m.trim().replace(/^["']|["']$/g, ''))
    .filter(m => m.length > 0);
}

/**
 * Merge metrics from watchlist into results
 */
function mergeMetrics(results, watchlistData) {
  const metricsMap = {};

  console.log('Processing watchlist data for metrics...');

  if (watchlistData && watchlistData.length > 0) {
    watchlistData.forEach((row) => {
      const symbol = row.Symbol || row.symbol || row.SYMBOL;
      const signals = row.Signals || row.signals;

      if (symbol && signals) {
        const cleanSymbol = symbol.trim().toUpperCase();
        const parsedMetrics = parseMetrics(signals);
        metricsMap[cleanSymbol] = parsedMetrics;
      }
    });
  }

  console.log(`Total symbols with metrics in watchlist: ${Object.keys(metricsMap).length}`);

  if (results.buy) {
    results.buy = results.buy.map(stock => {
      const symbol = (stock.Symbol || stock.SYMBOL || stock.symbol || '').trim().toUpperCase();
      const metrics = metricsMap[symbol] || [];

      return {
        ...stock,
        metrics: metrics
      };
    });
  }

  if (results.wait) {
    results.wait = results.wait.map(stock => {
      const symbol = (stock.Symbol || stock.SYMBOL || stock.symbol || '').trim().toUpperCase();
      const metrics = metricsMap[symbol] || [];

      return {
        ...stock,
        metrics: metrics
      };
    });
  }

  return results;
}

/**
 * Clean up old cache files (older than 7 days)
 */
async function cleanupOldCache() {
  try {
    const files = await fs.readdir(CACHE_DIR);
    const now = Date.now();
    const maxAge = 365 * 24 * 60 * 60 * 1000; // 365 days (1 year)

    let cleaned = 0;
    for (const file of files) {
      const filePath = path.join(CACHE_DIR, file);
      const stats = await fs.stat(filePath);

      if (now - stats.mtime.getTime() > maxAge) {
        await fs.unlink(filePath);
        cleaned++;
      }
    }

    if (cleaned > 0) {
      console.log(`🧹 Cleaned up ${cleaned} old cache files`);
    }
  } catch (error) {
    console.error(`Cache cleanup error: ${error.message}`);
  }
}

/**
 * Start a new scan or load from cache
 */
app.post('/api/start-scan', async (req, res) => {
  const { mainChoice, universeChoice } = req.body;
  const scanId = `scan_${Date.now()}`;
  const cacheKey = getCacheKey(mainChoice, universeChoice);

  console.log(`[${scanId}] Scan request: ${SCAN_TYPE_MAP[mainChoice]} - ${UNIVERSE_MAP[universeChoice]}`);
  console.log(`[${scanId}] Cache key: ${cacheKey}`);

  const scanState = {
    id: scanId,
    progress: 'Checking cache...',
    logs: '',
    completed: false,
    results: null,
    error: null,
    fromCache: false,
    startTime: new Date()
  };
  activeScans.set(scanId, scanState);

  res.json({ scanId });

  // Cleanup old cache in background
  cleanupOldCache().catch(err => console.error('Cache cleanup failed:', err));

  // Check cache first
  const cacheExists = await checkCache(cacheKey, mainChoice);

  if (cacheExists) {
    console.log(`[${scanId}] ✅ Loading from cache: ${cacheKey}`);
    scanState.progress = `Loading cached results from ${cacheKey}...`;
    scanState.fromCache = true;

    try {
      const files = getCacheFiles(cacheKey);
      const results = {
        buy: [],
        wait: [],
        watchlist: []
      };

      if (fsSync.existsSync(files.buy)) {
        results.buy = await parseCSV(files.buy);
      }

      if (fsSync.existsSync(files.wait)) {
        results.wait = await parseCSV(files.wait);
      }

      if (fsSync.existsSync(files.watchlist)) {
        results.watchlist = await parseCSV(files.watchlist);
        // Merge metrics
        scanState.results = mergeMetrics(results, results.watchlist);
      } else {
        scanState.results = results;
      }

      scanState.progress = 'Loaded from cache';
      scanState.completed = true;
      scanState.logs = `✅ Results loaded from cache (${cacheKey})\nNo scan needed - using today's cached results.`;

      console.log(`[${scanId}] Cache load complete: ${results.buy.length} BUY, ${results.wait.length} WAIT, ${results.watchlist.length} WATCHLIST`);

      return;
    } catch (error) {
      console.error(`[${scanId}] Cache load failed: ${error.message}`);
      scanState.progress = 'Cache load failed, running fresh scan...';
      scanState.fromCache = false;
    }
  }

  // Run fresh scan
  console.log(`[${scanId}] 🔄 Running fresh Python scan...`);
  scanState.progress = 'Starting Python scan...';
  scanState.fromCache = false;

  // Use 'python' (most common on Windows with standard installer).
  // If your system uses 'py' launcher or 'python3', change this env var or update below.
  const PYTHON_CMD = process.env.PYTHON_CMD || 'py';
  const pythonProcess = spawn(PYTHON_CMD, [
    'python-scanner-script.py',
    mainChoice,
    universeChoice
  ]);

  pythonProcess.on('error', (err) => {
    const msg = `Cannot start Python ("${PYTHON_CMD}"): ${err.message}. Set the PYTHON_CMD env var to the correct executable.`;
    console.error(`[${scanId}] ${msg}`);
    scanState.error = msg;
    scanState.completed = true;
  });

  pythonProcess.stdout.on('data', (data) => {
    const logChunk = data.toString();
    scanState.logs += logChunk;
    scanState.progress = logChunk.split('\n').filter(Boolean).pop() || scanState.progress;
    console.log(`[${scanId} STDOUT]: ${logChunk}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    const errorChunk = data.toString();
    scanState.logs += `[STDERR] ${errorChunk}\n`;

    const isWarning = errorChunk.includes('possibly delisted') ||
      errorChunk.includes('no timezone found') ||
      errorChunk.includes('Warning:') ||
      errorChunk.includes('FutureWarning') ||
      errorChunk.includes('DeprecationWarning');

    if (!isWarning) {
      scanState.error = (scanState.error || '') + errorChunk;
    }

    console.error(`[${scanId} STDERR]: ${errorChunk}`);
  });

  pythonProcess.on('close', async (code) => {
    console.log(`[${scanId}] Python script exited with code ${code}`);

    if (code !== 0 && scanState.error && scanState.error.trim().length > 0) {
      scanState.completed = true;
      return;
    }

    if (code === 0) {
      scanState.error = null;
    }

    try {
      scanState.progress = 'Caching results...';

      // Cache the results
      await cacheResults(cacheKey, mainChoice);

      // Load cached results
      const files = getCacheFiles(cacheKey);
      const results = {
        buy: [],
        wait: [],
        watchlist: []
      };

      if (fsSync.existsSync(files.buy)) {
        results.buy = await parseCSV(files.buy);
      }

      if (fsSync.existsSync(files.wait)) {
        results.wait = await parseCSV(files.wait);
      }

      if (fsSync.existsSync(files.watchlist)) {
        results.watchlist = await parseCSV(files.watchlist);
        scanState.results = mergeMetrics(results, results.watchlist);
      } else {
        scanState.results = results;
      }

      scanState.progress = 'Scan complete and cached.';
      console.log(`[${scanId}] Scan complete and cached: ${results.buy.length} BUY, ${results.wait.length} WAIT, ${results.watchlist.length} WATCHLIST`);

    } catch (err) {
      console.error(`[${scanId}] Failed to process results: ${err.message}`);
      scanState.error = 'Failed to process scan results.';
    } finally {
      scanState.completed = true;
    }
  });
});

/**
 * Check scan status
 */
app.get('/api/scan-status/:scanId', (req, res) => {
  const { scanId } = req.params;
  const scan = activeScans.get(scanId);

  if (!scan) {
    return res.status(404).json({ error: 'Scan not found.' });
  }

  res.json({
    completed: scan.completed,
    error: scan.error,
    progress: scan.progress,
    fromCache: scan.fromCache
  });
});

/**
 * Get final scan results
 */
app.get('/api/scan-results/:scanId', (req, res) => {
  const { scanId } = req.params;
  const scan = activeScans.get(scanId);

  if (!scan) {
    return res.status(404).json({ error: 'Scan not found.' });
  }

  if (!scan.completed) {
    return res.status(202).json({ message: 'Scan still in progress.' });
  }

  res.json({
    logs: scan.logs,
    results: scan.results,
    error: scan.error,
    fromCache: scan.fromCache
  });

  activeScans.delete(scanId);
});


/**
 * Get available dates for performance tracking (filtered by universe)
 */
app.get('/api/performance/dates', async (req, res) => {
  const { universe } = req.query;

  try {
    const files = await fs.readdir(CACHE_DIR);
    const dates = new Set();

    console.log('📅 Scanning cache directory for dates...');
    console.log('   Universe filter:', universe || 'ALL');
    console.log('   Found files:', files.length);

    // Extract unique dates from cache files, filtered by universe
    files.forEach(file => {
      if (universe) {
        // Filter by specific universe and acceptable scan types (DAILY or COMBINED)
        const pattern_daily = new RegExp(`^(\\d{4}-\\d{2}-\\d{2})_DAILY_${universe}_`);
        const pattern_combined = new RegExp(`^(\\d{4}-\\d{2}-\\d{2})_COMBINED_${universe}_`);

        let match = file.match(pattern_daily);

        if (!match) {
          match = file.match(pattern_combined);
        }

        if (match) {
          dates.add(match[1]);
          console.log('   ✓ Found date:', match[1], 'from', file);
        }
      } else {
        // No filter, show all DAILY or COMBINED scans
        const match_daily = file.match(/^(\d{4}-\d{2}-\d{2})_DAILY_/);
        const match_combined = file.match(/^(\d{4}-\d{2}-\d{2})_COMBINED_/);

        let match = match_daily || match_combined;

        if (match) {
          dates.add(match[1]);
          console.log('   ✓ Found date:', match[1], 'from', file);
        }
      }
    });

    // Sort dates in descending order (newest first)
    const sortedDates = Array.from(dates).sort().reverse();

    console.log(`📅 Available dates for ${universe || 'ALL'}:`, sortedDates);
    res.json({ dates: sortedDates });
  } catch (error) {
    console.error('❌ Error getting dates:', error);
    res.status(500).json({ error: 'Failed to load dates' });
  }
});

/**
 * Fetch current prices for symbols using Python/yfinance
 */
async function fetchCurrentPrices(symbols) {
  return new Promise((resolve, reject) => {
    // Create a temporary Python script to fetch prices
    const pythonScript = `
import yfinance as yf
import json
import sys

symbols = ${JSON.stringify(symbols)}
prices = {}

for symbol in symbols:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        if not hist.empty:
            prices[symbol] = float(hist['Close'].iloc[-1])
        else:
            prices[symbol] = None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}", file=sys.stderr)
        prices[symbol] = None

print(json.dumps(prices))
`;

    const tempScriptPath = path.join(__dirname, 'temp_price_fetch.py');
    fsSync.writeFileSync(tempScriptPath, pythonScript);

    const pythonProcess = spawn('python', [tempScriptPath]);

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    pythonProcess.on('close', (code) => {
      // Clean up temp file
      try {
        fsSync.unlinkSync(tempScriptPath);
      } catch (err) {
        console.error('Failed to delete temp script:', err);
      }

      if (code !== 0) {
        console.error('Python price fetch failed:', errorOutput);
        return resolve({}); // Return empty object on failure
      }

      try {
        const prices = JSON.parse(output.trim());
        resolve(prices);
      } catch (err) {
        console.error('Failed to parse price data:', err);
        resolve({});
      }
    });

    // Timeout after 60 seconds
    setTimeout(() => {
      pythonProcess.kill();
      resolve({});
    }, 60000);
  });
}

/**
 * Analyze performance of a historical scan
 */
app.get('/api/performance/analyze', async (req, res) => {
  const { date, universe } = req.query;

  console.log(`\n📊 Performance Analysis Request:`);
  console.log(`   Date: ${date}`);
  console.log(`   Universe: ${universe}`);

  if (!date || !universe) {
    return res.status(400).json({ error: 'Date and universe required' });
  }

  try {
    let cacheKey = `${date}_DAILY_${universe}`;
    let buyFile = path.join(CACHE_DIR, `${cacheKey}_BUY.csv`);

    console.log(`   Looking for: ${buyFile}`);
    console.log(`   File exists: ${fsSync.existsSync(buyFile)}`);

    // 2. If not found, check for the file saved as a COMBINED scan
    if (!fsSync.existsSync(buyFile)) {
      console.log(`   DAILY file not found. Checking for COMBINED scan...`);
      cacheKey = `${date}_COMBINED_${universe}`;
      buyFile = path.join(CACHE_DIR, `${cacheKey}_BUY.csv`);
    }

    console.log(`   Looking for: ${buyFile}`);
    console.log(`   File exists: ${fsSync.existsSync(buyFile)}`);

    if (!fsSync.existsSync(buyFile)) {
      // List what files DO exist for debugging
      const allFiles = await fs.readdir(CACHE_DIR);
      console.log(`   Available cache files:`, allFiles.filter(f => f.includes(date)));
      return res.status(404).json({
        error: `No scan data found for ${date} and ${universe}`,
        hint: 'Make sure you have run a DAILY or COMBINED scan for this date and universe'
      });
    }

    console.log(`   ✓ Found BUY file, parsing...`);

    // Read historical BUY signals
    const historicalSignals = await parseCSV(buyFile);

    if (!historicalSignals || historicalSignals.length === 0) {
      return res.status(404).json({ error: 'No BUY signals found in file' });
    }

    console.log(`   ✓ Loaded ${historicalSignals.length} signals`);

    // Calculate days held
    const scanDate = new Date(date);
    const today = new Date();
    const daysHeld = Math.floor((today - scanDate) / (1000 * 60 * 60 * 24));

    console.log(`   Days held: ${daysHeld}`);

    // Fetch current prices for all symbols using yfinance via Python
    const symbols = historicalSignals.map(s => s.Symbol);
    console.log(`   Fetching current prices for ${symbols.length} symbols...`);

    const currentPrices = await fetchCurrentPrices(symbols);

    console.log(`   ✓ Fetched ${Object.keys(currentPrices).length} prices`);

    // Calculate performance for each signal
    const signals = historicalSignals.map(signal => {
      const symbol = signal.Symbol;
      const signalPrice = parseFloat(signal.Current_Price || signal.Signal_Price || 0);
      const stopLoss = parseFloat(signal.Stop_Loss || 0);
      const target1 = parseFloat(signal.Target_1 || 0);
      const currentPrice = currentPrices[symbol];

      let returnPct = null;
      let status = 'NO_DATA';

      if (currentPrice && signalPrice) {
        returnPct = ((currentPrice - signalPrice) / signalPrice) * 100;

        // Determine status
        if (target1 && currentPrice >= target1) {
          status = 'TARGET_HIT';
        } else if (stopLoss && currentPrice <= stopLoss) {
          status = 'STOP_HIT';
        } else {
          status = 'OPEN';
        }
      }

      return {
        symbol: symbol,
        grade: signal.Grade || 'N/A',
        score: parseInt(signal.Score) || 0,
        signalPrice: signalPrice,
        currentPrice: currentPrice,
        stopLoss: stopLoss,
        target1: target1,
        returnPct: returnPct,
        status: status
      };
    });

    // Calculate summary statistics
    const validReturns = signals.filter(s => s.returnPct !== null);
    const avgReturn = validReturns.length > 0
      ? validReturns.reduce((sum, s) => sum + s.returnPct, 0) / validReturns.length
      : 0;
    const winners = validReturns.filter(s => s.returnPct > 0).length;

    console.log(`   ✓ Analysis complete: Avg return ${avgReturn.toFixed(2)}%, ${winners}/${validReturns.length} winners`);

    res.json({
      scanDate: date,
      daysHeld: daysHeld,
      totalSignals: signals.length,
      avgReturn: avgReturn,
      winners: winners,
      signals: signals.sort((a, b) => (b.returnPct || 0) - (a.returnPct || 0))
    });

  } catch (error) {
    console.error('❌ Performance analysis error:', error);
    res.status(500).json({ error: 'Failed to analyze performance: ' + error.message });
  }
});

/**
 * BATCH ANALYTICS ENDPOINT
 * Analyzes multiple past scans to generate aggregate statistics
 */
app.get('/api/analytics/batch', async (req, res) => {
  const { universe, limit = 5 } = req.query; // Default to last 5 scans

  if (!universe) {
    return res.status(400).json({ error: 'Universe is required' });
  }

  console.log(`\n📊 Batch Analytics Request: ${universe} (Last ${limit} scans)`);

  try {
    // 1. Get all available dates
    const files = await fs.readdir(CACHE_DIR);
    const dates = new Set();
    
    files.forEach(file => {
      // Match DAILY or COMBINED files for this universe
      const dailyMatch = file.match(new RegExp(`^(\\d{4}-\\d{2}-\\d{2})_DAILY_${universe}_`));
      const combinedMatch = file.match(new RegExp(`^(\\d{4}-\\d{2}-\\d{2})_COMBINED_${universe}_`));
      
      if (dailyMatch) dates.add(dailyMatch[1]);
      if (combinedMatch) dates.add(combinedMatch[1]);
    });

    // Sort dates descending and take the requested limit
    const recentDates = Array.from(dates).sort().reverse().slice(0, parseInt(limit));
    
    if (recentDates.length === 0) {
      return res.status(404).json({ error: 'No scan history found for this universe' });
    }

    console.log(`   Analyzing dates: ${recentDates.join(', ')}`);

    // 2. Load signals from these dates
    let allSignals = [];
    const symbolsToFetch = new Set();

    for (const date of recentDates) {
      // Try DAILY then COMBINED
      let filePath = path.join(CACHE_DIR, `${date}_DAILY_${universe}_BUY.csv`);
      if (!fsSync.existsSync(filePath)) {
        filePath = path.join(CACHE_DIR, `${date}_COMBINED_${universe}_BUY.csv`);
      }

      if (fsSync.existsSync(filePath)) {
        const signals = await parseCSV(filePath);
        signals.forEach(s => {
            // Attach scan date to signal for aging calc
            s._scanDate = date; 
            allSignals.push(s);
            symbolsToFetch.add(s.Symbol);
        });
      }
    }

    console.log(`   Loaded ${allSignals.length} total signals across ${recentDates.length} days`);
    console.log(`   Fetching live prices for ${symbolsToFetch.size} unique symbols...`);

    // 3. Batch fetch current prices
    // Note: If list is huge (>500), you might want to chunk this. 
    // For now, assuming reasonable size < 200.
    const currentPrices = await fetchCurrentPrices(Array.from(symbolsToFetch));

    // 4. Calculate Performance Metrics
    const analyzedSignals = allSignals.map(signal => {
      const symbol = signal.Symbol;
      const entryPrice = parseFloat(signal.Current_Price || signal.Signal_Price || 0);
      const currentPrice = currentPrices[symbol];
      const scanDate = new Date(signal._scanDate);
      const today = new Date();
      const daysHeld = Math.floor((today - scanDate) / (1000 * 60 * 60 * 24));

      let returnPct = 0;
      let status = 'OPEN';

      if (currentPrice && entryPrice) {
        returnPct = ((currentPrice - entryPrice) / entryPrice) * 100;
        
        const stopLoss = parseFloat(signal.Stop_Loss || 0);
        const target1 = parseFloat(signal.Target_1 || 0);

        if (target1 && currentPrice >= target1) status = 'TARGET_HIT';
        else if (stopLoss && currentPrice <= stopLoss) status = 'STOP_HIT';
        else status = returnPct > 0 ? 'PROFIT' : 'LOSS';
      }

      return {
        symbol,
        date: signal._scanDate,
        grade: signal.Grade || 'N/A',
        score: parseInt(signal.Score) || 0,
        returnPct,
        daysHeld,
        status,
        volumeRatio: parseFloat(signal.Volume_Ratio || 0),
        vwapDeviation: parseFloat(signal['VWAP_Deviation_%'] || 0), // Handle special char key
        marketTrend: signal.Market_Trend || 'Neutral'
      };
    });

    res.json({
      dates: recentDates,
      totalSignals: analyzedSignals.length,
      signals: analyzedSignals
    });

  } catch (error) {
    console.error('❌ Batch analytics error:', error);
    res.status(500).json({ error: error.message });
  }
});

// SPA fallback — MUST be last; serves index.html for any non-API route
app.get('*', (req, res) => {
  const indexPath = fsSync.existsSync(DIST_DIR)
    ? path.join(DIST_DIR, 'index.html')
    : path.join(__dirname, 'index.html');
  res.sendFile(indexPath);
});

app.listen(PORT, () => {
  console.log(`🚀 Momentum Scanner UI running at http://localhost:${PORT}`);
  console.log(`📁 Cache directory: ${CACHE_DIR}`);
});