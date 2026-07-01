// server.js - Stock Scanner Backend with Smart Caching
require('dotenv').config();

const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs').promises;
const fsSync = require('fs');
const csv = require('csv-parser');
const createReadStream = require('fs').createReadStream;
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_CMD_ENV = process.env.PYTHON_CMD || 'py';

// ── SQLite DB (scan_history.db written by Python scanner) ─────────────────────
const DB_PATH = path.join(__dirname, 'scan_history.db');
let db = null;
if (fsSync.existsSync(DB_PATH)) {
  db = new sqlite3.Database(DB_PATH, sqlite3.OPEN_READONLY, (err) => {
    if (err) {
      console.error('⚠️  Could not open scan_history.db:', err.message);
      db = null;
    } else {
      console.log('🗄️  scan_history.db connected (read-only)');
    }
  });
} else {
  console.log('ℹ️  scan_history.db not found — DB endpoints will return empty');
}

// Helper: promisify db.all
function dbAll(query, params = []) {
  return new Promise((resolve, reject) => {
    if (!db) return resolve([]);
    db.all(query, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows || []);
    });
  });
}
function dbGet(query, params = []) {
  return new Promise((resolve, reject) => {
    if (!db) return resolve(null);
    db.get(query, params, (err, row) => {
      if (err) reject(err);
      else resolve(row || null);
    });
  });
}

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
  '6': 'ALL_Files',
  '7': 'Quality_Universe',
};

// Prune scan states older than 2 hours to prevent memory leak
setInterval(() => {
  const cutoff = Date.now() - 2 * 60 * 60 * 1000;
  for (const [id, scan] of activeScans.entries()) {
    if (scan.startTime && scan.startTime.getTime() < cutoff) {
      console.log(`🧹 Pruning stale scan state: ${id}`);
      activeScans.delete(id);
    }
  }
}, 30 * 60 * 1000); // run every 30 min

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
    // Find and move BUY file — always use __dirname so CWD doesn't matter
    if (mainChoice === '2' || mainChoice === '3') {
      const buyPattern = /^BUY_SIGNALS_.*\.csv$/;
      const buyFiles = (await fs.readdir(__dirname)).filter(f => buyPattern.test(f));

      if (buyFiles.length > 0) {
        const srcBuy = path.join(__dirname, buyFiles[0]);
        await fs.rename(srcBuy, files.buy);
        console.log(`📦 Cached: ${buyFiles[0]} → ${files.buy}`);
      }

      // Find and move WAIT file
      const waitPattern = /^WAIT_LIST_.*\.csv$/;
      const waitFiles = (await fs.readdir(__dirname)).filter(f => waitPattern.test(f));

      if (waitFiles.length > 0) {
        const srcWait = path.join(__dirname, waitFiles[0]);
        await fs.rename(srcWait, files.wait);
        console.log(`📦 Cached: ${waitFiles[0]} → ${files.wait}`);
      }
    }

    // Find and move WATCHLIST file
    if (mainChoice === '1' || mainChoice === '3') {
      const watchlistFile = path.join(__dirname, 'watchlist_momentum_current.csv');
      if (fsSync.existsSync(watchlistFile)) {
        await fs.rename(watchlistFile, files.watchlist);
        console.log(`📦 Cached: watchlist_momentum_current.csv → ${files.watchlist}`);
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

  // Input validation
  if (!SCAN_TYPE_MAP[mainChoice] || !UNIVERSE_MAP[universeChoice]) {
    return res.status(400).json({ error: `Invalid mainChoice "${mainChoice}" or universeChoice "${universeChoice}"` });
  }

  // Concurrent scan guard — one active scan at a time
  const running = [...activeScans.values()].find(s => !s.completed);
  if (running) {
    return res.status(409).json({
      error: 'A scan is already in progress. Wait for it to complete before starting another.',
      activeScanId: running.id,
    });
  }

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

  const pythonProcess = spawn(PYTHON_CMD_ENV, [
    path.join(__dirname, 'python-scanner-script.py'),
    mainChoice,
    universeChoice
  ]);

  // 10-minute hard timeout — kills Python if it hangs
  const scanTimeout = setTimeout(() => {
    if (!scanState.completed) {
      console.error(`[${scanId}] ⏰ Scan timed out after 10 min — killing process`);
      pythonProcess.kill('SIGTERM');
      scanState.error = 'Scan timed out after 10 minutes. Try a smaller universe or check Python logs.';
      scanState.completed = true;
    }
  }, 10 * 60 * 1000);

  pythonProcess.on('error', (err) => {
    clearTimeout(scanTimeout);
    const msg = `Cannot start Python ("${PYTHON_CMD_ENV}"): ${err.message}. Set PYTHON_CMD in .env to the correct executable.`;
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
    clearTimeout(scanTimeout);
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

    // Write to OS temp dir (not project root) so concurrent requests don't overwrite each other
    const tempScriptPath = path.join(os.tmpdir(), `price_fetch_${Date.now()}_${Math.random().toString(36).slice(2)}.py`);
    fsSync.writeFileSync(tempScriptPath, pythonScript);

    const pythonProcess = spawn(PYTHON_CMD_ENV, [tempScriptPath]);

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
        riskPct: parseFloat(signal['Risk_%'] || signal.Risk_Pct || 0),
        returnPct: returnPct,
        status: status
      };
    });

    // Calculate summary statistics
    const validReturns = signals.filter(s => s.returnPct !== null);
    const avgReturn = validReturns.length > 0
      ? validReturns.reduce((sum, s) => sum + s.returnPct, 0) / validReturns.length
      : 0;
    // winners = TARGET_HIT count (not just returnPct > 0, which is misleading)
    const winners = signals.filter(s => s.status === 'TARGET_HIT').length;

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
        vwapDeviation: parseFloat(signal['VWAP_Deviation_%'] || 0),
        marketTrend: signal.Market_Trend || 'Neutral',
        riskPct: parseFloat(signal['Risk_%'] || signal.Risk_Pct || 0),
        stopLoss: parseFloat(signal.Stop_Loss || 0),
        target1: parseFloat(signal.Target_1 || 0),
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

/**
 * Clear today's scan cache (or a specific date's cache) to force a fresh scan.
 * Query params: universe (optional), date (optional, defaults to today)
 */
app.delete('/api/cache/clear', async (req, res) => {
  // Sanitize params — only allow date format and universe key characters
  const rawDate = String(req.query.date || '');
  const rawUniverse = String(req.query.universe || '');
  const targetDate = /^\d{4}-\d{2}-\d{2}$/.test(rawDate)
    ? rawDate
    : new Date().toISOString().split('T')[0];
  const universe = /^[\w]+$/.test(rawUniverse) ? rawUniverse : '';

  try {
    const files = await fs.readdir(CACHE_DIR);
    let deleted = 0;

    for (const file of files) {
      if (!file.startsWith(targetDate)) continue;
      if (universe && !file.includes(universe)) continue;
      await fs.unlink(path.join(CACHE_DIR, file));
      deleted++;
      console.log(`🗑️  Deleted cache file: ${file}`);
    }

    console.log(`🧹 Cache cleared: ${deleted} files for date=${targetDate} universe=${universe || 'ALL'}`);
    res.json({ deleted, message: `Cleared ${deleted} cache file(s) for ${targetDate}` });
  } catch (err) {
    console.error('Cache clear error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    uptime: Math.round(process.uptime()),
    activeScans: activeScans.size,
    dbConnected: db !== null,
    pythonCmd: PYTHON_CMD_ENV,
    timestamp: new Date().toISOString(),
  });
});

// ── DB endpoints ──────────────────────────────────────────────────────────────

app.get('/api/db/stats', async (_req, res) => {
  try {
    const [daily, weekly, meta, lastScan] = await Promise.all([
      dbGet('SELECT COUNT(*) as n FROM daily_signals'),
      dbGet('SELECT COUNT(*) as n FROM weekly_watchlist'),
      dbGet('SELECT COUNT(*) as n FROM scan_metadata'),
      dbGet("SELECT MAX(DATE(scan_date)) as d FROM scan_metadata"),
    ]);
    res.json({
      dailySignals:   daily?.n  ?? 0,
      weeklyEntries:  weekly?.n ?? 0,
      totalScans:     meta?.n   ?? 0,
      lastScanDate:   lastScan?.d ?? null,
      dbSizeMb:       +(fsSync.statSync(DB_PATH).size / 1048576).toFixed(2),
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Rich signals from DB — includes wyckoff_phase, VWAP, POC, target_2, reasons
app.get('/api/db/signals', async (req, res) => {
  const { universe, date, decision = 'BUY', limit = 200 } = req.query;
  const safeUniverse = /^[\w ]+$/.test(String(universe || '')) ? universe : null;
  const safeDate     = /^\d{4}-\d{2}-\d{2}$/.test(String(date || '')) ? date : null;
  const safeDecision = ['BUY', 'WAIT', 'SKIP'].includes(String(decision).toUpperCase())
    ? String(decision).toUpperCase() : 'BUY';

  try {
    let query = `
      SELECT scan_date, symbol, decision, grade, score, wyckoff_phase,
             signal_price, current_price, price_change_pct,
             vwap, vwap_deviation_pct,
             poc, value_area_low, value_area_high, in_value_area,
             volume_ratio, above_ma20, above_ma50, ma_aligned,
             entry_low, entry_high, stop_loss, risk_pct, target_1, target_2,
             reasons, wait_factors, market_trend, market_strength, universe
      FROM daily_signals
      WHERE decision = ?`;
    const params = [safeDecision];

    if (safeDate) { query += ' AND DATE(scan_date) = DATE(?)'; params.push(safeDate); }
    if (safeUniverse) { query += ' AND universe LIKE ?'; params.push(`%${safeUniverse}%`); }
    query += ` ORDER BY scan_date DESC, score DESC LIMIT ?`;
    params.push(parseInt(limit) || 200);

    const rows = await dbAll(query, params);
    // Parse JSON fields
    const signals = rows.map(r => ({
      ...r,
      reasons:      tryParseJson(r.reasons,      []),
      wait_factors: tryParseJson(r.wait_factors, []),
    }));
    res.json({ signals, total: signals.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Available scan dates from DB (augments /api/performance/dates)
app.get('/api/db/dates', async (req, res) => {
  const { universe, decision = 'BUY' } = req.query;
  const safeUniverse = /^[\w ]+$/.test(String(universe || '')) ? universe : null;
  try {
    let query = `SELECT DISTINCT DATE(scan_date) as d FROM daily_signals WHERE decision = ?`;
    const params = [String(decision).toUpperCase()];
    if (safeUniverse) { query += ' AND universe LIKE ?'; params.push(`%${safeUniverse}%`); }
    query += ' ORDER BY d DESC LIMIT 60';
    const rows = await dbAll(query, params);
    res.json({ dates: rows.map(r => r.d) });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Scan history from metadata table
app.get('/api/db/scan-history', async (req, res) => {
  const days = Math.min(parseInt(req.query.days) || 30, 365);
  try {
    const rows = await dbAll(`
      SELECT DATE(scan_date) as date, scan_type, universe,
             total_signals, buy_signals, wait_signals, skip_signals,
             scan_duration_seconds
      FROM scan_metadata
      WHERE scan_date >= datetime('now', '-' || ? || ' days')
      ORDER BY scan_date DESC
    `, [days]);
    res.json({ history: rows });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Weekly watchlist history from DB
app.get('/api/db/watchlist', async (req, res) => {
  const { universe, limit = 100 } = req.query;
  const safeUniverse = /^[\w ]+$/.test(String(universe || '')) ? universe : null;
  try {
    let query = `
      SELECT DATE(scan_date) as date, symbol, stage,
             total_score, accumulation_score, breakout_score,
             current_price, stop_price, risk_reward,
             rs_score, base_quality, distance_to_resistance, universe
      FROM weekly_watchlist`;
    const params = [];
    if (safeUniverse) { query += ' WHERE universe LIKE ?'; params.push(`%${safeUniverse}%`); }
    query += ` ORDER BY scan_date DESC, total_score DESC LIMIT ?`;
    params.push(parseInt(limit) || 100);
    const rows = await dbAll(query, params);
    res.json({ watchlist: rows });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

function tryParseJson(val, fallback) {
  if (!val) return fallback;
  try { return JSON.parse(val); } catch { return fallback; }
}

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