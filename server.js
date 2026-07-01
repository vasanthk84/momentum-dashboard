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
const https = require('https');
const cron = require('node-cron');

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

// Reconnect DB after Python scanner creates it on first run
function openDbIfNeeded() {
  if (db !== null) return;
  if (!fsSync.existsSync(DB_PATH)) return;
  db = new sqlite3.Database(DB_PATH, sqlite3.OPEN_READONLY, (err) => {
    if (err) { console.error('⚠️  Could not reopen scan_history.db:', err.message); db = null; }
    else console.log('🗄️  scan_history.db reconnected (read-only)');
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

// Map choices to descriptive names (used for cache keys and display)
const UNIVERSE_MAP = {
  '1': 'Nifty_50',
  '2': 'Nifty_Next_50',
  '3': 'Large_Cap',
  '4': 'Mid_Cap',
  '5': 'Small_Cap',
  '6': 'ALL_Files',
  '7': 'Quality_Universe',
};

// Exact universe names as stored by Python in the DB
const DB_UNIVERSE_MAP = {
  '1': 'Nifty 50',
  '2': 'Nifty Next 50',
  '3': 'Large Cap',
  '4': 'Mid Cap',
  '5': 'Small Cap',
  '6': 'ALL_Files',
  '7': 'Quality Universe',
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

// ── DB-first helpers ──────────────────────────────────────────────────────────

/** Normalize a daily_signals DB row to the CSV-column format the frontend expects */
function normalizeDbSignal(row) {
  const reasons      = tryParseJson(row.reasons,      []);
  const waitFactors  = tryParseJson(row.wait_factors, []);
  return {
    Symbol:               row.symbol,
    Grade:                row.grade,
    Score:                row.score,
    Current_Price:        row.signal_price ?? row.current_price,
    Entry_Low:            row.entry_low,
    Entry_High:           row.entry_high,
    Stop_Loss:            row.stop_loss,
    Target_1:             row.target_1,
    Target_2:             row.target_2,
    'Risk_%':             row.risk_pct,
    Volume_Ratio:         row.volume_ratio,
    Market_Trend:         row.market_trend,
    VWAP:                 row.vwap,
    'VWAP_Deviation_%':   row.vwap_deviation_pct,
    Wyckoff_Phase:        row.wyckoff_phase,
    Reason:               waitFactors.join('; ') || null,
    metrics:              reasons,
  };
}

/** Normalize a weekly_watchlist DB row to the CSV-column format the frontend expects */
function normalizeDbWatchlist(row) {
  const signals = tryParseJson(row.signals, []);
  return {
    Symbol:       row.symbol,
    Stage:        row.stage,
    Total_Score:  row.total_score,
    Current_Price: row.current_price ?? row.signal_price,
    Stop_Price:   row.stop_price,
    Risk_Reward:  row.risk_reward,
    RS_Score:     row.rs_score,
    Base_Quality: row.base_quality,
    metrics:      signals,
  };
}

/** Check if today's scan data exists in DB (replaces checkCache) */
async function checkDbCache(mainChoice, universeChoice) {
  if (!db) return false;
  const dbUniverse = DB_UNIVERSE_MAP[universeChoice] || UNIVERSE_MAP[universeChoice];
  const today = new Date().toISOString().split('T')[0];
  try {
    if (mainChoice === '1') {
      const row = await dbGet(
        `SELECT COUNT(*) as n FROM weekly_watchlist WHERE DATE(scan_date) = DATE(?) AND universe = ?`,
        [today, dbUniverse]
      );
      return (row?.n ?? 0) > 0;
    } else {
      const row = await dbGet(
        `SELECT COUNT(*) as n FROM daily_signals WHERE DATE(scan_date) = DATE(?) AND universe = ?`,
        [today, dbUniverse]
      );
      return (row?.n ?? 0) > 0;
    }
  } catch { return false; }
}

/** Load today's scan results from DB (replaces CSV parseCSV flow) */
async function loadFromDb(mainChoice, universeChoice) {
  const dbUniverse = DB_UNIVERSE_MAP[universeChoice] || UNIVERSE_MAP[universeChoice];
  const today = new Date().toISOString().split('T')[0];

  const needsDaily     = mainChoice === '2' || mainChoice === '3';
  const needsWatchlist = mainChoice === '1' || mainChoice === '3';

  const [buyRows, waitRows, watchlistRows] = await Promise.all([
    needsDaily
      ? dbAll(`SELECT * FROM daily_signals WHERE decision='BUY'  AND DATE(scan_date)=DATE(?) AND universe=? ORDER BY score DESC`, [today, dbUniverse])
      : Promise.resolve([]),
    needsDaily
      ? dbAll(`SELECT * FROM daily_signals WHERE decision='WAIT' AND DATE(scan_date)=DATE(?) AND universe=? ORDER BY score DESC`, [today, dbUniverse])
      : Promise.resolve([]),
    needsWatchlist
      ? dbAll(`SELECT * FROM weekly_watchlist WHERE DATE(scan_date)=DATE(?) AND universe=? ORDER BY total_score DESC`, [today, dbUniverse])
      : Promise.resolve([]),
  ]);

  return {
    buy:       buyRows.map(normalizeDbSignal),
    wait:      waitRows.map(normalizeDbSignal),
    watchlist: watchlistRows.map(normalizeDbWatchlist),
  };
}

/** Delete leftover CSV files Python emits (BUY_SIGNALS_*, WAIT_LIST_*, watchlist_momentum_current.csv) */
async function cleanupScanCsvFiles() {
  try {
    const files = await fs.readdir(__dirname);
    for (const f of files) {
      if (/^(BUY_SIGNALS_|WAIT_LIST_).*\.csv$/.test(f) || f === 'watchlist_momentum_current.csv') {
        await fs.unlink(path.join(__dirname, f)).catch(() => {});
        console.log(`🧹 Removed leftover CSV: ${f}`);
      }
    }
  } catch (err) {
    console.error('CSV cleanup error:', err.message);
  }
}

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

  // Check DB first — skip Python if today's data already exists
  const dbCached = await checkDbCache(mainChoice, universeChoice);

  if (dbCached) {
    console.log(`[${scanId}] ✅ DB cache hit — loading from database`);
    scanState.progress = 'Loading today\'s results from database...';
    scanState.fromCache = true;
    try {
      scanState.results = await loadFromDb(mainChoice, universeChoice);
      const { buy, wait, watchlist } = scanState.results;
      scanState.progress = 'Loaded from DB';
      scanState.completed = true;
      scanState.logs = `✅ Results loaded from database (${new Date().toISOString().split('T')[0]})\nBUY: ${buy.length}  WAIT: ${wait.length}  WATCHLIST: ${watchlist.length}`;
      console.log(`[${scanId}] DB load: ${buy.length} BUY, ${wait.length} WAIT, ${watchlist.length} WATCHLIST`);
    } catch (err) {
      console.error(`[${scanId}] DB load failed: ${err.message} — falling through to fresh scan`);
      scanState.fromCache = false;
    }
    if (scanState.completed) return;
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
      scanState.progress = 'Reading results from database...';

      // Clean up CSV files Python wrote (DB is the source of truth now)
      await cleanupScanCsvFiles();

      // Reconnect DB if Python just created it on first run
      openDbIfNeeded();
      // Brief pause so SQLite WAL checkpoint completes
      await new Promise(r => setTimeout(r, 800));

      const results = await loadFromDb(mainChoice, universeChoice);
      scanState.results = results;

      const { buy, wait, watchlist } = results;
      scanState.progress = 'Scan complete.';
      console.log(`[${scanId}] Scan complete (DB): ${buy.length} BUY, ${wait.length} WAIT, ${watchlist.length} WATCHLIST`);

    } catch (err) {
      console.error(`[${scanId}] Failed to read DB results: ${err.message}`);
      scanState.error = 'Scan ran but failed to read results from database.';
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
 * Get available dates for performance tracking — queries DB directly
 */
app.get('/api/performance/dates', async (req, res) => {
  const { universe } = req.query;
  try {
    let query = `SELECT DISTINCT DATE(scan_date) as d FROM daily_signals WHERE 1=1`;
    const params = [];
    if (universe) {
      // universe param comes in as UNIVERSE_MAP value (underscore format); find the DB name
      const choice = Object.keys(UNIVERSE_MAP).find(k => UNIVERSE_MAP[k] === universe);
      const dbUniverse = choice ? (DB_UNIVERSE_MAP[choice] || universe) : universe;
      query += ` AND universe = ?`;
      params.push(dbUniverse);
    }
    query += ` ORDER BY d DESC LIMIT 90`;
    const rows = await dbAll(query, params);
    const dates = rows.map(r => r.d).filter(Boolean);
    console.log(`📅 DB dates for ${universe || 'ALL'}: ${dates.length} found`);
    res.json({ dates });
  } catch (err) {
    console.error('❌ Error getting dates from DB:', err);
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
 * Analyze performance of a historical scan — reads from DB
 */
app.get('/api/performance/analyze', async (req, res) => {
  const { date, universe } = req.query;

  console.log(`\n📊 Performance Analysis: date=${date} universe=${universe}`);

  if (!date || !universe) {
    return res.status(400).json({ error: 'Date and universe required' });
  }

  try {
    // universe param arrives as UNIVERSE_MAP value (underscore); resolve to DB name
    const choice = Object.keys(UNIVERSE_MAP).find(k => UNIVERSE_MAP[k] === universe);
    const dbUniverse = choice ? (DB_UNIVERSE_MAP[choice] || universe) : universe;

    const rows = await dbAll(
      `SELECT symbol, grade, score, signal_price, current_price, stop_loss, target_1, risk_pct
       FROM daily_signals
       WHERE decision = 'BUY' AND DATE(scan_date) = DATE(?) AND universe = ?
       ORDER BY score DESC`,
      [date, dbUniverse]
    );

    if (!rows.length) {
      return res.status(404).json({
        error: `No BUY signals in DB for ${date} / ${universe}`,
        hint: 'Run a Daily scan for this date and universe first',
      });
    }

    console.log(`   ✓ ${rows.length} signals from DB`);

    const scanDate = new Date(date);
    const daysHeld = Math.floor((Date.now() - scanDate) / (1000 * 60 * 60 * 24));

    const symbols = rows.map(r => r.symbol);
    console.log(`   Fetching live prices for ${symbols.length} symbols…`);
    const currentPrices = await fetchCurrentPrices(symbols);
    console.log(`   ✓ ${Object.keys(currentPrices).length} prices received`);

    const signals = rows.map(r => {
      const signalPrice  = r.signal_price ?? r.current_price ?? 0;
      const currentPrice = currentPrices[r.symbol] ?? null;
      const stopLoss     = r.stop_loss ?? 0;
      const target1      = r.target_1  ?? 0;

      let returnPct = null;
      let status = 'NO_DATA';
      if (currentPrice && signalPrice) {
        returnPct = ((currentPrice - signalPrice) / signalPrice) * 100;
        if (target1 && currentPrice >= target1)     status = 'TARGET_HIT';
        else if (stopLoss && currentPrice <= stopLoss) status = 'STOP_HIT';
        else                                           status = 'OPEN';
      }

      return {
        symbol:       r.symbol,
        grade:        r.grade || 'N/A',
        score:        r.score ?? 0,
        signalPrice,
        currentPrice,
        stopLoss,
        target1,
        riskPct:      r.risk_pct ?? 0,
        returnPct,
        status,
      };
    });

    const valid   = signals.filter(s => s.returnPct !== null);
    const avgReturn = valid.length
      ? valid.reduce((sum, s) => sum + s.returnPct, 0) / valid.length
      : 0;
    const winners = signals.filter(s => s.status === 'TARGET_HIT').length;

    console.log(`   ✓ avgReturn=${avgReturn.toFixed(2)}% winners=${winners}/${valid.length}`);
    res.json({
      scanDate: date,
      daysHeld,
      totalSignals: signals.length,
      avgReturn,
      winners,
      signals: signals.sort((a, b) => (b.returnPct ?? 0) - (a.returnPct ?? 0)),
    });

  } catch (err) {
    console.error('❌ Performance analysis error:', err);
    res.status(500).json({ error: 'Failed to analyze performance: ' + err.message });
  }
});

/**
 * BATCH ANALYTICS — reads from DB, no CSV dependency
 */
app.get('/api/analytics/batch', async (req, res) => {
  const { universe, limit = 5 } = req.query;

  if (!universe) return res.status(400).json({ error: 'Universe is required' });

  console.log(`\n📊 Batch Analytics (DB): universe=${universe} limit=${limit}`);

  try {
    // Resolve universe name to DB format
    const choice = Object.keys(UNIVERSE_MAP).find(k => UNIVERSE_MAP[k] === universe);
    const dbUniverse = choice ? (DB_UNIVERSE_MAP[choice] || universe) : universe;

    // Distinct scan dates for this universe in DB
    const dateRows = await dbAll(
      `SELECT DISTINCT DATE(scan_date) as d FROM daily_signals
       WHERE decision='BUY' AND universe=?
       ORDER BY d DESC LIMIT ?`,
      [dbUniverse, parseInt(limit) || 5]
    );

    const recentDates = dateRows.map(r => r.d).filter(Boolean);
    if (!recentDates.length) {
      return res.status(404).json({ error: 'No scan history in DB for this universe' });
    }

    console.log(`   DB dates: ${recentDates.join(', ')}`);

    // Load all BUY signals for those dates in one query
    const placeholders = recentDates.map(() => '?').join(',');
    const rows = await dbAll(
      `SELECT *, DATE(scan_date) as _date FROM daily_signals
       WHERE decision='BUY' AND universe=? AND DATE(scan_date) IN (${placeholders})
       ORDER BY scan_date DESC`,
      [dbUniverse, ...recentDates]
    );

    console.log(`   ${rows.length} signals from DB, fetching live prices…`);
    const symbols = [...new Set(rows.map(r => r.symbol))];
    const currentPrices = await fetchCurrentPrices(symbols);
    console.log(`   ✓ ${Object.keys(currentPrices).length} prices`);

    const analyzedSignals = rows.map(r => {
      const entryPrice   = r.signal_price ?? r.current_price ?? 0;
      const currentPrice = currentPrices[r.symbol] ?? null;
      const daysHeld     = Math.floor((Date.now() - new Date(r._date)) / 864e5);

      let returnPct = 0;
      let status = 'OPEN';
      if (currentPrice && entryPrice) {
        returnPct = ((currentPrice - entryPrice) / entryPrice) * 100;
        if (r.target_1 && currentPrice >= r.target_1)       status = 'TARGET_HIT';
        else if (r.stop_loss && currentPrice <= r.stop_loss) status = 'STOP_HIT';
        else                                                  status = returnPct > 0 ? 'PROFIT' : 'LOSS';
      }

      return {
        symbol:       r.symbol,
        date:         r._date,
        grade:        r.grade || 'N/A',
        score:        r.score ?? 0,
        returnPct,
        daysHeld,
        status,
        volumeRatio:  r.volume_ratio ?? 0,
        vwapDeviation: r.vwap_deviation_pct ?? 0,
        marketTrend:  r.market_trend || 'Neutral',
        riskPct:      r.risk_pct ?? 0,
        stopLoss:     r.stop_loss ?? 0,
        target1:      r.target_1  ?? 0,
      };
    });

    res.json({ dates: recentDates, totalSignals: analyzedSignals.length, signals: analyzedSignals });

  } catch (err) {
    console.error('❌ Batch analytics error:', err);
    res.status(500).json({ error: err.message });
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

// ── Telegram notification ─────────────────────────────────────────────────────
async function sendTelegram(text) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  if (!token || !chatId) return;
  return new Promise((resolve) => {
    const body = JSON.stringify({ chat_id: chatId, text, parse_mode: 'HTML' });
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${token}/sendMessage`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, (res) => {
      res.on('data', () => {});
      res.on('end', () => { console.log(`📱 Telegram sent (${res.statusCode})`); resolve(); });
    });
    req.on('error', err => { console.error('Telegram error:', err.message); resolve(); });
    req.write(body);
    req.end();
  });
}

// ── Scheduled daily scan (called by cron) ────────────────────────────────────
async function runScheduledScan() {
  const universeChoice = process.env.SCHEDULED_UNIVERSE || '2';
  if (!UNIVERSE_MAP[universeChoice]) {
    console.error(`⏰ Scheduled scan: invalid SCHEDULED_UNIVERSE="${universeChoice}"`);
    return;
  }
  const scanTypeChoice = '2'; // Always Tier-2 daily scan

  const running = [...activeScans.values()].find(s => !s.completed);
  if (running) {
    console.log('⏰ Scheduled scan skipped — another scan is already active');
    return;
  }

  const scanId = `scheduled_${Date.now()}`;
  const cacheKey = getCacheKey(scanTypeChoice, universeChoice);
  const scanState = {
    id: scanId, progress: 'Scheduled scan starting...', logs: '',
    completed: false, results: null, error: null, fromCache: false, startTime: new Date(),
  };
  activeScans.set(scanId, scanState);

  const t0 = Date.now();
  console.log(`\n⏰ [${scanId}] Scheduled daily scan — ${UNIVERSE_MAP[universeChoice]}`);

  await new Promise((resolve) => {
    const proc = spawn(PYTHON_CMD_ENV, [
      path.join(__dirname, 'python-scanner-script.py'), scanTypeChoice, universeChoice,
    ]);

    const killTimer = setTimeout(() => {
      proc.kill('SIGTERM');
      scanState.error = 'Timed out after 15 min';
      scanState.completed = true;
      sendTelegram(`⚠️ <b>Momentum Scanner</b>\nScheduled scan timed out after 15 minutes.`)
        .finally(resolve);
    }, 15 * 60 * 1000);

    proc.stdout.on('data', d => { scanState.logs += d.toString(); });
    proc.stderr.on('data', d => { scanState.logs += `[ERR] ${d}`; });

    proc.on('error', async err => {
      clearTimeout(killTimer);
      console.error(`[${scanId}] spawn error:`, err.message);
      scanState.completed = true;
      await sendTelegram(`❌ <b>Momentum Scanner</b>\nScheduled scan failed to start:\n${err.message}`);
      resolve();
    });

    proc.on('close', async (code) => {
      clearTimeout(killTimer);
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`[${scanId}] Python exited code=${code} (${elapsed}s)`);

      try {
        await cacheResults(cacheKey, scanTypeChoice);
        openDbIfNeeded();
        // Allow DB writes to settle before querying
        await new Promise(r => setTimeout(r, 2000));

        const today = new Date().toISOString().split('T')[0];
        const signals = await dbAll(`
          SELECT symbol, grade, score, entry_low, entry_high, stop_loss, target_1, market_trend
          FROM daily_signals
          WHERE decision = 'BUY' AND DATE(scan_date) = DATE(?)
          ORDER BY score DESC LIMIT 50
        `, [today]);

        const gradeMap = {};
        signals.forEach(s => { gradeMap[s.grade || '?'] = (gradeMap[s.grade || '?'] || 0) + 1; });
        const gradeStr = Object.entries(gradeMap).sort()
          .map(([g, n]) => `${g}:${n}`).join('  ');

        const top5 = signals.slice(0, 5).map((s, i) => {
          const ent = (s.entry_low && s.entry_high)
            ? `₹${Number(s.entry_low).toFixed(0)}–${Number(s.entry_high).toFixed(0)}` : '—';
          const sl = s.stop_loss ? `  SL:₹${Number(s.stop_loss).toFixed(0)}` : '';
          const t1 = s.target_1  ? `  T1:₹${Number(s.target_1).toFixed(0)}`  : '';
          return `  ${i + 1}. ${s.symbol} (${s.grade}, ${s.score})  ${ent}${sl}${t1}`;
        }).join('\n');

        const trend = signals[0]?.market_trend || '—';
        const msg = [
          `📊 <b>Momentum Scanner — Daily Scan</b>`,
          `📅 ${today}   🌐 ${UNIVERSE_MAP[universeChoice]}`,
          ``,
          `🎯 BUY Signals: <b>${signals.length}</b>   ${gradeStr}`,
          `📈 Market: ${trend}`,
          ``,
          signals.length > 0 ? `🔥 Top picks:\n${top5}` : `ℹ️ No BUY signals today.`,
          ``,
          `⏱ Completed in ${elapsed}s`,
        ].join('\n');

        await sendTelegram(msg);
      } catch (err) {
        console.error(`[${scanId}] Post-scan error:`, err.message);
        await sendTelegram(
          `⚠️ <b>Momentum Scanner</b>\nScan ran but result read failed:\n${err.message}`
        );
      } finally {
        scanState.completed = true;
        resolve();
      }
    });
  });
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

// ── Cron: 4:30 PM IST = Mon–Fri ──────────────────────────────────────────────
cron.schedule('30 16 * * 1-5', () => {
  console.log('⏰ Cron fired: starting scheduled daily scan (4:30 PM IST)');
  runScheduledScan().catch(err => console.error('Scheduled scan uncaught error:', err.message));
}, { timezone: 'Asia/Kolkata' });
console.log('⏰ Scheduled scan registered: 4:30 PM IST, Mon–Fri');