// server.js - Stock Scanner Backend Server with Metrics Support
const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const csv = require('csv-parser');
const createReadStream = require('fs').createReadStream;
const cors = require('cors');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname))); 

const activeScans = new Map();

/**
 * Parse CSV file and return JSON
 */
function parseCSV(filePath) {
  return new Promise((resolve, reject) => {
    const results = [];
    console.log(`Attempting to read: ${filePath}`);
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
 * Find the most recent CSV files
 */
async function findRecentCSVFiles(pattern) {
  try {
    const files = await fs.readdir('.');
    const matchedFiles = files
      .filter(f => f.startsWith(pattern) && f.endsWith('.csv'))
      .sort()
      .reverse();
    
    return matchedFiles;
  } catch (err) {
    console.error(`Error finding CSVs: ${err.message}`);
    return [];
  }
}

/**
 * Clean up old CSV files
 */
async function cleanupOldFiles() {
    console.log("🧹 Cleaning up old scan files...");
    const patterns = [
      'BUY_SIGNALS_',
      'WAIT_LIST_',
      'watchlist_momentum_current.csv',
      'entry_decisions_', 
      'watchlist_entry_timing_', 
      'BUY_SIGNALS_SIMPLE_'
    ];
    let cleanedCount = 0;
    
    for (const pattern of patterns) {
        try {
            const files = await findRecentCSVFiles(pattern.split('_')[0]);
            for (const file of files) {
                await fs.unlink(file);
                cleanedCount++;
            }
        } catch (err) {
            // Ignore errors
        }
    }
    console.log(`🧹 Cleaned up ${cleanedCount} files.`);
}

/**
 * Parse metrics string from CSV (removes brackets and quotes, splits by comma)
 */
function parseMetrics(metricsString) {
  if (!metricsString) return [];
  
  let cleaned = metricsString.toString().trim();
  
  // Log the original string for debugging
  console.log('Parsing metrics string:', cleaned.substring(0, 100) + '...');
  
  // Check if it's a Python list format like "['item1', 'item2', 'item3']"
  if (cleaned.startsWith('[') && cleaned.endsWith(']')) {
    // Remove outer brackets
    cleaned = cleaned.slice(1, -1);
    
    // Split by ', ' but be careful with commas inside quotes
    const metrics = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < cleaned.length; i++) {
      const char = cleaned[i];
      const nextChar = cleaned[i + 1];
      
      if ((char === '"' || char === "'") && (i === 0 || cleaned[i - 1] !== '\\')) {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes && nextChar === ' ') {
        // Found a separator between items
        const item = current.trim().replace(/^["']|["']$/g, '');
        if (item) metrics.push(item);
        current = '';
        i++; // Skip the space after comma
      } else {
        current += char;
      }
    }
    
    // Add the last item
    const lastItem = current.trim().replace(/^["']|["']$/g, '');
    if (lastItem) metrics.push(lastItem);
    
    console.log(`Parsed ${metrics.length} metrics from Python list format`);
    return metrics;
  }
  
  // Fallback for simple comma-separated strings
  return cleaned
    .split(',')
    .map(m => m.trim().replace(/^["']|["']$/g, ''))
    .filter(m => m.length > 0);
}

/**
 * Merge metrics from watchlist into results
 */
function mergeMetrics(results, watchlistData) {
  // Create a map of symbol -> metrics from watchlist
  const metricsMap = {};
  
  console.log('Processing watchlist data for metrics...');
  
  if (watchlistData && watchlistData.length > 0) {
    console.log('Watchlist columns:', Object.keys(watchlistData[0]));
    
    watchlistData.forEach((row, index) => {
      const symbol = row.Symbol || row.symbol || row.SYMBOL || row.Ticker || row.ticker;
      
      // Get Signals column from watchlist
      const signals = row.Signals || row.signals || row.SIGNALS;
      
      if (symbol && signals) {
        const cleanSymbol = symbol.trim().toUpperCase();
        
        if (index < 2) { // Log first 2 in detail for debugging
          console.log(`\n--- Processing ${cleanSymbol} ---`);
          console.log('Raw signals:', signals.substring(0, 200));
        }
        
        const parsedMetrics = parseMetrics(signals);
        metricsMap[cleanSymbol] = parsedMetrics;
        
        if (index < 2) {
          console.log(`Parsed into ${parsedMetrics.length} metrics:`);
          parsedMetrics.forEach((m, i) => console.log(`  ${i + 1}. ${m}`));
        }
      }
    });
  }
  
  console.log(`Total symbols with metrics in watchlist: ${Object.keys(metricsMap).length}`);
  
  // Merge metrics into buy signals
  if (results.buy) {
    results.buy = results.buy.map(stock => {
      const symbol = (stock.Symbol || stock.SYMBOL || stock.symbol || '').trim().toUpperCase();
      const metrics = metricsMap[symbol] || [];
      
      if (metrics.length > 0) {
        console.log(`BUY: Attached ${metrics.length} metrics to ${symbol}`);
      } else {
        console.log(`BUY: No watchlist metrics found for ${symbol}`);
      }
      
      return {
        ...stock,
        metrics: metrics
      };
    });
  }
  
  // Merge metrics into wait signals
  if (results.wait) {
    results.wait = results.wait.map(stock => {
      const symbol = (stock.Symbol || stock.SYMBOL || stock.symbol || '').trim().toUpperCase();
      const metrics = metricsMap[symbol] || [];
      
      if (metrics.length > 0) {
        console.log(`WAIT: Attached ${metrics.length} metrics to ${symbol}`);
      } else {
        console.log(`WAIT: No watchlist metrics found for ${symbol}`);
      }
      
      return {
        ...stock,
        metrics: metrics
      };
    });
  }
  
  // Watchlist already has metrics added in the parsing step
  
  return results;
}

/**
 * Start a new scan
 */
app.post('/api/start-scan', async (req, res) => {
  const { mainChoice, universeChoice } = req.body;
  const scanId = `scan_${Date.now()}`;
  
  console.log(`[${scanId}] Scan request: Main=${mainChoice}, Universe=${universeChoice}`);

  await cleanupOldFiles();

  const scanState = {
    id: scanId,
    progress: 'Starting...',
    logs: '',
    completed: false,
    results: null,
    error: null,
    startTime: new Date()
  };
  activeScans.set(scanId, scanState);

  res.json({ scanId });

  const pythonProcess = spawn('python', [
    'python-scanner-script.py',
    mainChoice,
    universeChoice
  ]);

  pythonProcess.stdout.on('data', (data) => {
    const logChunk = data.toString();
    scanState.logs += logChunk;
    scanState.progress = logChunk.split('\n').filter(Boolean).pop() || scanState.progress;
    console.log(`[${scanId} STDOUT]: ${logChunk}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    const errorChunk = data.toString();
    scanState.logs += `[STDERR] ${errorChunk}\n`;
    
    // Check if this is actually a warning vs a fatal error
    const isWarning = errorChunk.includes('possibly delisted') || 
                     errorChunk.includes('no timezone found') ||
                     errorChunk.includes('Warning:') ||
                     errorChunk.includes('FutureWarning') ||
                     errorChunk.includes('DeprecationWarning');
    
    // Only treat as error if it's not a warning
    if (!isWarning) {
      scanState.error = (scanState.error || '') + errorChunk;
    }
    
    console.error(`[${scanId} STDERR]: ${errorChunk}`);
  });

  pythonProcess.on('close', async (code) => {
    console.log(`[${scanId}] Python script exited with code ${code}`);
    
    // Only fail if exit code is non-zero AND there are actual errors (not just warnings)
    if (code !== 0 && scanState.error && scanState.error.trim().length > 0) {
      scanState.error = scanState.error || 'Python script exited with non-zero code.';
      scanState.completed = true;
      return;
    }
    
    // Clear any warning-only errors if script completed successfully
    if (code === 0) {
      scanState.error = null;
    }

    try {
      scanState.progress = 'Parsing CSV results...';
      const results = {
        buy: [],
        wait: [],
        watchlist: []
      };

      const buyFiles = await findRecentCSVFiles('BUY_SIGNALS_'); 
      const waitFiles = await findRecentCSVFiles('WAIT_LIST_');
      const watchlistFile = 'watchlist_momentum_current.csv';

      const parsePromises = [];

      if (buyFiles.length > 0) {
        console.log(`[${scanId}] Found BUY file: ${buyFiles[0]}`);
        parsePromises.push(parseCSV(buyFiles[0]).then(data => {
          // Don't add metrics yet - we'll merge them from watchlist after
          results.buy = data.map(stock => ({
            ...stock,
            metrics: [] // Will be filled from watchlist
          }));
        }));
      }
      
      if (waitFiles.length > 0) {
        console.log(`[${scanId}] Found WAIT file: ${waitFiles[0]}`);
        parsePromises.push(parseCSV(waitFiles[0]).then(data => {
          // Don't add metrics yet - we'll merge them from watchlist after
          results.wait = data.map(stock => ({
            ...stock,
            metrics: [] // Will be filled from watchlist
          }));
        }));
      }

      let watchlistData = [];
      try {
        await fs.access(watchlistFile);
        console.log(`[${scanId}] Found Weekly file: ${watchlistFile}`);
        watchlistData = await parseCSV(watchlistFile);
        
        // Process watchlist data and extract metrics from Signals column
        results.watchlist = watchlistData.map(stock => {
          const metrics = [];
          
          // Collect Signals
          if (stock.Signals && stock.Signals.trim()) {
            const signals = parseMetrics(stock.Signals);
            metrics.push(...signals);
          }
          
          // Also check for other reason columns
          const otherReasons = stock.Selection_Reasons || stock.Reasons || stock.Signal_Reasons || stock.Metrics;
          if (otherReasons) {
            metrics.push(...parseMetrics(otherReasons));
          }
          
          return {
            ...stock,
            metrics: metrics
          };
        });
      } catch (e) {
          console.log(`[${scanId}] No weekly file found. (This is normal for Daily-only scans)`);
      }

      await Promise.all(parsePromises);

      // Now merge metrics from watchlist into BUY and WAIT signals
      scanState.results = mergeMetrics(results, watchlistData);

      // Log how many metrics were found
      if (scanState.results.buy && scanState.results.buy.length > 0) {
        const metricsCount = scanState.results.buy.filter(s => s.metrics && s.metrics.length > 0).length;
        console.log(`[${scanId}] BUY: ${metricsCount}/${scanState.results.buy.length} stocks have metrics`);
        if (scanState.results.buy[0].metrics) {
          console.log(`[${scanId}] First BUY stock (${scanState.results.buy[0].Symbol}) has ${scanState.results.buy[0].metrics.length} metrics`);
        }
      }
      
      if (scanState.results.wait && scanState.results.wait.length > 0) {
        const metricsCount = scanState.results.wait.filter(s => s.metrics && s.metrics.length > 0).length;
        console.log(`[${scanId}] WAIT: ${metricsCount}/${scanState.results.wait.length} stocks have metrics`);
      }
      
      if (scanState.results.watchlist && scanState.results.watchlist.length > 0) {
        const metricsCount = scanState.results.watchlist.filter(s => s.metrics && s.metrics.length > 0).length;
        console.log(`[${scanId}] WATCHLIST: ${metricsCount}/${scanState.results.watchlist.length} stocks have metrics`);
      }

      scanState.progress = 'Scan complete.';
      console.log(`[${scanId}] Scan complete. Results: ${scanState.results.buy.length} BUY, ${scanState.results.wait.length} WAIT, ${scanState.results.watchlist.length} WATCHLIST`);

    } catch (err) {
      console.error(`[${scanId}] Failed to parse CSVs: ${err.message}`);
      scanState.error = 'Failed to parse CSV output files.';
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
        return res.status(202).json({ message: 'Scan still in progress.'});
    }

    res.json({
        logs: scan.logs,
        results: scan.results,
        error: scan.error
    });
    
    activeScans.delete(scanId);
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Momentum Scanner UI running at http://localhost:${PORT}`);
});