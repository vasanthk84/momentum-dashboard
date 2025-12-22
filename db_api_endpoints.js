/**
 * DATABASE API ENDPOINTS
 * Add these to your existing server.js file
 * 
 * These endpoints provide:
 * - Database statistics
 * - Query historical signals
 * - Export to Excel/CSV
 * - Scan history retrieval
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

// ============================================================================
// DATABASE CONNECTION
// ============================================================================

// Path to your scan database (created by db_persistence.py)
const DB_PATH = path.join(__dirname, 'scan_history.db');

/**
 * Get database connection
 */
function getDbConnection() {
  if (!fs.existsSync(DB_PATH)) {
    console.warn('⚠️  Database not found:', DB_PATH);
    return null;
  }
  return new sqlite3.Database(DB_PATH, sqlite3.OPEN_READONLY);
}

// ============================================================================
// API ENDPOINTS - ADD THESE TO YOUR server.js
// ============================================================================

/**
 * GET /api/db/stats
 * Get database statistics
 */
app.get('/api/db/stats', (req, res) => {
  const db = getDbConnection();
  if (!db) {
    return res.status(503).json({ error: 'Database not available' });
  }

  const stats = {};

  db.serialize(() => {
    // Weekly watchlist count
    db.get('SELECT COUNT(*) as count FROM weekly_watchlist', (err, row) => {
      if (!err) stats.weekly_total = row.count;
    });

    // Daily signals count
    db.get('SELECT COUNT(*) as count FROM daily_signals', (err, row) => {
      if (!err) stats.daily_total = row.count;
    });

    // BUY signals count
    db.get("SELECT COUNT(*) as count FROM daily_signals WHERE decision = 'BUY'", (err, row) => {
      if (!err) stats.buy_signals = row.count;
    });

    // Unique scan dates
    db.get('SELECT COUNT(DISTINCT DATE(scan_date)) as count FROM scan_metadata', (err, row) => {
      if (!err) stats.unique_scan_dates = row.count;

      // Send response after all queries complete
      db.close();
      res.json(stats);
    });
  });
});

/**
 * GET /api/db/scan-history?days=30
 * Get scan history for last N days
 */
app.get('/api/db/scan-history', (req, res) => {
  const db = getDbConnection();
  if (!db) {
    return res.status(503).json({ error: 'Database not available' });
  }

  const days = parseInt(req.query.days) || 30;

  const query = `
    SELECT 
      scan_date,
      scan_type,
      universe,
      total_signals,
      buy_signals,
      wait_signals,
      skip_signals,
      avg_score,
      avg_rs_score
    FROM scan_metadata 
    WHERE scan_date >= datetime('now', '-' || ? || ' days')
    ORDER BY scan_date DESC
  `;

  db.all(query, [days], (err, rows) => {
    db.close();
    
    if (err) {
      return res.status(500).json({ error: err.message });
    }
    
    res.json({ scans: rows });
  });
});

/**
 * GET /api/db/signals?date=2025-12-14&universe=Nifty_50
 * Get all signals for a specific date
 */
app.get('/api/db/signals', (req, res) => {
  const db = getDbConnection();
  if (!db) {
    return res.status(503).json({ error: 'Database not available' });
  }

  const { date, universe } = req.query;

  if (!date) {
    db.close();
    return res.status(400).json({ error: 'Date parameter required' });
  }

  let query = `
    SELECT 
      symbol,
      decision,
      grade,
      score,
      current_price,
      stop_loss,
      target_1,
      target_2,
      vwap,
      volume_ratio,
      reasons,
      wyckoff_phase
    FROM daily_signals 
    WHERE DATE(scan_date) = DATE(?)
  `;

  const params = [date];

  if (universe) {
    query += ' AND universe = ?';
    params.push(universe);
  }

  query += ' ORDER BY score DESC';

  db.all(query, params, (err, rows) => {
    if (err) {
      db.close();
      return res.status(500).json({ error: err.message });
    }

    // Count by decision
    const buys = rows.filter(r => r.decision === 'BUY').length;
    const waits = rows.filter(r => r.decision === 'WAIT').length;
    const skips = rows.filter(r => r.decision === 'SKIP').length;

    db.close();

    res.json({
      total: rows.length,
      buy: buys,
      wait: waits,
      skip: skips,
      signals: rows.map(row => ({
        ...row,
        reasons: row.reasons ? JSON.parse(row.reasons) : []
      }))
    });
  });
});

/**
 * GET /api/db/export?table=daily_signals
 * Export a table to CSV
 */
app.get('/api/db/export', (req, res) => {
  const db = getDbConnection();
  if (!db) {
    return res.status(503).json({ error: 'Database not available' });
  }

  const table = req.query.table;
  const validTables = ['weekly_watchlist', 'daily_signals', 'performance_tracking', 'scan_metadata'];

  if (!validTables.includes(table)) {
    db.close();
    return res.status(400).json({ error: 'Invalid table name' });
  }

  db.all(`SELECT * FROM ${table}`, [], (err, rows) => {
    db.close();

    if (err) {
      return res.status(500).json({ error: err.message });
    }

    if (rows.length === 0) {
      return res.status(404).json({ error: 'No data found' });
    }

    // Convert to CSV
    const headers = Object.keys(rows[0]);
    const csvRows = [headers.join(',')];

    for (const row of rows) {
      const values = headers.map(header => {
        const value = row[header];
        // Escape quotes and wrap in quotes if contains comma
        if (value === null || value === undefined) return '';
        const escaped = String(value).replace(/"/g, '""');
        return escaped.includes(',') ? `"${escaped}"` : escaped;
      });
      csvRows.push(values.join(','));
    }

    const csv = csvRows.join('\n');

    // Send as downloadable file
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="${table}_export.csv"`);
    res.send(csv);
  });
});

/**
 * GET /api/db/export-all
 * Export all tables to CSV files
 */
app.get('/api/db/export-all', (req, res) => {
  const db = getDbConnection();
  if (!db) {
    return res.status(503).json({ error: 'Database not available' });
  }

  const tables = ['weekly_watchlist', 'daily_signals', 'performance_tracking', 'scan_metadata'];
  const timestamp = new Date().toISOString().split('T')[0];
  const exportedFiles = [];

  let completed = 0;

  tables.forEach(table => {
    db.all(`SELECT * FROM ${table}`, [], (err, rows) => {
      if (err) {
        console.error(`Error exporting ${table}:`, err);
        completed++;
        return;
      }

      if (rows.length > 0) {
        const headers = Object.keys(rows[0]);
        const csvRows = [headers.join(',')];

        for (const row of rows) {
          const values = headers.map(header => {
            const value = row[header];
            if (value === null || value === undefined) return '';
            const escaped = String(value).replace(/"/g, '""');
            return escaped.includes(',') ? `"${escaped}"` : escaped;
          });
          csvRows.push(values.join(','));
        }

        const csv = csvRows.join('\n');
        const filename = `${table}_export_${timestamp}.csv`;
        const filepath = path.join(__dirname, filename);

        fs.writeFileSync(filepath, csv);
        exportedFiles.push(filename);
      }

      completed++;

      if (completed === tables.length) {
        db.close();
        res.json({ 
          success: true, 
          files: exportedFiles,
          message: `Exported ${exportedFiles.length} files`
        });
      }
    });
  });
});

/**
 * GET /api/db/performance?symbol=RELIANCE.NS&days=30
 * Get performance data for a symbol
 */
app.get('/api/db/performance', (req, res) => {
  const db = getDbConnection();
  if (!db) {
    return res.status(503).json({ error: 'Database not available' });
  }

  const { symbol, days = 30 } = req.query;

  let query = `
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
  `;

  const params = [days];

  if (symbol) {
    query += ' AND p.symbol = ?';
    params.push(symbol);
  }

  query += ' ORDER BY p.entry_date DESC';

  db.all(query, params, (err, rows) => {
    db.close();

    if (err) {
      return res.status(500).json({ error: err.message });
    }

    // Calculate statistics
    const returns = rows.map(r => r.return_pct).filter(r => r !== null);
    const avgReturn = returns.length > 0 ? 
      returns.reduce((a, b) => a + b, 0) / returns.length : 0;
    const winners = returns.filter(r => r > 0).length;
    const winRate = returns.length > 0 ? (winners / returns.length) * 100 : 0;

    res.json({
      signals: rows,
      statistics: {
        total: rows.length,
        avgReturn: avgReturn.toFixed(2),
        winners,
        winRate: winRate.toFixed(2)
      }
    });
  });
});

// ============================================================================
// HELPER: Install sqlite3 if not already installed
// ============================================================================

/**
 * Add this to your package.json dependencies:
 * 
 * "dependencies": {
 *   "express": "^4.18.2",
 *   "body-parser": "^1.20.2",
 *   "sqlite3": "^5.1.6",    <-- ADD THIS
 *   ...
 * }
 * 
 * Then run: npm install sqlite3
 */

// ============================================================================
// INTEGRATION CHECKLIST
// ============================================================================

/**
 * ✅ SETUP STEPS:
 * 
 * 1. Install sqlite3:
 *    npm install sqlite3
 * 
 * 2. Copy these endpoints to your server.js
 * 
 * 3. Save the dashboard HTML as 'history.html' in your project
 * 
 * 4. Add route to serve the dashboard:
 *    app.get('/history', (req, res) => {
 *      res.sendFile(path.join(__dirname, 'history.html'));
 *    });
 * 
 * 5. Make sure scan_history.db exists (run scanner with DB integration first)
 * 
 * 6. Access dashboard at: http://localhost:3000/history
 */

// ============================================================================
// EXAMPLE: Full server.js integration
// ============================================================================

/**
 * Here's how your server.js should look with DB endpoints:
 */

const express = require('express');
const bodyParser = require('body-parser');
const sqlite3 = require('sqlite3').verbose();  // ADD THIS
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname)));

// ============================================================================
// EXISTING SCANNER ENDPOINTS (keep these)
// ============================================================================
app.post('/api/start-scan', (req, res) => { /* your existing code */ });
app.get('/api/scan-status/:scanId', (req, res) => { /* your existing code */ });
app.get('/api/scan-results/:scanId', (req, res) => { /* your existing code */ });
app.get('/api/performance/dates', (req, res) => { /* your existing code */ });
app.get('/api/performance/analyze', (req, res) => { /* your existing code */ });

// ============================================================================
// NEW DATABASE ENDPOINTS (add all the endpoints from above)
// ============================================================================
app.get('/api/db/stats', (req, res) => { /* from above */ });
app.get('/api/db/scan-history', (req, res) => { /* from above */ });
app.get('/api/db/signals', (req, res) => { /* from above */ });
app.get('/api/db/export', (req, res) => { /* from above */ });
app.get('/api/db/export-all', (req, res) => { /* from above */ });
app.get('/api/db/performance', (req, res) => { /* from above */ });

// ============================================================================
// PAGE ROUTES
// ============================================================================
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/performance', (req, res) => {
  res.sendFile(path.join(__dirname, 'performance.html'));
});

app.get('/history', (req, res) => {  // NEW ROUTE
  res.sendFile(path.join(__dirname, 'history.html'));
});

app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
  console.log(`📊 Scanner: http://localhost:${PORT}/`);
  console.log(`📈 Performance: http://localhost:${PORT}/performance`);
  console.log(`💾 History DB: http://localhost:${PORT}/history`);  // NEW
});

module.exports = app;