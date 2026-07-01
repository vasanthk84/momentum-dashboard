// ── Scan config ──────────────────────────────────────────────────────────────
export type MainChoice = '1' | '2' | '3'
export type UniverseChoice = '1' | '2' | '3' | '4' | '5' | '6' | '7'

export const SCAN_TYPE_LABELS: Record<MainChoice, string> = {
  '1': 'Weekly Scan (Tier 1)',
  '2': 'Daily Scan (Tier 2)',
  '3': 'Combined (Both Tiers)',
}

export const UNIVERSE_LABELS: Record<UniverseChoice, string> = {
  '1': 'Nifty 50',
  '2': 'Nifty Next 50',
  '3': 'Large Cap',
  '4': 'Mid Cap',
  '5': 'Small Cap',
  '6': 'ALL Files',
  '7': 'Quality Universe (558)',
}

export const UNIVERSE_KEYS: Record<UniverseChoice, string> = {
  '1': 'Nifty_50',
  '2': 'Nifty_Next_50',
  '3': 'Large_Cap',
  '4': 'Mid_Cap',
  '5': 'Small_Cap',
  '6': 'ALL_Files',
  '7': 'Quality_Universe',
}

// ── Scanner signals ───────────────────────────────────────────────────────────
export interface BuySignal {
  Symbol: string
  Grade: 'A' | 'B' | 'C' | string
  Score: number
  Current_Price: number
  Entry_Low: number
  Entry_High: number
  Stop_Loss: number
  Target_1: number
  'Risk_%': number
  Volume_Ratio: number
  Market_Trend: string
  metrics?: string[]
  [key: string]: unknown
}

export interface WaitSignal {
  Symbol: string
  Grade: 'A' | 'B' | 'C' | string
  Score: number
  Current_Price: number
  Stop_Loss: number
  Target_1: number
  'Risk_%': number
  Volume_Ratio: number
  Reason: string
  Market_Trend: string
  metrics?: string[]
  [key: string]: unknown
}

export interface WatchlistItem {
  Symbol: string
  Stage: 'BREAKOUT_READY' | 'ACCUMULATION' | string
  Total_Score: number
  Current_Price: number
  Stop_Price: number
  Risk_Reward: number
  RS_Score: number
  Base_Quality: number
  Signals?: string
  metrics?: string[]
  [key: string]: unknown
}

export interface ScanResults {
  buy: BuySignal[]
  wait: WaitSignal[]
  watchlist: WatchlistItem[]
}

// ── API responses ─────────────────────────────────────────────────────────────
export interface StartScanResponse {
  scanId: string
}

export interface ScanStatusResponse {
  completed: boolean
  error: string | null
  progress: string
  fromCache: boolean
}

export interface ScanResultsResponse {
  logs: string
  results: ScanResults
  error: string | null
  fromCache: boolean
}

// ── Performance ───────────────────────────────────────────────────────────────
export type SignalStatus = 'TARGET_HIT' | 'STOP_HIT' | 'OPEN' | 'PROFIT' | 'LOSS' | 'NO_DATA'

export interface PerfSignal {
  symbol: string
  grade: string
  score: number
  signalPrice: number
  currentPrice: number | null
  stopLoss: number
  target1: number
  riskPct: number
  returnPct: number | null
  status: SignalStatus
}

export interface PerfAnalysisResponse {
  scanDate: string
  daysHeld: number
  totalSignals: number
  avgReturn: number
  winners: number
  signals: PerfSignal[]
}

export interface PerfDatesResponse {
  dates: string[]
}

// ── Analytics ─────────────────────────────────────────────────────────────────
export interface BatchSignal {
  symbol: string
  date: string
  grade: string
  score: number
  returnPct: number
  daysHeld: number
  status: SignalStatus
  volumeRatio: number
  vwapDeviation: number
  marketTrend: string
  riskPct: number
  stopLoss: number
  target1: number
}

export interface BatchAnalyticsResponse {
  dates: string[]
  totalSignals: number
  signals: BatchSignal[]
}

// ── DB / SQLite types ─────────────────────────────────────────────────────────
export interface DbSignal {
  scan_date: string
  symbol: string
  decision: 'BUY' | 'WAIT' | 'SKIP'
  grade: string
  score: number
  wyckoff_phase: string | null
  signal_price: number
  current_price: number | null
  price_change_pct: number | null
  vwap: number | null
  vwap_deviation_pct: number | null
  poc: number | null
  value_area_low: number | null
  value_area_high: number | null
  in_value_area: 0 | 1 | null
  volume_ratio: number
  above_ma20: 0 | 1 | null
  above_ma50: 0 | 1 | null
  ma_aligned: 0 | 1 | null
  entry_low: number | null
  entry_high: number | null
  stop_loss: number | null
  risk_pct: number | null
  target_1: number | null
  target_2: number | null
  reasons: string[]
  wait_factors: string[]
  market_trend: string | null
  market_strength: number | null
  universe: string | null
}

export interface DbStats {
  dailySignals: number
  weeklyEntries: number
  totalScans: number
  lastScanDate: string | null
  dbSizeMb: number
}

export interface DbWatchlistItem {
  date: string
  symbol: string
  stage: string
  total_score: number
  accumulation_score: number | null
  breakout_score: number | null
  current_price: number
  stop_price: number
  risk_reward: number | null
  rs_score: number | null
  base_quality: number | null
  distance_to_resistance: number | null
  universe: string | null
}

export interface DbScanHistory {
  date: string
  scan_type: string
  universe: string
  total_signals: number
  buy_signals: number
  wait_signals: number
  skip_signals: number
  scan_duration_seconds: number | null
}

// ── Theme ─────────────────────────────────────────────────────────────────────
export type Mood = 'bloomberg' | 'carbon' | 'paper'

// ── UI tabs ───────────────────────────────────────────────────────────────────
export type MainTab = 'scanner' | 'watchlist' | 'performance' | 'analytics'
export type ResultTab = 'buy' | 'wait' | 'watchlist'
