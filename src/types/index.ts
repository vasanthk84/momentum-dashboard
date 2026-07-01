// ── Scan config ──────────────────────────────────────────────────────────────
export type MainChoice = '1' | '2' | '3'
export type UniverseChoice = '1' | '2' | '3' | '4' | '5' | '6'

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
}

export const UNIVERSE_KEYS: Record<UniverseChoice, string> = {
  '1': 'Nifty_50',
  '2': 'Nifty_Next_50',
  '3': 'Large_Cap',
  '4': 'Mid_Cap',
  '5': 'Small_Cap',
  '6': 'ALL_Files',
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
}

export interface BatchAnalyticsResponse {
  dates: string[]
  totalSignals: number
  signals: BatchSignal[]
}

// ── Theme ─────────────────────────────────────────────────────────────────────
export type Mood = 'bloomberg' | 'carbon' | 'paper'

// ── UI tabs ───────────────────────────────────────────────────────────────────
export type MainTab = 'scanner' | 'watchlist' | 'performance' | 'analytics'
export type ResultTab = 'buy' | 'wait' | 'watchlist'
