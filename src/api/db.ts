import type { DbSignal, DbStats, DbWatchlistItem, DbScanHistory } from '../types'

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error((err as { error: string }).error || res.statusText)
  }
  return res.json() as Promise<T>
}

export async function fetchDbStats(): Promise<DbStats> {
  return apiFetch<DbStats>('/api/db/stats')
}

export interface FetchSignalsOpts {
  universe?: string
  date?: string
  decision?: 'BUY' | 'WAIT' | 'SKIP'
  limit?: number
}
export async function fetchDbSignals(opts: FetchSignalsOpts = {}): Promise<{ signals: DbSignal[]; total: number }> {
  const p = new URLSearchParams()
  if (opts.universe) p.set('universe', opts.universe)
  if (opts.date)     p.set('date', opts.date)
  if (opts.decision) p.set('decision', opts.decision)
  if (opts.limit)    p.set('limit', String(opts.limit))
  return apiFetch(`/api/db/signals${p.toString() ? `?${p}` : ''}`)
}

export async function fetchDbDates(universe?: string, decision = 'BUY'): Promise<{ dates: string[] }> {
  const p = new URLSearchParams({ decision })
  if (universe) p.set('universe', universe)
  return apiFetch(`/api/db/dates?${p}`)
}

export async function fetchDbScanHistory(days = 30): Promise<{ history: DbScanHistory[] }> {
  return apiFetch(`/api/db/scan-history?days=${days}`)
}

export async function fetchDbWatchlist(universe?: string, limit = 100): Promise<{ watchlist: DbWatchlistItem[] }> {
  const p = new URLSearchParams({ limit: String(limit) })
  if (universe) p.set('universe', universe)
  return apiFetch(`/api/db/watchlist?${p}`)
}

export async function fetchHealth(): Promise<{
  status: string
  uptime: number
  activeScans: number
  dbConnected: boolean
  pythonCmd: string
  timestamp: string
}> {
  return apiFetch('/api/health')
}
