import type { PerfDatesResponse, PerfAnalysisResponse } from '../types'

export async function getPerfDates(universe: string): Promise<PerfDatesResponse> {
  const res = await fetch(`/api/performance/dates?universe=${universe}`)
  if (!res.ok) throw new Error('Failed to load dates')
  return res.json()
}

export async function analyzePerfSignals(
  date: string,
  universe: string
): Promise<PerfAnalysisResponse> {
  const res = await fetch(`/api/performance/analyze?date=${date}&universe=${universe}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Server error' }))
    throw new Error(err.error || 'Failed to analyze')
  }
  return res.json()
}
