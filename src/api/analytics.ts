import type { BatchAnalyticsResponse } from '../types'

export async function getBatchAnalytics(
  universe: string,
  limit: number
): Promise<BatchAnalyticsResponse> {
  const res = await fetch(`/api/analytics/batch?universe=${universe}&limit=${limit}`)
  if (!res.ok) throw new Error('Failed to load analytics')
  return res.json()
}

export async function clearCache(
  universe?: string,
  date?: string
): Promise<{ deleted: number; message: string }> {
  const params = new URLSearchParams()
  if (universe) params.set('universe', universe)
  if (date) params.set('date', date)
  const qs = params.toString()
  const res = await fetch(`/api/cache/clear${qs ? `?${qs}` : ''}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to clear cache')
  return res.json()
}
