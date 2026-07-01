import type { BatchAnalyticsResponse } from '../types'

export async function getBatchAnalytics(
  universe: string,
  limit: number
): Promise<BatchAnalyticsResponse> {
  const res = await fetch(`/api/analytics/batch?universe=${universe}&limit=${limit}`)
  if (!res.ok) throw new Error('Failed to load analytics')
  return res.json()
}
