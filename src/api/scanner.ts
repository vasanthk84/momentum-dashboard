import type {
  MainChoice,
  UniverseChoice,
  StartScanResponse,
  ScanStatusResponse,
  ScanResultsResponse,
} from '../types'

export async function startScan(
  mainChoice: MainChoice,
  universeChoice: UniverseChoice
): Promise<StartScanResponse> {
  const res = await fetch('/api/start-scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mainChoice, universeChoice }),
  })
  if (!res.ok) throw new Error('Failed to start scan')
  return res.json()
}

export async function getScanStatus(scanId: string): Promise<ScanStatusResponse> {
  const res = await fetch(`/api/scan-status/${scanId}`)
  if (!res.ok) throw new Error('Failed to fetch scan status')
  return res.json()
}

export async function getScanResults(scanId: string): Promise<ScanResultsResponse> {
  const res = await fetch(`/api/scan-results/${scanId}`)
  if (!res.ok) throw new Error('Failed to fetch scan results')
  return res.json()
}
