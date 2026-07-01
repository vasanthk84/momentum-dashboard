import { useCallback, useRef, useState } from 'react'
import { startScan, getScanStatus, getScanResults } from '../api/scanner'
import type { MainChoice, ScanResults, UniverseChoice } from '../types'

export type ScanState = 'idle' | 'scanning' | 'done' | 'error'

export function useScan() {
  const [scanState, setScanState] = useState<ScanState>('idle')
  const [progress, setProgress]   = useState('')
  const [fromCache, setFromCache] = useState(false)
  const [results, setResults]     = useState<ScanResults | null>(null)
  const [logs, setLogs]           = useState('')
  const [error, setError]         = useState<string | null>(null)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPoll = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const run = useCallback(async (main: MainChoice, universe: UniverseChoice) => {
    stopPoll()
    setScanState('scanning')
    setProgress('Starting…')
    setError(null)
    setResults(null)
    setLogs('')
    setFromCache(false)

    try {
      const { scanId } = await startScan(main, universe)

      pollRef.current = setInterval(async () => {
        try {
          const status = await getScanStatus(scanId)

          if (status.progress) setProgress(status.progress)
          if (status.fromCache) setFromCache(true)

          if (status.error) {
            throw new Error(status.error)
          }

          if (status.completed) {
            stopPoll()
            const data = await getScanResults(scanId)
            if (data.error) throw new Error(data.error)
            if (data.logs)  setLogs(data.logs)
            setResults(data.results)
            setFromCache(status.fromCache)
            setScanState('done')
          }
        } catch (err) {
          stopPoll()
          setError(err instanceof Error ? err.message : 'Unknown error')
          setScanState('error')
        }
      }, 2500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setScanState('error')
    }
  }, [])

  return { scanState, progress, fromCache, results, logs, error, run }
}
