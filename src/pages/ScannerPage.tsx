import { useState } from 'react'
import ScanControls from '../components/scanner/ScanControls'
import ProgressBar  from '../components/scanner/ProgressBar'
import ScanKpis     from '../components/scanner/ScanKpis'
import ResultTabs   from '../components/scanner/ResultTabs'
import SignalTable  from '../components/scanner/SignalTable'
import EmptyState   from '../components/ui/EmptyState'
import type { MainChoice, ResultTab, ScanResults, UniverseChoice } from '../types'

interface Props {
  scanning: boolean
  progress: string
  fromCache: boolean
  results: ScanResults | null
  logs: string
  error: string | null
  logsVisible: boolean
  mainChoice: MainChoice
  universeChoice: UniverseChoice
  onMainChange: (v: MainChoice) => void
  onUniverseChange: (v: UniverseChoice) => void
}

export default function ScannerPage({
  scanning, progress, fromCache, results, logs, error, logsVisible,
  mainChoice, universeChoice, onMainChange, onUniverseChange,
}: Props) {
  const [resultTab, setResultTab] = useState<ResultTab>('buy')

  const currentData = results
    ? (resultTab === 'buy' ? results.buy : resultTab === 'wait' ? results.wait : results.watchlist) ?? []
    : []

  return (
    <div>
      <ScanControls
        mainChoice={mainChoice}
        universeChoice={universeChoice}
        onMainChange={onMainChange}
        onUniverseChange={onUniverseChange}
      />

      {scanning && <ProgressBar text={progress} fromCache={fromCache} />}

      {error && <div className="error-bar">⚠ {error}</div>}

      {results && <ScanKpis results={results} fromCache={fromCache} />}

      <div className="card card-flush">
        {results && (
          <ResultTabs active={resultTab} results={results} onChange={tab => setResultTab(tab)} />
        )}

        {results ? (
          <SignalTable data={currentData as Record<string, unknown>[]} tab={resultTab} />
        ) : !scanning && !error ? (
          <EmptyState
            icon={
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
            }
            title="No scan results yet"
            message={<>Configure scan type and universe above, then click <b>Run Scan</b> in the toolbar.</>}
          />
        ) : null}
      </div>

      {logsVisible && logs && (
        <div style={{ marginTop: 'var(--gap)' }}>
          <div className="logs-header">
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted)', letterSpacing: '.04em', textTransform: 'uppercase' }}>Python Logs</span>
          </div>
          <pre className="logs-pre">{logs}</pre>
        </div>
      )}
    </div>
  )
}
