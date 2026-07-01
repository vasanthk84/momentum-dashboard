import { useState } from 'react'
import Chrome       from './components/layout/Chrome'
import TabNav       from './components/layout/TabNav'
import ScannerPage  from './pages/ScannerPage'
import WatchlistPage from './pages/WatchlistPage'
import PerformancePage from './pages/PerformancePage'
import AnalyticsPage   from './pages/AnalyticsPage'
import { useTheme }  from './hooks/useTheme'
import { useScan }   from './hooks/useScan'
import type { MainChoice, MainTab, UniverseChoice, WatchlistItem } from './types'

export default function App() {
  const { mood, setMood }   = useTheme()
  const [activeTab, setTab] = useState<MainTab>('scanner')
  const [logsVisible, setLogsVisible] = useState(false)
  const [mainChoice,    setMainChoice]    = useState<MainChoice>('2')
  const [universeChoice, setUniverseChoice] = useState<UniverseChoice>('2')

  const { scanState, progress, fromCache, results, logs, error, run } = useScan()
  const scanning = scanState === 'scanning'

  const handleRunScan = () => run(mainChoice, universeChoice)

  const watchlistItems: WatchlistItem[] = results?.watchlist ?? []

  return (
    <>
      <Chrome
        mood={mood}
        onMoodChange={setMood}
        scanning={scanning}
        logsVisible={logsVisible}
        onToggleLogs={() => setLogsVisible(v => !v)}
        onRunScan={handleRunScan}
      />

      <TabNav active={activeTab} onChange={setTab} />

      <main className="page">
        {activeTab === 'scanner' && (
          <ScannerPage
            scanning={scanning}
            progress={progress}
            fromCache={fromCache}
            results={results}
            logs={logs}
            error={error}
            logsVisible={logsVisible}
            mainChoice={mainChoice}
            universeChoice={universeChoice}
            onMainChange={setMainChoice}
            onUniverseChange={setUniverseChoice}
          />
        )}

        {activeTab === 'watchlist' && (
          <WatchlistPage items={watchlistItems} />
        )}

        {activeTab === 'performance' && <PerformancePage />}
        {activeTab === 'analytics'   && <AnalyticsPage />}
      </main>
    </>
  )
}
