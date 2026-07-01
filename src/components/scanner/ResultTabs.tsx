import type { ResultTab, ScanResults } from '../../types'

interface Props {
  active: ResultTab
  results: ScanResults
  onChange: (t: ResultTab) => void
}

const TABS: { key: ResultTab; label: string }[] = [
  { key: 'buy',       label: 'BUY'       },
  { key: 'wait',      label: 'WAIT'      },
  { key: 'watchlist', label: 'WATCHLIST' },
]

export default function ResultTabs({ active, results, onChange }: Props) {
  const counts: Record<ResultTab, number> = {
    buy:       results.buy?.length       ?? 0,
    wait:      results.wait?.length      ?? 0,
    watchlist: results.watchlist?.length ?? 0,
  }

  return (
    <div className="result-tabs">
      {TABS.map(t => (
        <button
          key={t.key}
          className={`result-tab ${active === t.key ? 'active' : ''}`}
          onClick={() => onChange(t.key)}
        >
          {t.label}
          <span className={`result-badge ${t.key}`}>{counts[t.key]}</span>
        </button>
      ))}
    </div>
  )
}
