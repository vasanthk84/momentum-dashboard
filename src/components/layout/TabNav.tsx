import type { MainTab } from '../../types'

const TABS: { key: MainTab; label: string }[] = [
  { key: 'scanner',     label: 'Scanner'     },
  { key: 'watchlist',   label: 'Watchlist'   },
  { key: 'performance', label: 'Performance' },
  { key: 'analytics',   label: 'Analytics'   },
]

interface Props {
  active: MainTab
  onChange: (t: MainTab) => void
}

export default function TabNav({ active, onChange }: Props) {
  return (
    <nav className="tb-tabs">
      {TABS.map(t => (
        <button
          key={t.key}
          className={`tb-tab ${active === t.key ? 'active' : ''}`}
          onClick={() => onChange(t.key)}
        >
          {t.label}
        </button>
      ))}
    </nav>
  )
}
