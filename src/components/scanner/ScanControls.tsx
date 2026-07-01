import { useState } from 'react'
import { clearCache } from '../../api/analytics'
import { UNIVERSE_KEYS, SCAN_TYPE_LABELS, UNIVERSE_LABELS } from '../../types'
import type { MainChoice, UniverseChoice } from '../../types'

interface Props {
  mainChoice: MainChoice
  universeChoice: UniverseChoice
  onMainChange: (v: MainChoice) => void
  onUniverseChange: (v: UniverseChoice) => void
}

export default function ScanControls({ mainChoice, universeChoice, onMainChange, onUniverseChange }: Props) {
  const [clearing, setClearing] = useState(false)
  const [msg, setMsg]           = useState<string | null>(null)

  async function handleClearCache() {
    setClearing(true)
    setMsg(null)
    try {
      const res = await clearCache(UNIVERSE_KEYS[universeChoice])
      setMsg(`Cleared ${res.deleted} cache file(s). Next scan will run fresh.`)
      setTimeout(() => setMsg(null), 5000)
    } catch {
      setMsg('Clear failed.')
    } finally {
      setClearing(false)
    }
  }

  return (
    <div className="scan-controls">
      <div>
        <label className="ctrl-label">Scan Type</label>
        <select
          className="ctrl-select"
          value={mainChoice}
          onChange={e => onMainChange(e.target.value as MainChoice)}
        >
          {(Object.entries(SCAN_TYPE_LABELS) as [MainChoice, string][]).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="ctrl-label">Stock Universe</label>
        <select
          className="ctrl-select"
          value={universeChoice}
          onChange={e => onUniverseChange(e.target.value as UniverseChoice)}
        >
          {(Object.entries(UNIVERSE_LABELS) as [UniverseChoice, string][]).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="ctrl-label">Cache</label>
        <button
          className="btn-ghost"
          style={{ fontSize: 11, marginTop: 2 }}
          disabled={clearing}
          onClick={handleClearCache}
          title="Delete today's cached results so the next Run Scan calls Python fresh"
        >
          {clearing ? 'Clearing…' : 'Clear Today'}
        </button>
        {msg && (
          <div style={{ fontSize: 10, color: 'var(--pos)', marginTop: 4, maxWidth: 200 }}>{msg}</div>
        )}
      </div>
    </div>
  )
}
