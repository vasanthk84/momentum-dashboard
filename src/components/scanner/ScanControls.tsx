import type { MainChoice, UniverseChoice } from '../../types'
import { SCAN_TYPE_LABELS, UNIVERSE_LABELS } from '../../types'

interface Props {
  mainChoice: MainChoice
  universeChoice: UniverseChoice
  onMainChange: (v: MainChoice) => void
  onUniverseChange: (v: UniverseChoice) => void
}

export default function ScanControls({ mainChoice, universeChoice, onMainChange, onUniverseChange }: Props) {
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
        <div style={{ fontSize: '11px', color: 'var(--muted-2)', paddingTop: 4 }}>
          Auto · same-day results reused
        </div>
      </div>
    </div>
  )
}
