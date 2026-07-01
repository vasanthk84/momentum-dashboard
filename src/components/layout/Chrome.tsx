import { useEffect, useState } from 'react'
import type { Mood } from '../../types'

interface Props {
  mood: Mood
  onMoodChange: (m: Mood) => void
  scanning: boolean
  logsVisible: boolean
  onToggleLogs: () => void
  onRunScan: () => void
}

const MOODS: { key: Mood; label: string }[] = [
  { key: 'bloomberg', label: 'Bloomberg' },
  { key: 'carbon',    label: 'Carbon'    },
  { key: 'paper',     label: 'Paper'     },
]

export default function Chrome({ mood, onMoodChange, scanning, logsVisible, onToggleLogs, onRunScan }: Props) {
  const [clock, setClock] = useState('—')

  useEffect(() => {
    const tick = () => {
      const now = new Intl.DateTimeFormat('en-IN', {
        timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
      }).format(new Date())
      setClock('IST ' + now)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="chrome">
      <div className="lights"><span /><span /><span /></div>
      <div className="chrome-t"><b>MS</b>Momentum Scanner · NSE</div>

      <div className="chrome-meta">
        <span>
          <span className={`live-dot ${scanning ? 'scanning' : ''}`} />
          {scanning ? 'SCANNING' : 'READY'}
        </span>
        <span style={{ color: 'var(--muted-2)' }}>{clock}</span>
      </div>

      <div className="chrome-actions">
        <div className="theme-switch">
          {MOODS.map(m => (
            <button
              key={m.key}
              className={mood === m.key ? 'on' : ''}
              onClick={() => onMoodChange(m.key)}
            >
              <span className={`theme-swatch ${m.key}`} />
              {m.label}
            </button>
          ))}
        </div>

        <button className="btn-ghost" onClick={onToggleLogs}>
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6">
            <rect x="2" y="2" width="12" height="12" rx="2" /><path d="M5 6h6M5 9h4" />
          </svg>
          {logsVisible ? 'Hide Logs' : 'Logs'}
        </button>

        <button className="btn-run" disabled={scanning} onClick={onRunScan}>
          {scanning ? (
            <>
              <svg className="spin-svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
              Scanning…
            </>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
                <path d="M4 2l10 6-10 6V2z" />
              </svg>
              Run Scan
            </>
          )}
        </button>
      </div>
    </div>
  )
}
