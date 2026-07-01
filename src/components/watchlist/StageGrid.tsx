import EmptyState from '../ui/EmptyState'
import type { WatchlistItem } from '../../types'

interface Props {
  items: WatchlistItem[]
  type: 'breakout' | 'accum'
}

export default function StageGrid({ items, type }: Props) {
  if (!items.length) {
    return (
      <EmptyState
        small
        icon={
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
          </svg>
        }
        message="None found. Run a Weekly or Combined scan."
      />
    )
  }

  const maxScore = Math.max(...items.map(r => Number(r.Total_Score ?? r.Score ?? 0)))

  return (
    <div>
      {items.slice(0, 20).map((r, i) => {
        const score = Number(r.Total_Score ?? r.Score ?? 0)
        const pct   = maxScore ? Math.round(score / maxScore * 100) : 0
        const rs    = Number(r.RS_Score ?? 0)
        const rr    = Number(r.Risk_Reward ?? 0)
        const price = Number(r.Current_Price ?? 0)

        return (
          <div key={r.Symbol} className="stage-row">
            <span className="stage-rank">{String(i + 1).padStart(2, '0')}</span>
            <span className="stage-sym">{r.Symbol}</span>
            <span className="stage-price">₹{price.toFixed(0)}</span>
            <div className="stage-bar-wrap">
              <div className={`stage-bar ${type}`} style={{ width: `${pct}%` }} />
            </div>
            <span className={`stage-score ${type === 'breakout' ? 'pos' : 'watch'}`}>{score}</span>
            <span className="stage-meta">{rs ? `RS ${rs}` : rr ? `R:R ${rr.toFixed(1)}` : '—'}</span>
          </div>
        )
      })}
    </div>
  )
}
