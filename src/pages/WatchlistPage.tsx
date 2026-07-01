import StageGrid   from '../components/watchlist/StageGrid'
import SignalTable  from '../components/scanner/SignalTable'
import EmptyState   from '../components/ui/EmptyState'
import Tag          from '../components/ui/Tag'
import type { WatchlistItem } from '../types'

interface Props {
  items: WatchlistItem[]
}

export default function WatchlistPage({ items }: Props) {
  const breakout = items.filter(r => String(r.Stage).includes('BREAKOUT'))
  const accum    = items.filter(r => !String(r.Stage).includes('BREAKOUT'))

  const EmptyIcon = (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>
    </svg>
  )

  if (!items.length) {
    return (
      <EmptyState
        icon={EmptyIcon}
        title="No watchlist data"
        message={<>Run a <b>Weekly</b> or <b>Combined</b> scan from the Scanner tab to populate the watchlist.</>}
      />
    )
  }

  return (
    <div>
      <div className="section-head">
        <div>
          <div className="eyebrow">Tier 1 — Weekly</div>
          <h2 className="section-h">Breakout Watchlist · <span className="muted">{items.length} stocks</span></h2>
        </div>
      </div>

      <div className="two-col">
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">🚀 Breakout Ready</div>
              <div className="card-sub">Tight base · near pivot · high RS</div>
            </div>
            <Tag variant="breakout">{breakout.length}</Tag>
          </div>
          <StageGrid items={breakout} type="breakout" />
        </div>

        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">📊 Accumulation</div>
              <div className="card-sub">Building base · monitor for tightening</div>
            </div>
            <Tag variant="accum">{accum.length}</Tag>
          </div>
          <StageGrid items={accum} type="accum" />
        </div>
      </div>

      <div className="card card-flush">
        <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
          <div className="card-title">Full Watchlist</div>
          <div className="card-sub">All Tier 1 momentum stocks — scored &amp; ranked</div>
        </div>
        <SignalTable
          data={items as unknown as Record<string, unknown>[]}
          tab="watchlist"
          footer="Tier 1 — Weekly scan"
        />
      </div>
    </div>
  )
}
