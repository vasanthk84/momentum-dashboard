import KpiCard from '../ui/KpiCard'
import type { ScanResults } from '../../types'

interface Props {
  results: ScanResults
  fromCache: boolean
}

export default function ScanKpis({ results, fromCache }: Props) {
  const buy   = results.buy?.length   ?? 0
  const wait  = results.wait?.length  ?? 0
  const watch = results.watchlist?.length ?? 0
  const total = Math.max(buy + wait, 1)
  const aGrade = results.buy?.filter(s => String(s.Grade).toUpperCase() === 'A').length ?? 0

  return (
    <div className="kpi-row">
      <KpiCard
        label="BUY Signals"
        value={buy}
        valueClass="pos"
        barPct={Math.round(buy / total * 100)}
        barClass="pos"
        sub={buy ? 'Enter tomorrow' : 'No signals'}
      />
      <KpiCard
        label="Wait List"
        value={wait}
        valueClass="wait"
        barPct={Math.round(wait / total * 100)}
        barClass="wait"
        sub={wait ? 'Monitor for entry' : 'None'}
      />
      <KpiCard
        label="Watchlist (T1)"
        value={watch}
        valueClass="watch"
        barPct={100}
        barClass="watch"
        sub={watch ? 'Tier 1 validated' : 'None'}
      />
      <KpiCard
        label="A-Grade Picks"
        value={aGrade}
        valueClass="pos"
        sub={aGrade && buy ? `${Math.round(aGrade / buy * 100)}% of BUY signals` : 'Highest conviction'}
      />
      <KpiCard
        label="Source"
        value={fromCache ? 'Cache' : 'Live'}
        valueClass="accent"
        sub={fromCache ? "Today's results reused" : 'Fresh scan'}
      />
    </div>
  )
}
