import { useState } from 'react'
import KpiCard    from '../components/ui/KpiCard'
import EmptyState from '../components/ui/EmptyState'
import Tag        from '../components/ui/Tag'
import { getBatchAnalytics } from '../api/analytics'
import { UNIVERSE_KEYS } from '../types'
import type { UniverseChoice, BatchAnalyticsResponse } from '../types'

const UNIVERSES = Object.entries(UNIVERSE_KEYS) as [UniverseChoice, string][]

export default function AnalyticsPage() {
  const [universe, setUniverse] = useState<UniverseChoice>('2')
  const [limit, setLimit]       = useState(5)
  const [loading, setLoading]   = useState(false)
  const [data, setData]         = useState<BatchAnalyticsResponse | null>(null)
  const [error, setError]       = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const res = await getBatchAnalytics(UNIVERSE_KEYS[universe], limit)
      setData(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  const signals    = data?.signals ?? []
  const validRets  = signals.filter(s => s.returnPct != null)
  const avgRet     = validRets.length ? validRets.reduce((a, b) => a + b.returnPct, 0) / validRets.length : 0
  const winners    = validRets.filter(s => s.returnPct > 0).length
  const winRate    = validRets.length ? Math.round(winners / validRets.length * 100) : 0
  const targets    = signals.filter(s => s.status === 'TARGET_HIT').length
  const stops      = signals.filter(s => s.status === 'STOP_HIT').length
  const sorted     = [...signals].sort((a, b) => (b.returnPct ?? 0) - (a.returnPct ?? 0))

  const EmptyIcon = (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/>
      <line x1="12" y1="2" x2="12" y2="22"/>
    </svg>
  )

  return (
    <div>
      <div className="section-head">
        <div>
          <div className="eyebrow">Aggregate</div>
          <h2 className="section-h">Batch Analytics</h2>
        </div>
      </div>

      <div className="perf-controls">
        <div>
          <label className="ctrl-label">Universe</label>
          <select className="ctrl-select" style={{ width: 160 }} value={universe} onChange={e => setUniverse(e.target.value as UniverseChoice)}>
            {UNIVERSES.map(([k, v]) => <option key={k} value={k}>{v.replace(/_/g, ' ')}</option>)}
          </select>
        </div>
        <div>
          <label className="ctrl-label">Last N Scans</label>
          <select className="ctrl-select" style={{ width: 100 }} value={limit} onChange={e => setLimit(Number(e.target.value))}>
            {[3, 5, 10].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <button className="btn-run" style={{ alignSelf: 'flex-end' }} disabled={loading} onClick={load}>
          {loading ? 'Loading…' : 'Load Analytics'}
        </button>
      </div>

      {error && <div className="error-bar">⚠ {error}</div>}

      {data && (
        <div className="kpi-row">
          <KpiCard label="Total Signals" value={signals.length}    valueClass="watch" sub={`Across ${data.dates.length} scan dates`} />
          <KpiCard label="Avg Return"    value={`${avgRet >= 0 ? '+' : ''}${avgRet.toFixed(2)}%`} valueClass={avgRet >= 0 ? 'pos' : 'neg'} sub="Across all signals" />
          <KpiCard label="Win Rate"      value={`${winRate}%`}     valueClass={winRate >= 50 ? 'pos' : 'neg'} barPct={winRate} barClass={winRate >= 50 ? 'pos' : 'neg'} sub="Profitable signals" />
          <KpiCard label="Target Hits"   value={targets}           valueClass="pos" sub="T1 achieved" />
          <KpiCard label="Stop Hits"     value={stops}             valueClass="neg" sub="SL triggered" />
        </div>
      )}

      <div className="card card-flush">
        <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
          <div className="card-title">All Signals — Aggregated</div>
          <div className="card-sub">{data ? data.dates.join(', ') : 'Select universe & click Load Analytics'}</div>
        </div>

        {!data && !loading && (
          <EmptyState icon={EmptyIcon} message={<>Select universe &amp; click <b>Load Analytics</b>.</>} />
        )}

        {loading && <EmptyState icon={EmptyIcon} message="Loading batch data…" />}

        {data && (
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  {['Date','Symbol','Grade','Return %','Days','Status','Vol Ratio','Market'].map(h => (
                    <th key={h} className={['Return %','Days','Vol Ratio'].includes(h) ? 'num' : ''}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((s, i) => {
                  const ret = s.returnPct
                  const retCls = ret >= 0 ? 'pos' : 'neg'
                  const statusKey = s.status.toLowerCase().replace(/_/g, '-')
                  const trendCls = s.marketTrend?.toLowerCase().includes('bull') ? 'pos'
                    : s.marketTrend?.toLowerCase().includes('bear') ? 'neg' : ''
                  return (
                    <tr key={i}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)' }}>{s.date}</td>
                      <td><b>{s.symbol}</b></td>
                      <td><Tag variant={s.grade}>{s.grade || '—'}</Tag></td>
                      <td className={`num ${retCls}`}>{ret != null ? `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%` : '—'}</td>
                      <td className="num">{s.daysHeld}</td>
                      <td><Tag variant={statusKey}>{s.status.replace(/_/g, ' ')}</Tag></td>
                      <td className={`num ${s.volumeRatio >= 1.5 ? 'pos' : ''}`}>{s.volumeRatio ? s.volumeRatio.toFixed(2) : '—'}</td>
                      <td><span className={trendCls}>{s.marketTrend || '—'}</span></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="tbl-foot">
              <span>{sorted.length} signals total</span>
              <span className="muted">Sorted by return %</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
