import { useEffect, useState } from 'react'
import KpiCard   from '../components/ui/KpiCard'
import EmptyState from '../components/ui/EmptyState'
import Tag        from '../components/ui/Tag'
import { getPerfDates, analyzePerfSignals } from '../api/performance'
import { UNIVERSE_KEYS } from '../types'
import type { UniverseChoice, PerfAnalysisResponse } from '../types'

const UNIVERSES = Object.entries(UNIVERSE_KEYS) as [UniverseChoice, string][]

export default function PerformancePage() {
  const [universe, setUniverse] = useState<UniverseChoice>('2')
  const [dates, setDates]       = useState<string[]>([])
  const [date, setDate]         = useState('')
  const [loading, setLoading]   = useState(false)
  const [data, setData]         = useState<PerfAnalysisResponse | null>(null)
  const [error, setError]       = useState<string | null>(null)

  useEffect(() => {
    loadDates(universe)
  }, [universe])

  async function loadDates(u: UniverseChoice) {
    setDates([])
    setDate('')
    try {
      const res = await getPerfDates(UNIVERSE_KEYS[u])
      setDates(res.dates)
    } catch {
      setDates([])
    }
  }

  async function analyze() {
    if (!date) return
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const res = await analyzePerfSignals(date, UNIVERSE_KEYS[universe])
      setData(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze')
    } finally {
      setLoading(false)
    }
  }

  const avgRet  = data?.avgReturn ?? 0
  const winRate = data ? Math.round(data.winners / data.totalSignals * 100) : 0

  const EmptyIcon = (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
    </svg>
  )

  return (
    <div>
      <div className="section-head">
        <div>
          <div className="eyebrow">Historical</div>
          <h2 className="section-h">Signal Performance Tracker</h2>
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
          <label className="ctrl-label">Scan Date</label>
          <select className="ctrl-select" style={{ width: 160 }} value={date} onChange={e => setDate(e.target.value)}>
            <option value="">{dates.length ? 'Select date…' : 'No history found'}</option>
            {dates.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <button className="btn-run" style={{ alignSelf: 'flex-end' }} disabled={!date || loading} onClick={analyze}>
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </div>

      {error && <div className="error-bar">⚠ {error}</div>}

      {data && (
        <div className="perf-stat-grid">
          <KpiCard
            label="Avg Return"
            value={`${avgRet >= 0 ? '+' : ''}${avgRet.toFixed(2)}%`}
            valueClass={avgRet >= 0 ? 'pos' : 'neg'}
            sub={avgRet >= 0 ? 'Positive avg return' : 'Negative avg return'}
          />
          <KpiCard
            label="Win Rate"
            value={`${winRate}%`}
            valueClass={winRate >= 50 ? 'pos' : 'neg'}
            barPct={winRate}
            barClass={winRate >= 50 ? 'pos' : 'neg'}
            sub={`${data.winners} of ${data.totalSignals} profitable`}
          />
          <KpiCard label="Signals"    value={data.totalSignals} valueClass="watch" sub={`Scan: ${data.scanDate}`} />
          <KpiCard label="Days Held"  value={data.daysHeld}     valueClass="mono"  sub="Since scan date" />
        </div>
      )}

      <div className="card card-flush">
        <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
          <div className="card-title">Signal Returns</div>
          <div className="card-sub">Sorted by return % · live prices via yfinance</div>
        </div>

        {!data && !loading && (
          <EmptyState icon={EmptyIcon} message={<>Select a universe &amp; date then click <b>Analyze</b>.</>} />
        )}

        {loading && (
          <EmptyState icon={EmptyIcon} message="Fetching live prices…" />
        )}

        {data && (
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  {['Symbol','Grade','Score','Entry ₹','Current ₹','Return %','Stop ₹','Target ₹','Status'].map(h => (
                    <th key={h} className={['Score','Entry ₹','Current ₹','Return %','Stop ₹','Target ₹'].includes(h) ? 'num' : ''}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.signals.map(s => {
                  const ret = s.returnPct
                  const retCls = ret === null ? '' : ret >= 0 ? 'pos' : 'neg'
                  const statusKey = s.status.toLowerCase().replace('_', '-')
                  return (
                    <tr key={s.symbol}>
                      <td><b>{s.symbol}</b></td>
                      <td><Tag variant={s.grade}>{s.grade || '—'}</Tag></td>
                      <td className="num">{s.score}</td>
                      <td className="num">₹{s.signalPrice.toFixed(2)}</td>
                      <td className="num">{s.currentPrice ? `₹${s.currentPrice.toFixed(2)}` : '—'}</td>
                      <td className={`num ${retCls}`}>{ret !== null ? `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%` : '—'}</td>
                      <td className="num neg">₹{s.stopLoss.toFixed(2)}</td>
                      <td className="num pos">₹{s.target1.toFixed(2)}</td>
                      <td><Tag variant={statusKey}>{s.status.replace('_', ' ')}</Tag></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="tbl-foot">
              <span>{data.signals.length} signals · {data.daysHeld}d since entry</span>
              <span className="muted">Live prices via yfinance</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
