import { useEffect, useState } from 'react'
import KpiCard    from '../components/ui/KpiCard'
import EmptyState from '../components/ui/EmptyState'
import Tag        from '../components/ui/Tag'
import { getPerfDates, analyzePerfSignals } from '../api/performance'
import { fetchDbSignals } from '../api/db'
import { UNIVERSE_KEYS } from '../types'
import type { UniverseChoice, PerfAnalysisResponse, PerfSignal, DbSignal } from '../types'

const UNIVERSES = Object.entries(UNIVERSE_KEYS) as [UniverseChoice, string][]

type Filter = 'all' | 'open' | 'target-hit' | 'stop-hit' | 'no-data'

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all',        label: 'All'        },
  { key: 'open',       label: 'Open'       },
  { key: 'target-hit', label: 'Target Hit' },
  { key: 'stop-hit',   label: 'Stop Hit'   },
  { key: 'no-data',    label: 'No Data'    },
]

export default function PerformancePage() {
  const [universe, setUniverse] = useState<UniverseChoice>('2')
  const [dates, setDates]       = useState<string[]>([])
  const [date, setDate]         = useState('')
  const [loading, setLoading]   = useState(false)
  const [data, setData]         = useState<PerfAnalysisResponse | null>(null)
  const [error, setError]       = useState<string | null>(null)
  const [filter, setFilter]     = useState<Filter>('all')
  const [dbMap, setDbMap]       = useState<Map<string, DbSignal>>(new Map())

  useEffect(() => { loadDates(universe) }, [universe])

  useEffect(() => {
    if (date) analyze()
  }, [date]) // eslint-disable-line react-hooks/exhaustive-deps

  async function loadDates(u: UniverseChoice) {
    setDates([])
    setDate('')
    setData(null)
    try {
      const res = await getPerfDates(UNIVERSE_KEYS[u])
      setDates(res.dates)
      if (res.dates.length > 0) setDate(res.dates[0])
    } catch {
      setDates([])
    }
  }

  async function analyze() {
    if (!date) return
    setLoading(true)
    setError(null)
    setData(null)
    setDbMap(new Map())
    try {
      const [res, dbRes] = await Promise.allSettled([
        analyzePerfSignals(date, UNIVERSE_KEYS[universe]),
        fetchDbSignals({ date, universe: UNIVERSE_KEYS[universe], decision: 'BUY', limit: 500 }),
      ])
      if (res.status === 'fulfilled') setData(res.value)
      else throw res.reason
      if (dbRes.status === 'fulfilled') {
        const m = new Map<string, DbSignal>()
        dbRes.value.signals.forEach(s => m.set(s.symbol.replace('.NS', ''), s))
        setDbMap(m)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze')
    } finally {
      setLoading(false)
    }
  }

  // ── Derived statistics ────────────────────────────────────────────────────
  const signals   = data?.signals ?? []
  const withPrice = signals.filter(s => s.returnPct !== null)
  const sortedByReturn = [...withPrice].sort((a, b) => (b.returnPct ?? 0) - (a.returnPct ?? 0))
  const best  = sortedByReturn[0] ?? null
  const worst = sortedByReturn[sortedByReturn.length - 1] ?? null
  const targetHits = signals.filter(s => s.status === 'TARGET_HIT').length
  const stopHits   = signals.filter(s => s.status === 'STOP_HIT').length
  const wins  = withPrice.filter(s => s.returnPct! > 0)
  const losses = withPrice.filter(s => s.returnPct! <= 0)
  const avgWin  = wins.length  ? wins.reduce((a, b)  => a + b.returnPct!, 0) / wins.length  : 0
  const avgLoss = losses.length ? losses.reduce((a, b) => a + b.returnPct!, 0) / losses.length : 0
  const winRate    = withPrice.length ? targetHits / withPrice.length : 0
  const winRatePct = Math.round(winRate * 100)
  const ev = (winRate * avgWin) + ((1 - winRate) * avgLoss)

  // Grade breakdown
  const gradeBreakdown = ['A', 'B', 'C'].map(g => {
    const gs    = withPrice.filter(s => s.grade === g)
    const gHits = gs.filter(s => s.status === 'TARGET_HIT')
    const gWins = gs.filter(s => s.returnPct! > 0)
    const gLoss = gs.filter(s => s.returnPct! <= 0)
    const gAvgRet = gs.length ? gs.reduce((a, b) => a + b.returnPct!, 0) / gs.length : 0
    const gAvgW   = gWins.length ? gWins.reduce((a, b) => a + b.returnPct!, 0) / gWins.length : 0
    const gAvgL   = gLoss.length ? gLoss.reduce((a, b) => a + b.returnPct!, 0) / gLoss.length : 0
    const gWr     = gs.length ? gHits.length / gs.length : 0
    const gEv     = (gWr * gAvgW) + ((1 - gWr) * gAvgL)
    return { grade: g, count: gs.length, winRate: Math.round(gWr * 100), avgRet: gAvgRet, ev: gEv }
  }).filter(g => g.count > 0)

  // Status filter
  const countOf = (f: Filter) => {
    if (f === 'all')        return signals.length
    if (f === 'open')       return signals.filter(s => s.status === 'OPEN').length
    if (f === 'target-hit') return targetHits
    if (f === 'stop-hit')   return stopHits
    return signals.filter(s => s.status === 'NO_DATA').length
  }
  const filtered: PerfSignal[] =
    filter === 'all'        ? signals :
    filter === 'open'       ? signals.filter(s => s.status === 'OPEN') :
    filter === 'target-hit' ? signals.filter(s => s.status === 'TARGET_HIT') :
    filter === 'stop-hit'   ? signals.filter(s => s.status === 'STOP_HIT') :
    signals.filter(s => s.status === 'NO_DATA')

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

      {/* Controls */}
      <div className="perf-controls">
        <div>
          <label className="ctrl-label">Universe</label>
          <select className="ctrl-select" style={{ width: 160 }} value={universe}
            onChange={e => setUniverse(e.target.value as UniverseChoice)}>
            {UNIVERSES.map(([k, v]) => <option key={k} value={k}>{v.replace(/_/g, ' ')}</option>)}
          </select>
        </div>
        <div>
          <label className="ctrl-label">Scan Date</label>
          <select className="ctrl-select" style={{ width: 160 }} value={date}
            onChange={e => setDate(e.target.value)}>
            <option value="">{dates.length ? 'Select date…' : 'No history found'}</option>
            {dates.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <button className="btn-run" style={{ alignSelf: 'flex-end' }} disabled={!date || loading} onClick={analyze}>
          {loading ? 'Analyzing…' : 'Refresh'}
        </button>
      </div>

      {error && <div className="error-bar">⚠ {error}</div>}

      {loading && (
        <div className="error-bar" style={{ background: 'var(--surface-2)', color: 'var(--ink-2)', border: '1px solid var(--border)' }}>
          Fetching live prices via yfinance — this takes ~10–20 s…
        </div>
      )}

      {/* Best / Worst callout */}
      {data && best && (
        <div style={{ display: 'flex', gap: 8, marginBottom: 'var(--gap)', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: 'var(--muted)', marginRight: 6 }}>Best</span>
            <b>{best.symbol}</b>
            <span className={best.returnPct! >= 0 ? 'pos' : 'neg'} style={{ marginLeft: 8 }}>{best.returnPct! >= 0 ? '+' : ''}{best.returnPct!.toFixed(2)}%</span>
            <span style={{ color: 'var(--muted)', marginLeft: 8 }}>{best.grade} · Score {best.score}</span>
          </div>
          {worst && worst.symbol !== best.symbol && (
            <div style={{ flex: 1, minWidth: 200, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 14px', fontSize: 12 }}>
              <span style={{ color: 'var(--muted)', marginRight: 6 }}>Worst</span>
              <b>{worst.symbol}</b>
              <span className="neg" style={{ marginLeft: 8 }}>{worst.returnPct!.toFixed(2)}%</span>
              <span style={{ color: 'var(--muted)', marginLeft: 8 }}>{worst.grade} · Score {worst.score}</span>
            </div>
          )}
          <div style={{ flex: 1, minWidth: 200, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: 'var(--muted)', marginRight: 6 }}>Spread</span>
            <span className="pos">{best.returnPct!.toFixed(2)}%</span>
            <span style={{ color: 'var(--muted)', margin: '0 4px' }}>→</span>
            <span className={worst && worst.returnPct! < 0 ? 'neg' : 'pos'}>{worst?.returnPct!.toFixed(2)}%</span>
            <span style={{ color: 'var(--muted)', marginLeft: 8 }}>{data.daysHeld}d hold</span>
          </div>
        </div>
      )}

      {/* KPIs */}
      {data && (
        <div className="perf-stat-grid">
          <KpiCard
            label="Target Hit Rate"
            value={`${winRatePct}%`}
            valueClass={winRatePct >= 40 ? 'pos' : 'neg'}
            barPct={winRatePct}
            barClass={winRatePct >= 40 ? 'pos' : 'neg'}
            sub={`${targetHits} of ${withPrice.length} hit T1`}
          />
          <KpiCard
            label="Avg Win"
            value={avgWin > 0 ? `+${avgWin.toFixed(2)}%` : '—'}
            valueClass="pos"
            sub={`${wins.length} winning trades`}
          />
          <KpiCard
            label="Avg Loss"
            value={avgLoss < 0 ? `${avgLoss.toFixed(2)}%` : '—'}
            valueClass="neg"
            sub={`${losses.length} losing trades`}
          />
          <KpiCard
            label="Expected Value"
            value={`${ev >= 0 ? '+' : ''}${ev.toFixed(2)}%`}
            valueClass={ev >= 0 ? 'pos' : 'neg'}
            sub="Per-trade EV · (WR×AvgW)+(LR×AvgL)"
          />
          <KpiCard
            label="Days Held"
            value={data.daysHeld}
            valueClass="mono"
            sub={`Since ${data.scanDate}`}
          />
        </div>
      )}

      {/* Signal table */}
      <div className="card card-flush">
        <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
          <div className="card-title">Signal Returns</div>
          <div className="card-sub">
            Sorted by return % · {data?.signals.length ?? 0} signals · live prices via yfinance
          </div>
        </div>

        {/* Status filter tabs */}
        {data && (
          <div style={{ display: 'flex', gap: 4, padding: '0 16px 10px', flexWrap: 'wrap' }}>
            {FILTERS.map(f => (
              <button
                key={f.key}
                className="btn-ghost"
                style={{
                  fontSize: 11, padding: '4px 10px',
                  ...(filter === f.key
                    ? { background: 'var(--accent)', color: '#fff', borderColor: 'var(--accent)' }
                    : {}),
                }}
                onClick={() => setFilter(f.key)}
              >
                {f.label} <span style={{ opacity: .65 }}>({countOf(f.key)})</span>
              </button>
            ))}
          </div>
        )}

        {!data && !loading && (
          <EmptyState icon={EmptyIcon} message="Select a universe & date — analysis loads automatically." />
        )}

        {data && (
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  {['Symbol', 'Grade', 'Score', 'Entry ₹', 'Current ₹', 'Return %', 'R-Multiple', 'Stop ₹', 'Target 1 ₹', 'Target 2 ₹', 'Wyckoff', 'VWAP Dev%', 'Status'].map(h => (
                    <th key={h} className={['Score', 'Entry ₹', 'Current ₹', 'Return %', 'R-Multiple', 'Stop ₹', 'Target 1 ₹', 'Target 2 ₹', 'VWAP Dev%'].includes(h) ? 'num' : ''}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map(s => {
                  const ret    = s.returnPct
                  const retCls = ret === null ? '' : ret >= 0 ? 'pos' : 'neg'
                  const rMult  = ret !== null && s.riskPct ? ret / s.riskPct : null
                  const rCls   = rMult !== null ? (rMult >= 0 ? 'pos' : 'neg') : ''
                  const statusKey = s.status.toLowerCase().replace(/_/g, '-')
                  const db    = dbMap.get(s.symbol)
                  const vdev  = db?.vwap_deviation_pct ?? null
                  const wyck  = db?.wyckoff_phase ?? null
                  const wyckShort = wyck ? wyck.replace('PHASE_', '').replace('_', ' ') : '—'
                  const reasons = db?.reasons ?? []
                  return (
                    <tr key={s.symbol} title={reasons.length ? reasons.join(' · ') : undefined}>
                      <td><b>{s.symbol}</b></td>
                      <td><Tag variant={s.grade}>{s.grade || '—'}</Tag></td>
                      <td className="num">{s.score}</td>
                      <td className="num">₹{s.signalPrice.toFixed(2)}</td>
                      <td className="num">{s.currentPrice != null ? `₹${s.currentPrice.toFixed(2)}` : '—'}</td>
                      <td className={`num ${retCls}`}>{ret !== null ? `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%` : '—'}</td>
                      <td className={`num ${rCls}`}>{rMult !== null ? `${rMult >= 0 ? '+' : ''}${rMult.toFixed(2)}R` : '—'}</td>
                      <td className="num neg">₹{s.stopLoss.toFixed(2)}</td>
                      <td className="num pos">₹{s.target1.toFixed(2)}</td>
                      <td className="num" style={{ color: 'var(--ink-2)' }}>{db?.target_2 ? `₹${db.target_2.toFixed(2)}` : '—'}</td>
                      <td style={{ fontSize: 11, color: 'var(--ink-2)' }}>{wyckShort}</td>
                      <td className={`num ${vdev != null ? (vdev >= 0 ? 'pos' : 'neg') : ''}`}>
                        {vdev != null ? `${vdev >= 0 ? '+' : ''}${vdev.toFixed(2)}%` : '—'}
                      </td>
                      <td><Tag variant={statusKey}>{s.status.replace(/_/g, ' ')}</Tag></td>
                    </tr>
                  )
                })}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={13} style={{ textAlign: 'center', padding: '24px 0', color: 'var(--muted)', fontSize: 12 }}>
                      No signals in this filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
            <div className="tbl-foot">
              <span>{filtered.length} of {signals.length} signals · {data.daysHeld}d since entry</span>
              <span className="muted">Stop hits: {stopHits} · Target hits: {targetHits}</span>
            </div>
          </div>
        )}
      </div>

      {/* Grade breakdown */}
      {data && gradeBreakdown.length > 0 && (
        <div className="card" style={{ marginTop: 'var(--gap)' }}>
          <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
            <div className="card-title">Grade Breakdown</div>
            <div className="card-sub">Does the grading have edge? Higher grade should deliver better EV.</div>
          </div>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Grade</th>
                  <th className="num">Signals</th>
                  <th className="num">Target Hit Rate</th>
                  <th className="num">Avg Return %</th>
                  <th className="num">Expected Value</th>
                </tr>
              </thead>
              <tbody>
                {gradeBreakdown.map(g => (
                  <tr key={g.grade}>
                    <td><Tag variant={g.grade}>{g.grade}</Tag></td>
                    <td className="num">{g.count}</td>
                    <td className={`num ${g.winRate >= 40 ? 'pos' : 'neg'}`}>{g.winRate}%</td>
                    <td className={`num ${g.avgRet >= 0 ? 'pos' : 'neg'}`}>{g.avgRet >= 0 ? '+' : ''}{g.avgRet.toFixed(2)}%</td>
                    <td className={`num ${g.ev >= 0 ? 'pos' : 'neg'}`}>{g.ev >= 0 ? '+' : ''}{g.ev.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
