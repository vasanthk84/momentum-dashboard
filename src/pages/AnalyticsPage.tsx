import { useState } from 'react'
import KpiCard    from '../components/ui/KpiCard'
import EmptyState from '../components/ui/EmptyState'
import Tag        from '../components/ui/Tag'
import { getBatchAnalytics, clearCache } from '../api/analytics'
import { fetchDbSignals } from '../api/db'
import { UNIVERSE_KEYS } from '../types'
import type { UniverseChoice, BatchAnalyticsResponse, BatchSignal, DbSignal } from '../types'

const UNIVERSES = Object.entries(UNIVERSE_KEYS) as [UniverseChoice, string][]

const MARKET_TREND_ORDER = ['STRONG_BULLISH', 'BULLISH', 'NEUTRAL', 'BEARISH', 'STRONG_BEARISH']

function normalizeStatus(s: string): string {
  return s === 'PROFIT' || s === 'LOSS' ? 'OPEN' : s
}

export default function AnalyticsPage() {
  const [universe, setUniverse] = useState<UniverseChoice>('2')
  const [limit, setLimit]       = useState(5)
  const [loading, setLoading]   = useState(false)
  const [clearing, setClearing] = useState(false)
  const [data, setData]         = useState<BatchAnalyticsResponse | null>(null)
  const [dbSignals, setDbSignals] = useState<DbSignal[]>([])
  const [error, setError]       = useState<string | null>(null)
  const [cacheMsg, setCacheMsg] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError(null)
    setCacheMsg(null)
    setDbSignals([])
    try {
      const [res, dbRes] = await Promise.allSettled([
        getBatchAnalytics(UNIVERSE_KEYS[universe], limit),
        fetchDbSignals({ universe: UNIVERSE_KEYS[universe], decision: 'BUY', limit: 1000 }),
      ])
      if (res.status === 'fulfilled') setData(res.value)
      else throw res.reason
      if (dbRes.status === 'fulfilled') setDbSignals(dbRes.value.signals)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  async function handleClearCache() {
    const univName = UNIVERSE_KEYS[universe].replace(/_/g, ' ')
    if (!window.confirm(`Clear today's scan cache for ${univName}?\n\nThe next scan will run fresh Python (not cached).`)) return
    setClearing(true)
    setCacheMsg(null)
    try {
      const res = await clearCache(UNIVERSE_KEYS[universe])
      setCacheMsg(`Cleared ${res.deleted} cache file(s) for today. Run a fresh scan to rebuild.`)
    } catch {
      setCacheMsg('Cache clear failed — see server logs.')
    } finally {
      setClearing(false)
    }
  }

  // ── Derived stats ─────────────────────────────────────────────────────────
  const signals   = data?.signals ?? []
  const withPrice = signals.filter(s => normalizeStatus(s.status) !== 'NO_DATA' && s.returnPct != null)
  const targets   = signals.filter(s => s.status === 'TARGET_HIT').length
  const stops     = signals.filter(s => s.status === 'STOP_HIT').length
  const opens     = signals.filter(s => ['OPEN', 'PROFIT', 'LOSS'].includes(s.status)).length
  const wins      = withPrice.filter(s => s.returnPct > 0)
  const losses    = withPrice.filter(s => s.returnPct <= 0)
  const avgWin    = wins.length   ? wins.reduce((a, b)   => a + b.returnPct, 0) / wins.length   : 0
  const avgLoss   = losses.length ? losses.reduce((a, b) => a + b.returnPct, 0) / losses.length : 0
  const winRate   = withPrice.length ? targets / withPrice.length : 0
  const winRatePct = Math.round(winRate * 100)
  const ev        = (winRate * avgWin) + ((1 - winRate) * avgLoss)
  const avgRet    = withPrice.length ? withPrice.reduce((a, b) => a + b.returnPct, 0) / withPrice.length : 0

  const dates = data?.dates ?? []
  const dateRangeLabel = dates.length > 1
    ? `${dates[dates.length - 1]} → ${dates[0]} (${dates.length} scans)`
    : dates.length === 1 ? dates[0] : ''

  // Grade breakdown
  const gradeBreakdown = ['A', 'B', 'C'].map(g => {
    const gs    = withPrice.filter(s => s.grade === g)
    const gHits = gs.filter(s => s.status === 'TARGET_HIT')
    const gWins = gs.filter(s => s.returnPct > 0)
    const gLoss = gs.filter(s => s.returnPct <= 0)
    const gAvgRet = gs.length ? gs.reduce((a, b) => a + b.returnPct, 0) / gs.length : 0
    const gAvgW   = gWins.length ? gWins.reduce((a, b) => a + b.returnPct, 0) / gWins.length : 0
    const gAvgL   = gLoss.length ? gLoss.reduce((a, b) => a + b.returnPct, 0) / gLoss.length : 0
    const gWr     = gs.length ? gHits.length / gs.length : 0
    const gEv     = (gWr * gAvgW) + ((1 - gWr) * gAvgL)
    return { grade: g, count: gs.length, winRate: Math.round(gWr * 100), avgRet: gAvgRet, ev: gEv }
  }).filter(g => g.count > 0)

  // Market condition breakdown
  const allTrends = [...new Set(signals.map(s => s.marketTrend).filter(Boolean))]
  const orderedTrends = [...MARKET_TREND_ORDER.filter(t => allTrends.includes(t)), ...allTrends.filter(t => !MARKET_TREND_ORDER.includes(t))]
  const marketBreakdown = orderedTrends.map(trend => {
    const ms    = withPrice.filter(s => s.marketTrend === trend)
    if (!ms.length) return null
    const mHits = ms.filter(s => s.status === 'TARGET_HIT')
    const mWins = ms.filter(s => s.returnPct > 0)
    const mLoss = ms.filter(s => s.returnPct <= 0)
    const mAvgRet = ms.reduce((a, b) => a + b.returnPct, 0) / ms.length
    const mWr     = mHits.length / ms.length
    const mAvgW   = mWins.length ? mWins.reduce((a, b) => a + b.returnPct, 0) / mWins.length : 0
    const mAvgL   = mLoss.length ? mLoss.reduce((a, b) => a + b.returnPct, 0) / mLoss.length : 0
    const mEv     = (mWr * mAvgW) + ((1 - mWr) * mAvgL)
    return { trend, count: ms.length, winRate: Math.round(mWr * 100), avgRet: mAvgRet, ev: mEv }
  }).filter(Boolean) as { trend: string; count: number; winRate: number; avgRet: number; ev: number }[]

  const sorted = [...signals].sort((a, b) => (b.returnPct ?? 0) - (a.returnPct ?? 0))

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
          <div className="eyebrow">Strategy</div>
          <h2 className="section-h">System Analytics</h2>
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
          <label className="ctrl-label">Last N Scans</label>
          <select className="ctrl-select" style={{ width: 100 }} value={limit}
            onChange={e => setLimit(Number(e.target.value))}>
            {[3, 5, 10].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <button className="btn-run" style={{ alignSelf: 'flex-end' }} disabled={loading} onClick={load}>
          {loading ? 'Loading…' : 'Load Analytics'}
        </button>
        <button className="btn-ghost" style={{ alignSelf: 'flex-end' }} disabled={clearing} onClick={handleClearCache}>
          {clearing ? 'Clearing…' : 'Clear Cache'}
        </button>
      </div>

      {error    && <div className="error-bar">⚠ {error}</div>}
      {cacheMsg && (
        <div className="error-bar" style={{ background: 'var(--surface-2)', color: 'var(--ink-2)', border: '1px solid var(--border)' }}>
          {cacheMsg}
        </div>
      )}

      {/* KPIs */}
      {data && (
        <div className="kpi-row">
          <KpiCard label="Total Signals"    value={signals.length}    valueClass="watch" sub={dateRangeLabel} />
          <KpiCard label="Avg Return"       value={`${avgRet >= 0 ? '+' : ''}${avgRet.toFixed(2)}%`} valueClass={avgRet >= 0 ? 'pos' : 'neg'} sub={`${withPrice.length} priced signals`} />
          <KpiCard label="Target Hit Rate"  value={`${winRatePct}%`}  valueClass={winRatePct >= 40 ? 'pos' : 'neg'} barPct={winRatePct} barClass={winRatePct >= 40 ? 'pos' : 'neg'} sub={`${targets} T1 · ${stops} SL`} />
          <KpiCard label="Expected Value"   value={`${ev >= 0 ? '+' : ''}${ev.toFixed(2)}%`} valueClass={ev >= 0 ? 'pos' : 'neg'} sub="(WR×AvgW)+(LR×AvgL)" />
          <KpiCard label="Open Positions"   value={opens}             valueClass="watch" sub={`${stops} stopped out`} />
        </div>
      )}

      {/* Grade breakdown */}
      {data && gradeBreakdown.length > 0 && (
        <div className="card" style={{ marginBottom: 'var(--gap)' }}>
          <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
            <div className="card-title">Grade Breakdown</div>
            <div className="card-sub">Does the grading system have predictive edge? Grade A should outperform Grade B.</div>
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

      {/* Score cutoff analysis */}
      {data && withPrice.length > 0 && (() => {
        const thresholds = [60, 70, 80, 90]
        const rows = thresholds.map(t => {
          const ss   = withPrice.filter(s => s.score >= t)
          if (!ss.length) return null
          const hits = ss.filter(s => s.status === 'TARGET_HIT').length
          const wr   = hits / ss.length
          const sw   = ss.filter(s => s.returnPct > 0)
          const sl   = ss.filter(s => s.returnPct <= 0)
          const avgW = sw.length ? sw.reduce((a, b) => a + b.returnPct, 0) / sw.length : 0
          const avgL = sl.length ? sl.reduce((a, b) => a + b.returnPct, 0) / sl.length : 0
          const sEv  = (wr * avgW) + ((1 - wr) * avgL)
          const avgR = ss.reduce((a, b) => a + b.returnPct, 0) / ss.length
          return { t, count: ss.length, hits, wr: Math.round(wr * 100), avgR, ev: sEv }
        }).filter(Boolean) as { t: number; count: number; hits: number; wr: number; avgR: number; ev: number }[]
        if (rows.length < 2) return null
        return (
          <div className="card" style={{ marginBottom: 'var(--gap)' }}>
            <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
              <div className="card-title">Score Cutoff Analysis</div>
              <div className="card-sub">At what minimum score does the system produce positive EV? Use this to filter which signals to act on.</div>
            </div>
            <div className="tbl-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Score ≥</th>
                    <th className="num">Signals</th>
                    <th className="num">Target Hits</th>
                    <th className="num">Hit Rate</th>
                    <th className="num">Avg Return %</th>
                    <th className="num">Expected Value</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(r => (
                    <tr key={r.t} style={r.ev > 0 ? { background: 'var(--surface-2)' } : {}}>
                      <td><b style={{ fontFamily: 'var(--font-mono)' }}>{r.t}</b>{r.ev > 0 && <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--pos)' }}>✓ +EV</span>}</td>
                      <td className="num">{r.count}</td>
                      <td className="num">{r.hits}</td>
                      <td className={`num ${r.wr >= 40 ? 'pos' : 'neg'}`}>{r.wr}%</td>
                      <td className={`num ${r.avgR >= 0 ? 'pos' : 'neg'}`}>{r.avgR >= 0 ? '+' : ''}{r.avgR.toFixed(2)}%</td>
                      <td className={`num ${r.ev >= 0 ? 'pos' : 'neg'}`}>{r.ev >= 0 ? '+' : ''}{r.ev.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      })()}

      {/* Market condition breakdown */}
      {data && marketBreakdown.length > 0 && (
        <div className="card" style={{ marginBottom: 'var(--gap)' }}>
          <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
            <div className="card-title">Market Condition Correlation</div>
            <div className="card-sub">Do signals perform better in bullish vs neutral vs bearish Nifty markets?</div>
          </div>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Market Trend</th>
                  <th className="num">Signals</th>
                  <th className="num">Target Hit Rate</th>
                  <th className="num">Avg Return %</th>
                  <th className="num">Expected Value</th>
                </tr>
              </thead>
              <tbody>
                {marketBreakdown.map(m => {
                  const trendCls = m.trend.toLowerCase().includes('bull') ? 'pos'
                    : m.trend.toLowerCase().includes('bear') ? 'neg' : ''
                  return (
                    <tr key={m.trend}>
                      <td><span className={trendCls}>{m.trend.replace(/_/g, ' ')}</span></td>
                      <td className="num">{m.count}</td>
                      <td className={`num ${m.winRate >= 40 ? 'pos' : 'neg'}`}>{m.winRate}%</td>
                      <td className={`num ${m.avgRet >= 0 ? 'pos' : 'neg'}`}>{m.avgRet >= 0 ? '+' : ''}{m.avgRet.toFixed(2)}%</td>
                      <td className={`num ${m.ev >= 0 ? 'pos' : 'neg'}`}>{m.ev >= 0 ? '+' : ''}{m.ev.toFixed(2)}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Wyckoff phase breakdown */}
      {data && dbSignals.length > 0 && (() => {
        const phaseMap = new Map<string, DbSignal[]>()
        dbSignals.forEach(s => {
          const phase = s.wyckoff_phase || 'UNKNOWN'
          if (!phaseMap.has(phase)) phaseMap.set(phase, [])
          phaseMap.get(phase)!.push(s)
        })
        // Find matching batch signals for return data
        const batchBySymbol = new Map<string, BatchSignal>()
        signals.forEach(s => batchBySymbol.set(s.symbol, s))
        const phases = [...phaseMap.entries()].map(([phase, ss]) => {
          const withRet = ss.map(s => batchBySymbol.get(s.symbol.replace('.NS', ''))).filter(Boolean) as BatchSignal[]
          const hits = withRet.filter(s => s.status === 'TARGET_HIT').length
          const wr   = withRet.length ? hits / withRet.length : 0
          const avgR = withRet.length ? withRet.reduce((a, b) => a + b.returnPct, 0) / withRet.length : 0
          return { phase, total: ss.length, matched: withRet.length, wr: Math.round(wr * 100), avgR }
        }).sort((a, b) => b.total - a.total)
        if (phases.length === 0) return null
        return (
          <div className="card" style={{ marginBottom: 'var(--gap)' }}>
            <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
              <div className="card-title">Wyckoff Phase Distribution</div>
              <div className="card-sub">Which accumulation phases generated the most signals? From DB scan data.</div>
            </div>
            <div className="tbl-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Phase</th>
                    <th className="num">DB Signals</th>
                    <th className="num">With Return</th>
                    <th className="num">Target Hit Rate</th>
                    <th className="num">Avg Return %</th>
                  </tr>
                </thead>
                <tbody>
                  {phases.map(p => (
                    <tr key={p.phase}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{p.phase.replace(/_/g, ' ')}</td>
                      <td className="num">{p.total}</td>
                      <td className="num">{p.matched}</td>
                      <td className={`num ${p.wr >= 40 ? 'pos' : p.matched > 0 ? 'neg' : ''}`}>
                        {p.matched > 0 ? `${p.wr}%` : '—'}
                      </td>
                      <td className={`num ${p.avgR >= 0 ? 'pos' : 'neg'}`}>
                        {p.matched > 0 ? `${p.avgR >= 0 ? '+' : ''}${p.avgR.toFixed(2)}%` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      })()}

      {/* All signals table */}
      <div className="card card-flush">
        <div className="card-inner-pad" style={{ paddingBottom: 10 }}>
          <div className="card-title">All Signals — Aggregated</div>
          <div className="card-sub">{data ? dateRangeLabel : 'Select universe & click Load Analytics'}</div>
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
                  {['Date', 'Symbol', 'Grade', 'Return %', 'Days', 'Status', 'Vol Ratio', 'Market'].map(h => (
                    <th key={h} className={['Return %', 'Days', 'Vol Ratio'].includes(h) ? 'num' : ''}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((s: BatchSignal, i) => {
                  const ret       = s.returnPct
                  const retCls    = ret >= 0 ? 'pos' : 'neg'
                  const norm      = normalizeStatus(s.status)
                  const statusKey = norm.toLowerCase().replace(/_/g, '-')
                  const trendCls  = s.marketTrend?.toLowerCase().includes('bull') ? 'pos'
                    : s.marketTrend?.toLowerCase().includes('bear') ? 'neg' : ''
                  return (
                    <tr key={i}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)' }}>{s.date}</td>
                      <td><b>{s.symbol}</b></td>
                      <td><Tag variant={s.grade}>{s.grade || '—'}</Tag></td>
                      <td className={`num ${retCls}`}>{ret != null ? `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%` : '—'}</td>
                      <td className="num">{s.daysHeld}</td>
                      <td><Tag variant={statusKey}>{norm.replace(/_/g, ' ')}</Tag></td>
                      <td className={`num ${s.volumeRatio >= 1.5 ? 'pos' : ''}`}>{s.volumeRatio ? s.volumeRatio.toFixed(2) : '—'}</td>
                      <td><span className={trendCls}>{s.marketTrend || '—'}</span></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="tbl-foot">
              <span>{sorted.length} signals total · sorted by return %</span>
              <span className="muted">Live prices via yfinance</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
