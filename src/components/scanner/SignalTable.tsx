import { useState } from 'react'
import Tag from '../ui/Tag'
import SearchBar from '../ui/SearchBar'
import EmptyState from '../ui/EmptyState'
import { useSort } from '../../hooks/useSort'
import type { ResultTab } from '../../types'

// Numeric columns (right-align + mono font)
const NUM_COLS = new Set([
  'Score','Current_Price','Entry_Low','Entry_High','Stop_Loss','Target_1',
  'Risk_%','Volume_Ratio','RS_Score','Total_Score','Risk_Reward','Base_Quality','Stop_Price','VWAP',
])

// Preferred column order per tab
const COL_ORDER: Record<ResultTab, string[]> = {
  buy:       ['Symbol','Grade','Score','Current_Price','Entry_Low','Entry_High','Stop_Loss','Target_1','Risk_%','Volume_Ratio','Market_Trend'],
  wait:      ['Symbol','Grade','Score','Current_Price','Stop_Loss','Target_1','Risk_%','Volume_Ratio','Reason','Market_Trend'],
  watchlist: ['Symbol','Stage','Total_Score','Current_Price','Stop_Price','Risk_Reward','RS_Score','Base_Quality'],
}

interface RowData extends Record<string, unknown> {
  metrics?: string[]
}

interface Props {
  data: RowData[]
  tab: ResultTab
  footer?: string
}

function cellContent(col: string, val: unknown): React.ReactNode {
  const str = String(val ?? '—')

  if (col === 'Grade')
    return <Tag variant={str}>{str}</Tag>

  if (col === 'Stage') {
    const cls = str.includes('BREAKOUT') ? 'breakout' : 'accum'
    return <Tag variant={cls}>{str.replace('_', ' ')}</Tag>
  }

  if (col === 'Market_Trend') {
    const lower = str.toLowerCase()
    const cls = lower.includes('bull') ? 'pos' : lower.includes('bear') ? 'neg' : ''
    return <span className={cls}>{str}</span>
  }

  if (NUM_COLS.has(col) && val !== undefined && val !== '') {
    const num = parseFloat(str)
    if (isNaN(num)) return str
    let cls = ''
    if (col === 'Risk_%')       cls = 'neg'
    if (col === 'Volume_Ratio') cls = num >= 1.5 ? 'pos' : num < 0.8 ? 'neg' : ''
    if (col === 'RS_Score')     cls = num >= 70  ? 'pos' : num < 40  ? 'neg' : ''
    return <span className={cls}>{num.toFixed(2)}</span>
  }

  return str
}

export default function SignalTable({ data, tab, footer }: Props) {
  const { sorted, search, setSearch, sortKey, sortDir, toggleSort } = useSort(data)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  if (!data.length) {
    return (
      <EmptyState
        icon={<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>}
        message={<>No <b>{tab.toUpperCase()}</b> signals found.</>}
      />
    )
  }

  const allKeys = Object.keys(data[0]).filter(k => k !== 'metrics' && k !== '_scanDate')
  const cols = COL_ORDER[tab].filter(c => allKeys.includes(c))

  const toggleExpand = (i: number) => {
    setExpanded(prev => {
      const next = new Set(prev)
      next.has(i) ? next.delete(i) : next.add(i)
      return next
    })
  }

  return (
    <>
      <div className="tbl-toolbar">
        <SearchBar value={search} onChange={setSearch} />
        <span className="tbl-toolbar-right">{sorted.length} rows</span>
      </div>

      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: 32 }} />
              {cols.map(col => (
                <th
                  key={col}
                  className={`${NUM_COLS.has(col) ? 'num' : ''} ${sortKey === col ? 'sorted' : ''}`}
                  onClick={() => toggleSort(col as keyof RowData)}
                >
                  {col.replace(/_/g, ' ')}
                  <span className="sort-arrow">
                    {sortKey === col ? (sortDir > 0 ? '▲' : '▼') : '↕'}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row, i) => {
              const hasMetrics = Array.isArray(row.metrics) && row.metrics.length > 0
              const isExpanded = expanded.has(i)
              return (
                <>
                  <tr key={i} className={isExpanded ? 'expanded' : ''}>
                    <td>
                      <button
                        className={`expand-btn ${isExpanded ? 'open' : ''}`}
                        style={{ visibility: hasMetrics ? 'visible' : 'hidden' }}
                        onClick={() => toggleExpand(i)}
                      >
                        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.8">
                          <path d="M3 1l4 4-4 4" />
                        </svg>
                      </button>
                    </td>
                    {cols.map(col => (
                      <td key={col} className={NUM_COLS.has(col) ? 'num' : ''}>
                        {cellContent(col, row[col])}
                      </td>
                    ))}
                  </tr>
                  {isExpanded && hasMetrics && (
                    <tr key={`${i}-metrics`}>
                      <td colSpan={cols.length + 1} className="metrics-cell">
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted)', letterSpacing: '.06em', textTransform: 'uppercase', marginBottom: 6 }}>
                          Signal Metrics
                        </div>
                        <div className="metrics-grid">
                          {(row.metrics as string[]).map((m, mi) => (
                            <span key={mi} className="metric-pill">{m}</span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              )
            })}
          </tbody>
        </table>
        <div className="tbl-foot">
          <span>{sorted.length} signal{sorted.length !== 1 ? 's' : ''}</span>
          <span className="muted">{footer ?? 'Click column header to sort'}</span>
        </div>
      </div>
    </>
  )
}
