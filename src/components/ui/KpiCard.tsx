interface Props {
  label: string
  value: React.ReactNode
  valueClass?: string
  barPct?: number
  barClass?: string
  sub?: string
}

export default function KpiCard({ label, value, valueClass = '', barPct, barClass = 'pos', sub }: Props) {
  return (
    <div className="kpi">
      <div className="kpi-lbl">{label}</div>
      <div className={`kpi-val ${valueClass}`}>{value}</div>
      {barPct !== undefined && (
        <div className="kpi-bar">
          <span className={`kpi-bar-fill ${barClass}`} style={{ width: `${barPct}%` }} />
        </div>
      )}
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}
