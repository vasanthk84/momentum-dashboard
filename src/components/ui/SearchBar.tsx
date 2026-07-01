interface Props {
  value: string
  onChange: (v: string) => void
  placeholder?: string
}

export default function SearchBar({ value, onChange, placeholder = 'Search symbol…' }: Props) {
  return (
    <div className="search-wrap">
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6">
        <circle cx="7" cy="7" r="4.5" /><path d="M11 11L14 14" strokeLinecap="round" />
      </svg>
      <input
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  )
}
