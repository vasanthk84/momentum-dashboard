import { useMemo, useState } from 'react'

export function useSort<T extends Record<string, unknown>>(data: T[]) {
  const [sortKey, setSortKey]   = useState<keyof T | null>(null)
  const [sortDir, setSortDir]   = useState<1 | -1>(-1)
  const [search,  setSearch]    = useState('')

  const filtered = useMemo(() => {
    if (!search) return data
    const q = search.toLowerCase()
    return data.filter(row =>
      Object.values(row).some(v => String(v).toLowerCase().includes(q))
    )
  }, [data, search])

  const sorted = useMemo(() => {
    if (!sortKey) return filtered
    return [...filtered].sort((a, b) => {
      const av = parseFloat(String(a[sortKey])) || String(a[sortKey])
      const bv = parseFloat(String(b[sortKey])) || String(b[sortKey])
      return av < bv ? -sortDir : av > bv ? sortDir : 0
    })
  }, [filtered, sortKey, sortDir])

  const toggleSort = (key: keyof T) => {
    if (sortKey === key) setSortDir(d => (d === 1 ? -1 : 1))
    else { setSortKey(key); setSortDir(-1) }
  }

  return { sorted, search, setSearch, sortKey, sortDir, toggleSort }
}
