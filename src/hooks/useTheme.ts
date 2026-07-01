import { useEffect, useState } from 'react'
import type { Mood } from '../types'

const STORAGE_KEY = 'ms-mood'

export function useTheme() {
  const [mood, setMoodState] = useState<Mood>(
    () => (localStorage.getItem(STORAGE_KEY) as Mood) ?? 'bloomberg'
  )

  useEffect(() => {
    document.body.setAttribute('data-mood', mood)
    localStorage.setItem(STORAGE_KEY, mood)
  }, [mood])

  const setMood = (m: Mood) => setMoodState(m)
  return { mood, setMood }
}
