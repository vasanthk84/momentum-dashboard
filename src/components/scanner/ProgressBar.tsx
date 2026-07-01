interface Props {
  text: string
  fromCache: boolean
}

export default function ProgressBar({ text, fromCache }: Props) {
  return (
    <div className="progress-bar">
      <svg className="progress-spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </svg>
      {fromCache
        ? <span><span className="from-cache-label">⚡ Cache hit</span> — {text}</span>
        : <span>{text}</span>
      }
    </div>
  )
}
