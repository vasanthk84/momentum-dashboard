interface Props {
  icon: React.ReactNode
  title?: string
  message: React.ReactNode
  small?: boolean
}

export default function EmptyState({ icon, title, message, small }: Props) {
  return (
    <div className={`empty-state ${small ? 'empty-state-sm' : ''}`}>
      {icon}
      {title && <><b>{title}</b><br /></>}
      <span>{message}</span>
    </div>
  )
}
