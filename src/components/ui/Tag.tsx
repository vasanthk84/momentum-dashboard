interface Props {
  variant: string
  children: React.ReactNode
}

export default function Tag({ variant, children }: Props) {
  return <span className={`tag ${variant.toLowerCase().replace(/[_\s]/g, '-')}`}>{children}</span>
}
