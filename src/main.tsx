import { StrictMode } from 'react'
import { createRoot }  from 'react-dom/client'
import './styles/tokens.css'
import './styles/base.css'
import './styles/components.css'
import App from './App'

// Google Fonts
const link = document.createElement('link')
link.href = 'https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap'
link.rel = 'stylesheet'
document.head.appendChild(link)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
