import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('error', (err) => console.error('[proxy error]', err.message))
          proxy.on('proxyReq', (_req, req) =>
            console.log('[proxy]', req.method, req.url)
          )
        },
      },
    },
  },
})
