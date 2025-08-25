import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite dev server proxy forwards API/video requests to Flask on :5000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
    proxy: {
      '/api': 'http://localhost:5000',
      '/outputs': 'http://localhost:5000'
    }
  }
})
