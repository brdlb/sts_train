import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    allowedHosts: [
      'funereal-doily-yasmin.ngrok-free.dev',
    ],
    proxy: {
      '/api': {
        target: 'https://funereal-doily-yasmin.ngrok-free.dev:8000',
        changeOrigin: true,
      },
    },
  },
})

