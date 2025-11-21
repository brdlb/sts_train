/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      animation: {
        'pulse-highlight': 'pulse-highlight 1.5s infinite',
      },
      keyframes: {
        'pulse-highlight': {
          '0%, 100%': {
            boxShadow: '0 0 0 0 rgba(250, 204, 21, 0.7)',
            transform: 'scale(1.02)',
          },
          '50%': {
            boxShadow: '0 0 0 8px rgba(250, 204, 21, 0)',
            transform: 'scale(1.08)',
          },
        },
      },
    },
  },
  plugins: [],
}

