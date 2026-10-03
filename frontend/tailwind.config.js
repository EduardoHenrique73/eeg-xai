/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        clinical: {
          50: '#f7faff',
          100: '#eef4fb',
          200: '#dce7f2',
          300: '#b9cadd',
          500: '#61758b',
          700: '#34485f',
          800: '#23384f',
          900: '#122a43',
        },
        accent: {
          DEFAULT: '#1d5fc1',
          dark: '#174c99',
          light: '#e8f2ff',
        },
        alert: {
          crisis: '#dc2626',
          normal: '#16a34a',
        },
      },
      fontFamily: {
        sans: ['"Segoe UI"', 'system-ui', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'Consolas', 'monospace'],
      },
      boxShadow: {
        clinical: '0 1px 3px 0 rgb(18 42 67 / 0.06), 0 8px 24px -18px rgb(18 42 67 / 0.18)',
      },
      keyframes: {
        'toast-in': {
          '0%': { opacity: '0', transform: 'translateY(12px) scale(0.96)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        'modal-in': {
          '0%': { opacity: '0', transform: 'scale(0.94) translateY(8px)' },
          '100%': { opacity: '1', transform: 'scale(1) translateY(0)' },
        },
        'backdrop-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'shake': {
          '0%, 100%': { transform: 'translateX(0)' },
          '15%': { transform: 'translateX(-6px)' },
          '45%': { transform: 'translateX(6px)' },
          '75%': { transform: 'translateX(-4px)' },
        },
        'pulse-bar': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'spin-slow': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
      },
      animation: {
        'toast-in': 'toast-in 0.22s cubic-bezier(0.16, 1, 0.3, 1) both',
        'modal-in': 'modal-in 0.22s cubic-bezier(0.16, 1, 0.3, 1) both',
        'backdrop-in': 'backdrop-in 0.18s ease both',
        'slide-up': 'slide-up 0.35s cubic-bezier(0.16, 1, 0.3, 1) both',
        'shake': 'shake 0.4s ease both',
        'pulse-bar': 'pulse-bar 1.6s ease-in-out infinite',
        'fade-in': 'fade-in 0.25s ease both',
        'spin-slow': 'spin-slow 0.8s linear infinite',
      },
    },
  },
  plugins: [],
}
