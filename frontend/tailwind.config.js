/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        clinical: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          500: '#64748b',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
        accent: {
          DEFAULT: '#0d9488',
          dark: '#0f766e',
          light: '#ccfbf1',
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
        clinical: '0 1px 3px 0 rgb(15 23 42 / 0.08), 0 1px 2px -1px rgb(15 23 42 / 0.06)',
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
