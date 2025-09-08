// Modern theme system with dark/light mode and glassmorphism
export const lightTheme = {
  name: 'light',
  
  // Background colors
  background: {
    primary: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    secondary: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    surface: 'rgba(255, 255, 255, 0.95)',
    glass: 'rgba(255, 255, 255, 0.1)',
    overlay: 'rgba(0, 0, 0, 0.5)',
  },
  
  // Text colors
  text: {
    primary: '#1a1a1a',
    secondary: '#6b7280',
    accent: '#4c1d95',
    inverse: '#ffffff',
    muted: '#9ca3af',
  },
  
  // Brand colors
  brand: {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    tertiary: '#06b6d4',
    gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%)',
  },
  
  // Semantic colors
  semantic: {
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
    honest: '#10b981',
    liar: '#ef4444',
  },
  
  // Interactive states
  interactive: {
    hover: 'rgba(99, 102, 241, 0.1)',
    active: 'rgba(99, 102, 241, 0.2)',
    disabled: 'rgba(156, 163, 175, 0.5)',
    focus: 'rgba(99, 102, 241, 0.3)',
  },
  
  // Game-specific colors
  game: {
    unrevealed: 'rgba(255, 255, 255, 0.3)',
    revealed: 'rgba(255, 255, 255, 0.9)',
    humanMove: '#10b981',
    aiMove: '#ef4444',
    hover: 'rgba(251, 191, 36, 0.8)',
    grid: 'rgba(255, 255, 255, 0.1)',
  },
  
  // Shadows and effects
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    glass: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
    neon: '0 0 20px rgba(99, 102, 241, 0.5)',
  },
  
  // Border radius
  radius: {
    sm: '0.375rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
    full: '9999px',
  },
  
  // Blur effects
  blur: {
    sm: 'blur(4px)',
    md: 'blur(8px)',
    lg: 'blur(16px)',
    xl: 'blur(24px)',
  },
};

export const darkTheme = {
  name: 'dark',
  
  // Background colors
  background: {
    primary: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 100%)',
    secondary: 'linear-gradient(135deg, #581c87 0%, #7c3aed 100%)',
    surface: 'rgba(17, 24, 39, 0.95)',
    glass: 'rgba(17, 24, 39, 0.3)',
    overlay: 'rgba(0, 0, 0, 0.7)',
  },
  
  // Text colors
  text: {
    primary: '#f9fafb',
    secondary: '#d1d5db',
    accent: '#c7d2fe',
    inverse: '#111827',
    muted: '#6b7280',
  },
  
  // Brand colors (same as light for consistency)
  brand: {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    tertiary: '#06b6d4',
    gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%)',
  },
  
  // Semantic colors (slightly adjusted for dark mode)
  semantic: {
    success: '#34d399',
    warning: '#fbbf24',
    error: '#f87171',
    info: '#60a5fa',
    honest: '#34d399',
    liar: '#f87171',
  },
  
  // Interactive states
  interactive: {
    hover: 'rgba(99, 102, 241, 0.2)',
    active: 'rgba(99, 102, 241, 0.3)',
    disabled: 'rgba(75, 85, 99, 0.5)',
    focus: 'rgba(99, 102, 241, 0.4)',
  },
  
  // Game-specific colors
  game: {
    unrevealed: 'rgba(17, 24, 39, 0.6)',
    revealed: 'rgba(249, 250, 251, 0.9)',
    humanMove: '#34d399',
    aiMove: '#f87171',
    hover: 'rgba(251, 191, 36, 0.9)',
    grid: 'rgba(249, 250, 251, 0.1)',
  },
  
  // Shadows and effects (enhanced for dark mode)
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2)',
    glass: '0 8px 32px 0 rgba(0, 0, 0, 0.5)',
    neon: '0 0 30px rgba(99, 102, 241, 0.7)',
  },
  
  // Border radius (same as light)
  radius: {
    sm: '0.375rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
    full: '9999px',
  },
  
  // Blur effects (same as light)
  blur: {
    sm: 'blur(4px)',
    md: 'blur(8px)',
    lg: 'blur(16px)',
    xl: 'blur(24px)',
  },
};

// Typography system
export const typography = {
  fontFamily: {
    sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
    mono: ['Fira Code', 'Monaco', 'Consolas', 'Liberation Mono', 'Courier New', 'monospace'],
  },
  
  fontSize: {
    xs: '0.75rem',
    sm: '0.875rem',
    base: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
    '2xl': '1.5rem',
    '3xl': '1.875rem',
    '4xl': '2.25rem',
    '5xl': '3rem',
    '6xl': '3.75rem',
  },
  
  fontWeight: {
    light: 300,
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
    extrabold: 800,
  },
  
  lineHeight: {
    none: 1,
    tight: 1.25,
    normal: 1.5,
    relaxed: 1.625,
    loose: 2,
  },
  
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0em',
    wide: '0.025em',
    wider: '0.05em',
  },
};

// Animation presets
export const animations = {
  // Durations
  duration: {
    fast: '150ms',
    normal: '300ms',
    slow: '500ms',
    slower: '800ms',
  },
  
  // Easing functions
  easing: {
    linear: 'linear',
    ease: 'ease',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
    spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
  },
  
  // Keyframes
  keyframes: {
    fadeIn: {
      '0%': { opacity: 0 },
      '100%': { opacity: 1 },
    },
    
    slideUp: {
      '0%': { transform: 'translateY(20px)', opacity: 0 },
      '100%': { transform: 'translateY(0)', opacity: 1 },
    },
    
    slideDown: {
      '0%': { transform: 'translateY(-20px)', opacity: 0 },
      '100%': { transform: 'translateY(0)', opacity: 1 },
    },
    
    scaleIn: {
      '0%': { transform: 'scale(0.95)', opacity: 0 },
      '100%': { transform: 'scale(1)', opacity: 1 },
    },
    
    pulse: {
      '0%, 100%': { opacity: 1 },
      '50%': { opacity: 0.5 },
    },
    
    glow: {
      '0%, 100%': { boxShadow: '0 0 20px rgba(99, 102, 241, 0.5)' },
      '50%': { boxShadow: '0 0 30px rgba(99, 102, 241, 0.8)' },
    },
    
    pixelReveal: {
      '0%': { 
        transform: 'scale(0.8)', 
        opacity: 0,
        filter: 'blur(4px)'
      },
      '50%': { 
        transform: 'scale(1.1)', 
        opacity: 0.8,
        filter: 'blur(0px)'
      },
      '100%': { 
        transform: 'scale(1)', 
        opacity: 1,
        filter: 'blur(0px)'
      },
    },
    
    float: {
      '0%, 100%': { transform: 'translateY(0px)' },
      '50%': { transform: 'translateY(-10px)' },
    },
  },
};

// Responsive breakpoints
export const breakpoints = {
  xs: '375px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
};

// Z-index layers
export const zIndex = {
  hide: -1,
  base: 0,
  dropdown: 1000,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  tooltip: 1600,
  toast: 1700,
};

// Spacing scale (based on 4px grid)
export const spacing = {
  0: '0',
  1: '0.25rem',  // 4px
  2: '0.5rem',   // 8px
  3: '0.75rem',  // 12px
  4: '1rem',     // 16px
  5: '1.25rem',  // 20px
  6: '1.5rem',   // 24px
  8: '2rem',     // 32px
  10: '2.5rem',  // 40px
  12: '3rem',    // 48px
  16: '4rem',    // 64px
  20: '5rem',    // 80px
  24: '6rem',    // 96px
  32: '8rem',    // 128px
};

export const getTheme = (isDark = false) => isDark ? darkTheme : lightTheme;