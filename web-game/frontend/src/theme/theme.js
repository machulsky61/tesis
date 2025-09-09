// Professional theme system following portfolio design with OKLCH colors
export const lightTheme = {
  name: 'light',
  
  // Background colors - Portfolio OKLCH system
  background: {
    primary: 'oklch(1 0 0)', // Pure white #ffffff
    surface: 'oklch(0.97 0 0)', // Very light gray cards #f8f8f8  
    muted: 'oklch(0.97 0 0)', // Soft gray #f8f8f8
    glass: 'rgba(255, 255, 255, 0.8)',
    overlay: 'rgba(0, 0, 0, 0.5)',
    border: 'oklch(0.922 0 0)', // Light gray borders #e5e5e5
    card: 'oklch(1 0 0)', // Card background white
    popover: 'oklch(1 0 0)', // Popover background
    secondary: 'oklch(0.97 0 0)', // Secondary elements
    accent: 'oklch(0.97 0 0)', // Accent background
    input: 'oklch(0.922 0 0)', // Input background
  },
  
  // Text colors - Portfolio OKLCH system
  text: {
    primary: 'oklch(0.145 0 0)', // Dark text #171717
    secondary: 'oklch(0.556 0 0)', // Secondary text #737373
    muted: 'oklch(0.556 0 0)', // Muted text #737373
    inverse: 'oklch(0.985 0 0)', // Inverse text #fafafa
    accent: 'oklch(0.205 0 0)', // Accent text
    ring: 'oklch(0.708 0 0)', // Focus rings
  },
  
  // Brand colors - AI/Research focused with portfolio consistency
  brand: {
    primary: 'oklch(0.205 0 0)', // Primary buttons #0a0a0a
    secondary: 'oklch(0.556 0.15 280)', // AI Safety purple
    tertiary: 'oklch(0.6 0.2 120)', // Success green
    gradient: 'linear-gradient(135deg, oklch(0.556 0.15 280) 0%, oklch(0.6 0.2 240) 100%)',
  },
  
  // Semantic colors - Portfolio compatible
  semantic: {
    success: 'oklch(0.6 0.2 120)', // Green success #10b981
    warning: 'oklch(0.7 0.15 60)', // Warning yellow
    error: 'oklch(0.577 0.245 27.325)', // Portfolio destructive color
    info: 'oklch(0.6 0.2 240)', // Info blue
    honest: 'oklch(0.6 0.2 120)', // Honest player green
    liar: 'oklch(0.577 0.245 27.325)', // Liar player red
  },
  
  // Interactive states - Clean portfolio approach
  interactive: {
    hover: 'oklch(0.97 0 0)', // Light hover
    active: 'oklch(0.922 0 0)', // Active state
    disabled: 'oklch(0.556 0 0 / 0.5)', // Disabled state
    focus: 'oklch(0.708 0 0)', // Focus ring color
  },
  
  // Game-specific colors - Fixed for light mode
  game: {
    unrevealed: 'oklch(0.97 0 0 / 0.8)', // Light unrevealed overlay
    revealed: 'transparent', // No overlay for revealed pixels
    humanMove: 'oklch(0.6 0.2 120)', // Green for honest
    aiMove: 'oklch(0.577 0.245 27.325)', // Red for liar
    hover: 'oklch(0.8 0.15 60 / 0.5)', // Hover highlight
    grid: 'oklch(0.922 0 0)', // Grid lines
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
  
  // Background colors - Portfolio dark mode
  background: {
    primary: 'oklch(0.145 0 0)', // Dark background #0a0a0a
    surface: 'oklch(0.205 0 0)', // Dark cards #262626
    muted: 'oklch(0.269 0 0)', // Muted background #404040
    glass: 'rgba(17, 24, 39, 0.8)',
    overlay: 'rgba(0, 0, 0, 0.7)',
    border: 'oklch(1 0 0 / 10%)', // Border transparent
    card: 'oklch(0.205 0 0)', // Card background
    popover: 'oklch(0.205 0 0)', // Popover background
    secondary: 'oklch(0.269 0 0)', // Secondary elements
    accent: 'oklch(0.269 0 0)', // Accent background
    input: 'oklch(1 0 0 / 15%)', // Input background
  },
  
  // Text colors - Portfolio dark mode
  text: {
    primary: 'oklch(0.985 0 0)', // Light text #fafafa
    secondary: 'oklch(0.708 0 0)', // Secondary text #a3a3a3
    muted: 'oklch(0.708 0 0)', // Muted text #a3a3a3
    inverse: 'oklch(0.145 0 0)', // Inverse text (dark)
    accent: 'oklch(0.985 0 0)', // Accent text
    ring: 'oklch(0.556 0 0)', // Focus rings
  },
  
  // Brand colors - Portfolio dark mode compatible
  brand: {
    primary: 'oklch(0.922 0 0)', // Primary buttons light #e5e5e5
    secondary: 'oklch(0.556 0.15 280)', // AI Safety purple
    tertiary: 'oklch(0.6 0.2 120)', // Success green
    gradient: 'linear-gradient(135deg, oklch(0.556 0.15 280) 0%, oklch(0.6 0.2 240) 100%)',
  },
  
  // Semantic colors - Portfolio dark mode
  semantic: {
    success: 'oklch(0.704 0.191 141)', // Success green for dark
    warning: 'oklch(0.7 0.15 60)', // Warning yellow
    error: 'oklch(0.704 0.191 22.216)', // Portfolio destructive dark
    info: 'oklch(0.6 0.2 240)', // Info blue
    honest: 'oklch(0.704 0.191 141)', // Honest player green
    liar: 'oklch(0.704 0.191 22.216)', // Liar player red
  },
  
  // Interactive states - Portfolio dark approach
  interactive: {
    hover: 'oklch(0.269 0 0)', // Dark hover
    active: 'oklch(0.205 0 0)', // Active state
    disabled: 'oklch(0.556 0 0 / 0.5)', // Disabled state
    focus: 'oklch(0.556 0 0)', // Focus ring color
  },
  
  // Game-specific colors - Fixed for dark mode
  game: {
    unrevealed: 'oklch(0.145 0 0 / 0.6)', // Dark unrevealed overlay
    revealed: 'transparent', // No overlay for revealed pixels
    humanMove: 'oklch(0.704 0.191 141)', // Green for honest
    aiMove: 'oklch(0.704 0.191 22.216)', // Red for liar
    hover: 'oklch(0.8 0.15 60 / 0.5)', // Hover highlight
    grid: 'oklch(1 0 0 / 10%)', // Grid lines
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

// Typography system - Portfolio compatible
export const typography = {
  fontFamily: {
    sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
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