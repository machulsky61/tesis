import React, { createContext, useContext, useState, useEffect } from 'react';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';
import { getTheme } from './theme';

const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  // Initialize theme from localStorage or system preference
  const [isDark, setIsDark] = useState(() => {
    try {
      const saved = localStorage.getItem('debate-game-theme');
      if (saved !== null) {
        return JSON.parse(saved);
      }
      // Default to system preference
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    } catch {
      return false; // Default to light theme for better consistency
    }
  });

  // Track if theme is being initialized to prevent flash
  const [isInitialized, setIsInitialized] = useState(false);

  const theme = getTheme(isDark);

  // Save theme preference
  useEffect(() => {
    localStorage.setItem('debate-game-theme', JSON.stringify(isDark));
  }, [isDark]);

  // Apply theme to document root for global styles
  useEffect(() => {
    const root = document.documentElement;
    
    // Add/remove dark class for portfolio compatibility
    if (isDark) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    
    // Set CSS custom properties for global access
    root.style.setProperty('--bg-primary', theme.background.primary);
    root.style.setProperty('--bg-surface', theme.background.surface);
    root.style.setProperty('--text-primary', theme.text.primary);
    root.style.setProperty('--text-secondary', theme.text.secondary);
    root.style.setProperty('--brand-primary', theme.brand.primary);
    
    // Update meta theme-color for mobile browsers (use theme colors)
    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (metaTheme) {
      metaTheme.setAttribute('content', theme.background.primary);
    }
    
    // Update body background immediately to prevent flash
    document.body.style.background = theme.background.primary;
    document.body.style.color = theme.text.primary;
    document.body.style.transition = isInitialized ? 'background-color 0.3s ease, color 0.3s ease' : 'none';
    
    // Mark as initialized after first render
    if (!isInitialized) {
      setIsInitialized(true);
    }
  }, [theme, isDark, isInitialized]);

  const toggleTheme = () => setIsDark(prev => !prev);

  const contextValue = {
    theme,
    isDark,
    toggleTheme,
    themeName: theme.name,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      <StyledThemeProvider theme={theme}>
        {children}
      </StyledThemeProvider>
    </ThemeContext.Provider>
  );
};

export default ThemeProvider;