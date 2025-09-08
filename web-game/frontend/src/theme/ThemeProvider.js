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
  // Initialize theme from localStorage or default to dark (more modern)
  const [isDark, setIsDark] = useState(() => {
    try {
      const saved = localStorage.getItem('debate-game-theme');
      return saved ? JSON.parse(saved) : true; // Default to dark theme
    } catch {
      return true;
    }
  });

  const theme = getTheme(isDark);

  // Save theme preference
  useEffect(() => {
    localStorage.setItem('debate-game-theme', JSON.stringify(isDark));
  }, [isDark]);

  // Apply theme to document root for global styles
  useEffect(() => {
    const root = document.documentElement;
    
    // Set CSS custom properties for global access
    root.style.setProperty('--bg-primary', theme.background.primary);
    root.style.setProperty('--bg-surface', theme.background.surface);
    root.style.setProperty('--text-primary', theme.text.primary);
    root.style.setProperty('--text-secondary', theme.text.secondary);
    root.style.setProperty('--brand-primary', theme.brand.primary);
    
    // Update meta theme-color for mobile browsers
    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (metaTheme) {
      metaTheme.setAttribute('content', isDark ? '#1e1b4b' : '#667eea');
    }
    
    // Update body background
    document.body.style.background = theme.background.primary;
    document.body.style.color = theme.text.primary;
  }, [theme, isDark]);

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