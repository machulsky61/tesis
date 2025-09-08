import { createGlobalStyle } from 'styled-components';
import { typography, animations } from './theme';

export const GlobalStyles = createGlobalStyle`
  /* Import Inter font from Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* CSS Custom Properties for dynamic theming */
  :root {
    --font-sans: ${typography.fontFamily.sans.join(', ')};
    --font-mono: ${typography.fontFamily.mono.join(', ')};
    
    /* Animation durations */
    --duration-fast: ${animations.duration.fast};
    --duration-normal: ${animations.duration.normal};
    --duration-slow: ${animations.duration.slow};
    --duration-slower: ${animations.duration.slower};
    
    /* Easing functions */
    --ease-spring: ${animations.easing.spring};
    --ease-bounce: ${animations.easing.bounce};
  }

  /* Reset and base styles */
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  html {
    font-size: 16px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    scroll-behavior: smooth;
  }

  body {
    font-family: var(--font-sans);
    font-weight: ${typography.fontWeight.normal};
    background: ${props => props.theme.background.primary};
    color: ${props => props.theme.text.primary};
    transition: background-color var(--duration-normal) var(--ease-spring),
                color var(--duration-normal) var(--ease-spring);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Typography improvements */
  h1, h2, h3, h4, h5, h6 {
    font-weight: ${typography.fontWeight.bold};
    line-height: ${typography.lineHeight.tight};
    color: ${props => props.theme.text.primary};
    margin-bottom: 0.5em;
  }

  h1 {
    font-size: ${typography.fontSize['4xl']};
    font-weight: ${typography.fontWeight.extrabold};
    letter-spacing: ${typography.letterSpacing.tight};
  }

  h2 {
    font-size: ${typography.fontSize['3xl']};
    font-weight: ${typography.fontWeight.bold};
  }

  h3 {
    font-size: ${typography.fontSize['2xl']};
    font-weight: ${typography.fontWeight.semibold};
  }

  p {
    line-height: ${typography.lineHeight.relaxed};
    color: ${props => props.theme.text.secondary};
    margin-bottom: 1em;
  }

  /* Interactive elements */
  button {
    font-family: inherit;
    font-weight: ${typography.fontWeight.medium};
    cursor: pointer;
    border: none;
    outline: none;
    transition: all var(--duration-fast) var(--ease-spring);
    
    &:focus-visible {
      outline: 2px solid ${props => props.theme.brand.primary};
      outline-offset: 2px;
    }
  }

  input, select, textarea {
    font-family: inherit;
    transition: all var(--duration-fast) var(--ease-spring);
    
    &:focus {
      outline: none;
    }
  }

  /* Links */
  a {
    color: ${props => props.theme.brand.primary};
    text-decoration: none;
    transition: color var(--duration-fast) var(--ease-spring);
    
    &:hover {
      color: ${props => props.theme.brand.secondary};
      text-decoration: underline;
    }
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar {
    width: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${props => props.theme.background.surface};
  }

  ::-webkit-scrollbar-thumb {
    background: ${props => props.theme.brand.primary};
    border-radius: 4px;
    
    &:hover {
      background: ${props => props.theme.brand.secondary};
    }
  }

  /* Selection styling */
  ::selection {
    background: ${props => props.theme.brand.primary};
    color: ${props => props.theme.text.inverse};
  }

  ::-moz-selection {
    background: ${props => props.theme.brand.primary};
    color: ${props => props.theme.text.inverse};
  }

  /* Animation keyframes */
  @keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
  }

  @keyframes slideUp {
    0% { 
      transform: translateY(20px); 
      opacity: 0; 
    }
    100% { 
      transform: translateY(0); 
      opacity: 1; 
    }
  }

  @keyframes slideDown {
    0% { 
      transform: translateY(-20px); 
      opacity: 0; 
    }
    100% { 
      transform: translateY(0); 
      opacity: 1; 
    }
  }

  @keyframes scaleIn {
    0% { 
      transform: scale(0.95); 
      opacity: 0; 
    }
    100% { 
      transform: scale(1); 
      opacity: 1; 
    }
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  @keyframes glow {
    0%, 100% { 
      box-shadow: 0 0 20px ${props => props.theme.brand.primary}40; 
    }
    50% { 
      box-shadow: 0 0 30px ${props => props.theme.brand.primary}80; 
    }
  }

  @keyframes pixelReveal {
    0% { 
      transform: scale(0.8); 
      opacity: 0;
      filter: blur(4px);
    }
    50% { 
      transform: scale(1.1); 
      opacity: 0.8;
      filter: blur(0px);
    }
    100% { 
      transform: scale(1); 
      opacity: 1;
      filter: blur(0px);
    }
  }

  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }

  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  /* Utility classes */
  .fade-in {
    animation: fadeIn var(--duration-normal) var(--ease-spring);
  }

  .slide-up {
    animation: slideUp var(--duration-normal) var(--ease-spring);
  }

  .slide-down {
    animation: slideDown var(--duration-normal) var(--ease-spring);
  }

  .scale-in {
    animation: scaleIn var(--duration-normal) var(--ease-spring);
  }

  .pulse {
    animation: pulse 2s ease-in-out infinite;
  }

  .glow {
    animation: glow 2s ease-in-out infinite;
  }

  .float {
    animation: float 3s ease-in-out infinite;
  }

  /* Glassmorphism utility */
  .glass {
    background: ${props => props.theme.background.glass};
    backdrop-filter: ${props => props.theme.blur.md};
    border: 1px solid ${props => props.theme.background.glass};
    box-shadow: ${props => props.theme.shadows.glass};
  }

  /* Responsive utilities */
  .hide-mobile {
    @media (max-width: 768px) {
      display: none !important;
    }
  }

  .hide-desktop {
    @media (min-width: 769px) {
      display: none !important;
    }
  }

  /* Accessibility */
  @media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }

  /* Focus styles for keyboard navigation */
  .focus-ring {
    &:focus-visible {
      outline: 2px solid ${props => props.theme.brand.primary};
      outline-offset: 2px;
      border-radius: ${props => props.theme.radius.sm};
    }
  }

  /* Loading spinner */
  .spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid ${props => props.theme.interactive.disabled};
    border-radius: 50%;
    border-top-color: ${props => props.theme.brand.primary};
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Text selection disable for game elements */
  .no-select {
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }

  /* Custom scrollbar for specific containers */
  .custom-scroll {
    scrollbar-width: thin;
    scrollbar-color: ${props => props.theme.brand.primary} ${props => props.theme.background.surface};
    
    &::-webkit-scrollbar {
      width: 6px;
    }
    
    &::-webkit-scrollbar-track {
      background: ${props => props.theme.background.surface};
      border-radius: 3px;
    }
    
    &::-webkit-scrollbar-thumb {
      background: ${props => props.theme.brand.primary};
      border-radius: 3px;
      
      &:hover {
        background: ${props => props.theme.brand.secondary};
      }
    }
  }
`;