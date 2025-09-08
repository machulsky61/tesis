import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from '../../theme/ThemeProvider';

const ToggleContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const ToggleButton = styled(motion.button)`
  position: relative;
  width: 3.5rem;
  height: 1.75rem;
  background: ${props => props.isDark ? props.theme.brand.primary : props.theme.background.glass};
  border: 2px solid ${props => props.isDark ? props.theme.brand.primary : props.theme.text.secondary}40;
  border-radius: ${props => props.theme.radius.full};
  cursor: pointer;
  transition: all 300ms cubic-bezier(0.175, 0.885, 0.32, 1.275);
  overflow: hidden;

  &:hover {
    box-shadow: ${props => props.theme.shadows.md};
    transform: scale(1.02);
  }

  &:focus-visible {
    outline: 2px solid ${props => props.theme.brand.primary};
    outline-offset: 2px;
  }
`;

const ToggleThumb = styled(motion.div)`
  position: absolute;
  top: 0.125rem;
  left: ${props => props.isDark ? '1.625rem' : '0.125rem'};
  width: 1.25rem;
  height: 1.25rem;
  background: ${props => props.theme.text.inverse};
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: ${props => props.theme.shadows.sm};
  transition: all 300ms cubic-bezier(0.175, 0.885, 0.32, 1.275);

  svg {
    width: 0.75rem;
    height: 0.75rem;
    color: ${props => props.isDark ? props.theme.brand.primary : '#f59e0b'};
  }
`;

const IconWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${props => props.theme.text.secondary};
  opacity: 0.7;
  transition: opacity 200ms ease;

  &.active {
    opacity: 1;
    color: ${props => props.theme.brand.primary};
  }

  svg {
    width: 1rem;
    height: 1rem;
  }
`;

const Label = styled.span`
  font-size: 0.875rem;
  font-weight: 500;
  color: ${props => props.theme.text.secondary};
  margin: 0 0.25rem;
  
  @media (max-width: 640px) {
    display: none;
  }
`;

const ThemeToggle = ({ showLabel = false, compact = false }) => {
  const { isDark, toggleTheme } = useTheme();

  const handleToggle = () => {
    toggleTheme();
  };

  if (compact) {
    return (
      <ToggleButton
        onClick={handleToggle}
        isDark={isDark}
        whileTap={{ scale: 0.95 }}
        title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        <ToggleThumb
          isDark={isDark}
          initial={false}
          animate={{
            x: isDark ? 24 : 0,
            rotate: isDark ? 180 : 0
          }}
          transition={{
            type: 'spring',
            stiffness: 500,
            damping: 30
          }}
        >
          {isDark ? <Moon /> : <Sun />}
        </ToggleThumb>
      </ToggleButton>
    );
  }

  return (
    <ToggleContainer>
      {showLabel && <Label>Light</Label>}
      
      <IconWrapper className={!isDark ? 'active' : ''}>
        <Sun />
      </IconWrapper>

      <ToggleButton
        onClick={handleToggle}
        isDark={isDark}
        whileTap={{ scale: 0.95 }}
        title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        <ToggleThumb
          isDark={isDark}
          initial={false}
          animate={{
            x: isDark ? 24 : 0,
            rotate: isDark ? 180 : 0
          }}
          transition={{
            type: 'spring',
            stiffness: 500,
            damping: 30
          }}
        >
          <motion.div
            initial={false}
            animate={{ 
              rotate: isDark ? 0 : 180,
              scale: isDark ? 1 : 0 
            }}
            transition={{ duration: 0.2 }}
          >
            {isDark ? <Moon /> : <Sun />}
          </motion.div>
        </ToggleThumb>
      </ToggleButton>

      <IconWrapper className={isDark ? 'active' : ''}>
        <Moon />
      </IconWrapper>
      
      {showLabel && <Label>Dark</Label>}
    </ToggleContainer>
  );
};

export default ThemeToggle;